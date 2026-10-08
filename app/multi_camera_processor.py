import cv2
import numpy as np
import os
import json
import datetime
import logging
from typing import Optional, Dict, List, Any, Set, Tuple

from app.reid.quality import calculate_quality
from app.reid.embedding_bank import TrackEmbeddingBank
from ultralytics import YOLO

from app.violation.line_crossing import LineCrossingDetector
from app.violation.attire_smoother import AttireStabilityManager
from app.violation.person_state import PersonStateManager
from app.violation.route_policy import RoutePolicy
from app.violation.evidence_manager import EvidenceManager
from app.violation.violation_engine import ViolationEngine
from app.violation.handwash_detector import HandwashDetector

from app.events.transition_detector import TransitionDetector, TransitionEvent
from app.events.roi_detector import RoiDetector, RoiEvent
from app.association.association_manager import AssociationManager
from app.journey.journey_manager import JourneyManager
from app.journey.journey_store import JourneyStore

class MultiCameraProcessor:
    def __init__(
        self,
        cameras_info,
        output_path,
        config,
        extractor,
        par_extractor,
        pose_extractor,
        gallery,
        matcher,
        attire_extractor=None,
        violation_engine=None,
        line_detectors=None,
        attire_smoother=None,
        person_state_manager=None,
        handwash_detector=None,
        association_manager=None,
        journey_manager=None,
        store=None
    ):
        """
        cameras_info: list of dicts [{"camera_id": "cctv1", "video_path": "..."}, ...]
        """
        self.cameras_info = cameras_info
        
        # Append date and time to the output path to prevent overriding
        base_name, ext = os.path.splitext(output_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"{base_name}_{timestamp}{ext}"
        
        # Ensure the directory exists
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        self.config = config
        self.extractor = extractor
        self.par_extractor = par_extractor
        self.pose_extractor = pose_extractor
        self.attire_extractor = attire_extractor
        self.gallery = gallery
        self.matcher = matcher
        
        # Initialize Violation Detection Layer
        self.violation_config = config.get("violation_detection", {})
        self.violation_enabled = bool(self.violation_config.get("enabled", False))
        
        if self.violation_enabled:
            cam_cfgs = self.violation_config.get("cameras", {})
            self.line_detectors = line_detectors or {}
            for cam in cameras_info:
                c_id = cam["camera_id"]
                if c_id not in self.line_detectors and c_id in cam_cfgs:
                    c_info = cam_cfgs[c_id]
                    line_cfg = c_info.get("line", {})
                    enabled = bool(line_cfg.get("enabled", False))
                    start = line_cfg.get("start", [0, 0])
                    end = line_cfg.get("end", [0, 0])
                    color = tuple(line_cfg.get("color", [0, 0, 255]))
                    self.line_detectors[c_id] = LineCrossingDetector(
                        camera_id=c_id,
                        line_start=tuple(start),
                        line_end=tuple(end),
                        enabled=enabled,
                        color=color
                    )
            
            attire_cfg = self.violation_config.get("attire", {})
            self.attire_smoother = attire_smoother or AttireStabilityManager(
                history_size=attire_cfg.get("history_size", 10),
                minimum_confidence=attire_cfg.get("minimum_confidence", 0.70),
                prediction_interval=attire_cfg.get("prediction_interval", 5)
            )
            
            self.person_state_manager = person_state_manager or PersonStateManager(camera_configs=cam_cfgs)
            self.route_policy = RoutePolicy(camera_configs=cam_cfgs)
            
            ev_cfg = self.violation_config.get("evidence", {})
            self.evidence_manager = EvidenceManager(
                output_dir=ev_cfg.get("output_dir", "output/violations"),
                enabled=bool(ev_cfg.get("enabled", True)),
                save_image=bool(ev_cfg.get("save_image", True)),
                save_metadata=bool(ev_cfg.get("save_metadata", True))
            )
            
            self.violation_engine = violation_engine or ViolationEngine(
                config=self.violation_config,
                route_policy=self.route_policy,
                evidence_manager=self.evidence_manager,
                person_state_manager=self.person_state_manager
            )
        else:
            self.line_detectors = {}
            self.attire_smoother = None
            self.person_state_manager = None
            self.route_policy = None
            self.evidence_manager = None
            self.violation_engine = None
        
        # Initialize Handwash Detection Layer
        hw_cfg = config.get("handwashing", {})
        if handwash_detector is not None:
            self.handwash_detector = handwash_detector
        elif hw_cfg.get("enabled", True):
            try:
                import torch
                dev = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                dev = "cpu"
            self.handwash_detector = HandwashDetector(config=hw_cfg, device=dev)
        else:
            self.handwash_detector = None
        
        self.association_manager = association_manager
        self.journey_manager = journey_manager
        self.store = store
        self.phase1_mode = bool(config.get("phase1_mode", True) or (association_manager is not None))

        # Initialize Phase 1 Event Detectors (Transition & ROI)
        self.transition_detectors = {}
        self.roi_detectors = {}
        cam_cfgs = self.config.get("cameras", {})
        for cam in cameras_info:
            c_id = cam["camera_id"]
            c_cfg = cam_cfgs.get(c_id, {})
            c_zone = c_cfg.get("zone", c_id)
            self.transition_detectors[c_id] = TransitionDetector(
                camera_id=c_id,
                zone=c_zone,
                transitions_config=c_cfg.get("transitions", {}),
                cooldown_frames=int(config.get("transition_cooldown_frames", 15))
            )
            self.roi_detectors[c_id] = RoiDetector(
                camera_id=c_id,
                zone=c_zone,
                roi_config=c_cfg.get("roi", {}),
                debounce_frames=int(config.get("roi_debounce_frames", 3))
            )

        # State per camera
        self.state = {}
        for cam in cameras_info:
            c_id = cam["camera_id"]
            c_cfg = cam_cfgs.get(c_id, {})
            c_zone = c_cfg.get("zone", c_id)
            v_path = cam.get("video_path", "")
            if not os.path.exists(v_path):
                # Fallback for common filename typos: panty <-> pantry
                if "pantry" in v_path and os.path.exists(v_path.replace("pantry", "panty")):
                    logging.warning(f"[{c_id}] Video '{v_path}' not found, falling back to '{v_path.replace('pantry', 'panty')}'")
                    v_path = v_path.replace("pantry", "panty")
                elif "panty" in v_path and os.path.exists(v_path.replace("panty", "pantry")):
                    logging.warning(f"[{c_id}] Video '{v_path}' not found, falling back to '{v_path.replace('panty', 'pantry')}'")
                    v_path = v_path.replace("panty", "pantry")

            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                logging.error(f"[{c_id}] FAILED to open video: '{v_path}'. Verify file exists and format is supported.")

            self.state[c_id] = {
                "cap": cap,
                "video_path": v_path,
                "zone": c_zone,
                "model": YOLO(config["model_path"]), # Independent YOLO state per camera!
                "banks": {},
                "global_assignments": {},
                "match_status": {},
                "attire_assignments": {},
                "attire_history": {},
                "last_frame": None,
                "finished": False,
                "detections": 0,
                "active_tracks": 0,
                "reid_extractions": 0,
                "global_ids_set": set(),
                "no_match_counters": {},
                "hysteresis_counters": {},
                "hysteresis_candidate": {},
                "last_seen_frame": {},
                "local_id_map": {},
                "next_local_id": 1,
                "comparison_history": {},
                "ambiguity_tracker": {},
                "pending_candidate": {},
                "pending_counters": {}
            }
            
        self.running = True

    def _sync_global_id_assignment(self, camera_id: str, track_id: int, gid: str):
        """
        Synchronizes Global ID assignment with PersonStateManager and HandwashDetector.
        Preserves established handwash state and links local track.
        """
        if not gid or str(gid).upper() in ["PENDING", "AMBIGUOUS", "UNKNOWN"]:
            return
        if self.person_state_manager:
            self.person_state_manager.link_track(camera_id, track_id, gid)
            if self.handwash_detector:
                t_hw = self.handwash_detector.get_status(track_id)
                if t_hw != "UNKNOWN":
                    self.person_state_manager.update_handwash_status(gid, t_hw)

    def _log_par_event(self, event_dict):
        try:
            os.makedirs("logs", exist_ok=True)
            with open("logs/par_detections.jsonl", "a") as f:
                f.write(json.dumps(event_dict) + "\n")
        except Exception:
            pass

    def run(self, max_frames: Optional[int] = None):
        # Validate cameras
        active_cameras = []
        for cam_id, state in self.state.items():
            if not state["cap"].isOpened():
                logging.error(f"[{cam_id}] Error opening video.")
                state["finished"] = True
            else:
                active_cameras.append(cam_id)
                
        if not active_cameras:
            logging.error("No valid cameras to process.")
            return

        # Use first active camera for base dimensions
        base_cap = self.state[active_cameras[0]]["cap"]
        w = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(base_cap.get(cv2.CAP_PROP_FPS))
        
        # We will stack horizontally, cctv1 processed and cctv2 processed.
        out_w = w * len(self.cameras_info)
        out_fps = fps if fps > 0 else 30
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, out_fps, (out_w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, out_fps, (out_w, h))
        
        cv2.namedWindow("Multi-Camera Tracker", cv2.WINDOW_NORMAL)
        
        frame_idx = 0
        if max_frames is None:
            max_frames = self.config.get("max_frames")
        while self.running:
            if max_frames and frame_idx >= max_frames:
                logging.info(f"Reached max_frames limit: {max_frames}. Ending processing.")
                break
            all_finished = True
            display_frames = []
            
            for cam in self.cameras_info:
                cam_id = cam["camera_id"]
                state = self.state[cam_id]
                
                if not state["finished"]:
                    ret, frame = state["cap"].read()
                    if not ret:
                        state["finished"] = True
                    else:
                        all_finished = False
                        
                        # Process frame
                        frame = self._process_frame(cam_id, frame, frame_idx, w, h)
                        state["last_frame"] = frame.copy()
                        
                        # Add camera name and zone overlay
                        c_zone_name = state.get("zone", cam_id).upper()
                        cv2.putText(state["last_frame"], f"Camera: {cam_id} | Zone: {c_zone_name}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                
                # If finished, we just use the last frame (frozen)
                if state["last_frame"] is not None:
                    display_frames.append(state["last_frame"])
                else:
                    # Never read a single frame (camera failed or offline)
                    placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                    cv2.putText(placeholder, f"CAMERA OFFLINE: {cam_id}", (50, h // 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.putText(placeholder, f"Path: {state.get('video_path', '')}", (50, (h // 2) + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)
                    display_frames.append(placeholder)
            
            if all_finished:
                break
                
            frame_idx += 1
            
            if frame_idx % 30 == 0:
                print(f"\n[FRAME {frame_idx}]")
                for cam in self.cameras_info:
                    c_id = cam["camera_id"]
                    st = self.state[c_id]
                    gids_str = ", ".join(sorted(list(st["global_ids_set"]))) if st["global_ids_set"] else "None"
                    print(f"{c_id.upper()}:")
                    print(f"  detections = {st['detections']}")
                    print(f"  active_tracks = {st['active_tracks']}")
                    print(f"  reid_extractions = {st['reid_extractions']}")
                    print(f"  global_ids = [{gids_str}]\n")
                    # Reset counters for the next 30 frames
                    st["detections"] = 0
                    st["active_tracks"] = 0
                    st["reid_extractions"] = 0
            
            # Ensure all frames are exactly the same height for hstack display
            for i in range(len(display_frames)):
                if display_frames[i].shape[:2] != (h, w):
                    display_frames[i] = cv2.resize(display_frames[i], (w, h))
                    
            combined_frame = np.hstack(display_frames)
            out.write(combined_frame)
            
            try:
                cv2.imshow("Multi-Camera Tracker", combined_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            except Exception:
                pass
                
        # Cleanup
        for state in self.state.values():
            if state["cap"].isOpened():
                state["cap"].release()
        out.release()
        cv2.destroyAllWindows()
        logging.info("Finished processing all cameras.")

    def _process_frame(self, camera_id, frame, frame_idx, w, h):
        state = self.state[camera_id]
        
        results = state["model"].track(
            frame, 
            classes=[0], 
            tracker=self.config.get("tracker_config", "botsort.yaml"), 
            persist=True, 
            verbose=False,
            conf=0.10,
            imgsz=self.config.get("imgsz", 1280)
        )
        
        # ADDED: User requested debug logging
        result = results[0]
        logging.info(f"{camera_id.upper()}: boxes={len(result.boxes)}, track_ids={result.boxes.id}")
        
        tracks = []
        raw_detections = []
        
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()
            
            # If tracker assigned IDs, map raw global IDs to camera-local IDs to isolate tracker state
            if results[0].boxes.id is not None:
                raw_track_ids = results[0].boxes.id.cpu().numpy()
                for b, r_tid, conf, cls in zip(boxes, raw_track_ids, confs, clss):
                    r_tid = int(r_tid)
                    if r_tid not in state["local_id_map"]:
                        state["local_id_map"][r_tid] = state["next_local_id"]
                        state["next_local_id"] += 1
                    local_tid = state["local_id_map"][r_tid]
                    tracks.append([*b, local_tid, conf, cls])
            else:
                # If tracker failed to assign IDs, keep raw detections for debugging
                for b, conf, cls in zip(boxes, confs, clss):
                    raw_detections.append([*b, conf, cls])
                    
        # Apply exclusion zones to filter out reflections/ignored areas
        exclusion_zones = self.config.get("exclusion_zones", {}).get(camera_id, [])
        if exclusion_zones:
            valid_tracks = []
            for t in tracks:
                cx, cy = (t[0] + t[2]) / 2 / w, (t[1] + t[3]) / 2 / h
                in_zone = any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in exclusion_zones)
                if not in_zone:
                    valid_tracks.append(t)
            tracks = valid_tracks
            
            valid_raw = []
            for d in raw_detections:
                cx, cy = (d[0] + d[2]) / 2 / w, (d[1] + d[3]) / 2 / h
                in_zone = any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in exclusion_zones)
                if not in_zone:
                    valid_raw.append(d)
            raw_detections = valid_raw
            
        state["detections"] += len(tracks) + len(raw_detections)
        state["active_tracks"] = len(tracks)
                
        # Find all currently active Global IDs in this camera frame (including recently occluded tracks)
        active_global_ids = set()
        
        # First update last seen frames for current tracks
        for t in tracks:
            track_id = int(t[4])
            state["last_seen_frame"][track_id] = frame_idx
            
        # Build active GIDs from any track seen in the last 30 frames (to cover brief BoT-SORT occlusions)
        for tid, last_frame in state["last_seen_frame"].items():
            if frame_idx - last_frame <= 30:
                if tid in state["global_assignments"]:
                    active_global_ids.add(state["global_assignments"][tid])

        cam_zone = state.get("zone", camera_id)
        fps = state["cap"].get(cv2.CAP_PROP_FPS)
        fps = fps if (fps and fps > 0) else 30.0
        current_time = float(frame_idx) / float(fps)

        # Handle tracks that disappeared (> 30 frames)
        current_track_ids = {int(t[4]) for t in tracks}
        for tid, l_frame in list(state["last_seen_frame"].items()):
            if tid not in current_track_ids and (frame_idx - l_frame > 30):
                if camera_id in self.roi_detectors:
                    self.roi_detectors[camera_id].on_track_lost(tid, frame_idx, current_time)
                if self.journey_manager:
                    self.journey_manager.on_track_lost(camera_id, cam_zone, tid, current_time, frame_idx)
                state["last_seen_frame"].pop(tid, None)

        # Periodic expiration check
        if self.journey_manager and frame_idx % 30 == 0:
            self.journey_manager.check_expirations(current_time, frame_idx)

        # Process each active track
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            # Event Detection: ROI
            if camera_id in self.roi_detectors:
                roi_events = self.roi_detectors[camera_id].check_rois(
                    local_track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    global_id=state["global_assignments"].get(track_id)
                )
                for revt in roi_events:
                    if self.journey_manager:
                        self.journey_manager.record_roi_event(revt)
                        logging.info(f"[ROI] Track {track_id} in {camera_id}: {revt.event} on {revt.roi_id}")

            # Event Detection: Transition Lines
            if camera_id in self.transition_detectors:
                trans_events = self.transition_detectors[camera_id].check_transitions(
                    local_track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    global_id=state["global_assignments"].get(track_id)
                )
                for tevt in trans_events:
                    if self.journey_manager:
                        self.journey_manager.record_transition(tevt)
                        logging.info(f"[TRANSITION] Track {track_id} in {camera_id}: crossed {tevt.transition_id} ({tevt.direction})")

            # ReID Feature Extraction & Bank Update
            if self.config.get("reid_enabled", True) and frame_idx % self.config.get("reid_interval", 3) == 0:
                added = False
                quality = 0.0
                if crop.size > 0:
                    quality, accepted, reason, metadata = calculate_quality(
                        crop, conf, 
                        self.config.get("min_reid_confidence", 0.35),
                        self.config.get("min_reid_width", 45),
                        self.config.get("min_reid_height", 90),
                        self.config.get("blur_threshold", 10.0),
                        w, h, (x1, y1, x2, y2)
                    )
                    
                    if accepted:
                        state["reid_extractions"] += 1
                        embedding = self.extractor.extract([crop])[0]
                        par_attr = self.par_extractor.extract_attributes(crop) if hasattr(self.par_extractor, "extract_attributes") else {}
                        par_features = par_attr.get("features", self.par_extractor.extract(crop))
                        viewpoint, vp_conf = self.pose_extractor.extract_viewpoint(crop)
                        metadata["viewpoint_confidence"] = float(vp_conf)
                        metadata["viewpoint"] = viewpoint
                        metadata["camera_id"] = camera_id
                        metadata["zone"] = cam_zone
                        metadata["track_id"] = track_id
                        metadata["timestamp"] = current_time
                        
                        if track_id not in state["banks"]:
                            state["banks"][track_id] = TrackEmbeddingBank(
                                camera_id, track_id, 
                                max_size_per_view=self.config.get("reid_bank_size", 15),
                                diversity_threshold=self.config.get("embedding_diversity_threshold", 0.90)
                            )
                            state["match_status"][track_id] = "PENDING"
                            
                        # Add embedding with metadata
                        added, reason, entry = state["banks"][track_id].add_embedding(
                            embedding=embedding, 
                            quality=quality, 
                            frame_id=frame_idx,
                            viewpoint=viewpoint,
                            par_features=par_features,
                            quality_metadata=metadata,
                            par_attributes=par_attr
                        )

                # ---------------------------------------------------------
                # Phase 1 Association via AssociationManager
                # ---------------------------------------------------------
                if track_id in state["banks"] and self.association_manager is not None:
                    bank_embeddings = state["banks"][track_id].get_all_embeddings()
                    if bank_embeddings:
                        assigned_gid, assoc_status, sim, details = self.association_manager.associate_track(
                            camera_id=camera_id,
                            zone=cam_zone,
                            local_track_id=track_id,
                            bank_embeddings=bank_embeddings,
                            current_time=current_time,
                            frame_idx=frame_idx,
                            active_camera_gids=active_global_ids
                        )
                        if assoc_status == "MATCHED" and assigned_gid:
                            state["global_assignments"][track_id] = assigned_gid
                            state["global_ids_set"].add(assigned_gid)
                            state["match_status"][track_id] = "MATCHED"
                            active_global_ids.add(assigned_gid)
                            update_min_q = float(self.config.get("gallery_update_min_quality", 0.60))
                            if added and quality >= update_min_q:
                                self.gallery.update_identity(
                                    assigned_gid, 
                                    [{"camera_id": camera_id, "track_id": track_id}], 
                                    bank_embeddings
                                )
                        elif assoc_status == "PENDING":
                            state["match_status"][track_id] = "PENDING"

        # ---------------------------------------------------------
        # Phase 1 Floor-Plan Annotations & Overlays
        # ---------------------------------------------------------
        if self.phase1_mode:
            self._draw_phase1_annotations(camera_id, frame, h, w, tracks, state, cam_zone)
        else:
            # Fallback legacy annotations if requested
            for t in tracks:
                x1, y1, x2, y2, track_id, conf, cls = t
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                gid = state["global_assignments"].get(int(track_id), "UNKNOWN")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Trk: {track_id} | GID: {gid}", (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        # Draw raw detections in yellow (if tracker dropped them)
        for d in raw_detections:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Det Only: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

        return frame

    def _draw_phase1_annotations(self, camera_id: str, frame: np.ndarray, h: int, w: int, tracks: list, state: dict, cam_zone: str):
        # 1. Draw ROIs
        if camera_id in self.roi_detectors:
            for roi_id, poly in self.roi_detectors[camera_id].polygons.items():
                cv2.polylines(frame, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = int(poly[0][0][0]), int(poly[0][0][1])
                cv2.putText(frame, f"ROI: {roi_id.upper()}", (max(10, cx - 40), max(20, cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2)

        # 2. Draw Transition Lines
        if camera_id in self.transition_detectors:
            for trans_id, line_info in self.transition_detectors[camera_id].lines.items():
                p1 = (int(line_info["start"][0]), int(line_info["start"][1]))
                p2 = (int(line_info["end"][0]), int(line_info["end"][1]))
                line_col = (0, 255, 255) if "yellow" in trans_id else ((0, 255, 0) if "green" in trans_id else (0, 165, 255))
                cv2.line(frame, p1, p2, line_col, 3)
                cv2.circle(frame, p1, 5, line_col, -1)
                cv2.circle(frame, p2, 5, line_col, -1)
                mid_x, mid_y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
                cv2.putText(frame, trans_id, (mid_x - 30, max(20, mid_y - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2)

        # 3. Draw Track Boxes & Journey Badges
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            foot_pt = (int((x1 + x2) / 2), int(y2))
            cv2.circle(frame, foot_pt, 4, (0, 255, 255), -1)

            gid = state["global_assignments"].get(track_id)
            m_status = state.get("match_status", {}).get(track_id, "PENDING")
            
            ident_state = self.journey_manager.get_identity(gid) if (gid and self.journey_manager) else None
            last_trans = ident_state.last_transition_id if ident_state else None
            curr_roi = ident_state.current_roi if ident_state else None

            if m_status == "MATCHED" and gid:
                color = (0, 255, 0)
                top_text = f"GID: {gid} | L{track_id} | CONFIRMED"
            elif m_status == "PENDING":
                color = (0, 165, 255) # Orange
                cand = self.association_manager.pending_candidate.get((camera_id, track_id)) if self.association_manager else None
                if cand:
                    cnt = self.association_manager.pending_counters.get((camera_id, track_id), 1)
                    top_text = f"PENDING ({cand} {cnt}/5) | L{track_id}"
                else:
                    top_text = f"PENDING | L{track_id}"
            else:
                color = (255, 255, 0) # Cyan/Yellow
                top_text = f"NEW/UNKNOWN | L{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, top_text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            sub_text = f"Zone: {cam_zone} | ROI: {curr_roi or 'None'} | Last: {last_trans or 'None'}"
            cv2.putText(frame, sub_text, (x1, min(h - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 2)

    def stop(self):
        self.running = False
