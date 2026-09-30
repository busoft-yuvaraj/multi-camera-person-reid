import cv2
import numpy as np
import os
import json
import datetime
import logging

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
        handwash_detector=None
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
        
        # State per camera
        self.state = {}
        for cam in cameras_info:
            c_id = cam["camera_id"]
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

    def run(self):
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
        while self.running:
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
                        
                        # Add camera name overlay
                        cv2.putText(state["last_frame"], f"Camera: {cam_id}", (20, 40), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
                
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

        # Pantry Handwashing Processing
        current_pathway = self.route_policy.get_pathway(camera_id) if self.route_policy else camera_id.upper()
        if (current_pathway == "PANTRY" or "pantry" in camera_id.lower()) and self.handwash_detector:
            self.handwash_detector.process_frame(
                frame=frame,
                tracks=tracks,
                frame_idx=frame_idx,
                fps=25.0,
                global_id_map=state["global_assignments"]
            )
            for t in tracks:
                tid = int(t[4])
                hw_st = self.handwash_detector.get_status(tid)
                gid = state["global_assignments"].get(tid)
                if gid and str(gid).upper() not in ["PENDING", "AMBIGUOUS", "UNKNOWN"]:
                    if hw_st != "UNKNOWN" and self.person_state_manager:
                        self.person_state_manager.update_handwash_status(gid, hw_st)

            # Check tracks that disappeared from pantry -> transition UNKNOWN to NOT_HANDWASHED
            current_track_ids = {int(t[4]) for t in tracks}
            for tid, l_frame in list(state["last_seen_frame"].items()):
                if tid not in current_track_ids and (frame_idx - l_frame > 30):
                    lost_status = self.handwash_detector.on_track_lost(tid)
                    gid = state["global_assignments"].get(tid)
                    if gid and str(gid).upper() not in ["PENDING", "AMBIGUOUS", "UNKNOWN"] and self.person_state_manager:
                        self.person_state_manager.update_handwash_status(gid, lost_status)

        # Draw virtual line if configured for this camera
        if self.violation_enabled and camera_id in self.line_detectors:
            ld = self.line_detectors[camera_id]
            if ld.enabled:
                line_col = getattr(ld, "color", (0, 0, 255))
                cv2.line(frame, ld.line_start, ld.line_end, line_col, 3)
                cv2.circle(frame, ld.line_start, 5, line_col, -1)
                cv2.circle(frame, ld.line_end, 5, line_col, -1)
                line_title = f"{current_pathway} Line" if current_pathway != "PANTRY" else "Zone Line"
                cv2.putText(frame, line_title, (ld.line_start[0], max(20, ld.line_start[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_col, 2)

        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            
        # Formal Attire Detection
            if self.attire_extractor and self.config.get("attire_detection_enabled", True) and crop.size > 0:
                pred_interval = self.config.get("violation_detection", {}).get("attire", {}).get(
                    "prediction_interval", self.config.get("attire_interval", 2)
                )
                should_run = (frame_idx % pred_interval == 0 or track_id not in state["attire_assignments"])
                if self.attire_smoother:
                    should_run = self.attire_smoother.should_predict(frame_idx, track_id)

                if should_run:
                    attire_res = self.attire_extractor.extract_attire(crop)
                    if attire_res["label"] != "unknown":
                        if self.attire_smoother:
                            self.attire_smoother.add_prediction(track_id, attire_res)
                            stable = self.attire_smoother.get_stable_attire(track_id)
                            state["attire_assignments"][track_id] = {
                                "label": stable["final_attire"].capitalize(),
                                "conf": stable["confidence"],
                                "is_formal": stable["is_formal"]
                            }
                        else:
                            if track_id not in state["attire_history"]:
                                state["attire_history"][track_id] = []
                            state["attire_history"][track_id].append(attire_res)
                            if len(state["attire_history"][track_id]) > 10:
                                state["attire_history"][track_id].pop(0)
                            
                            formal_count = sum(1 for a in state["attire_history"][track_id] if a.get("is_formal") is True)
                            informal_count = sum(1 for a in state["attire_history"][track_id] if a.get("is_formal") is False)
                            avg_conf = float(np.mean([a["conf"] for a in state["attire_history"][track_id]]))
                            is_formal = formal_count >= informal_count
                            state["attire_assignments"][track_id] = {
                                "label": "Formal" if is_formal else "Informal",
                                "conf": avg_conf,
                                "is_formal": is_formal
                            }
            
            if self.config["reid_enabled"] and frame_idx % self.config["reid_interval"] == 0:
                if crop.size > 0:
                    quality, accepted, reason, metadata = calculate_quality(
                        crop, conf, 
                        self.config["min_reid_confidence"],
                        self.config["min_reid_width"],
                        self.config["min_reid_height"],
                        self.config["blur_threshold"],
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
                        
                        if track_id not in state["banks"]:
                            state["banks"][track_id] = TrackEmbeddingBank(
                                camera_id, track_id, 
                                max_size_per_view=self.config.get("reid_bank_size", 5),
                                diversity_threshold=self.config.get("embedding_diversity_threshold", 0.90)
                            )
                            state["match_status"][track_id] = "PENDING"
                            state["pending_candidate"][track_id] = None
                            state["pending_counters"][track_id] = 0
                            
                        # Add embedding with all multi-evidence metadata
                        added, reason, entry = state["banks"][track_id].add_embedding(
                            embedding=embedding, 
                            quality=quality, 
                            frame_id=frame_idx,
                            viewpoint=viewpoint,
                            par_features=par_features,
                            quality_metadata=metadata,
                            par_attributes=par_attr
                        )
                        
                        logging.info(
                            f"[PAR EXTRACTION] Camera={camera_id}, Track={track_id}, View={viewpoint} | "
                            f"Shirt={par_attr.get('upper_color', 'unknown')}, Pants={par_attr.get('lower_color', 'unknown')}, "
                            f"Conf={par_attr.get('confidence', 0.0):.2f}"
                        )
                        self._log_par_event({
                            "timestamp": datetime.datetime.now().isoformat(),
                            "camera_id": camera_id,
                            "track_id": track_id,
                            "frame_idx": frame_idx,
                            "viewpoint": viewpoint,
                            "upper_color": par_attr.get("upper_color", "unknown"),
                            "lower_color": par_attr.get("lower_color", "unknown"),
                            "confidence": round(float(par_attr.get("confidence", 0.0)), 3)
                        })
                        
                        m_status = state["match_status"].get(track_id, "PENDING")
                        
                        if m_status in ["PENDING", "AMBIGUOUS"]:
                            bank_embeddings = state["banks"][track_id].get_all_embeddings()
                            confirm_frames = self.config.get("reid_confirm_frames", 5)
                            
                            # Only attempt to match if we have collected enough evidence
                            if len(bank_embeddings) >= confirm_frames:
                                # Exclude GIDs currently held by other tracks in this camera
                                camera_occupied_gids = set(active_global_ids)
                                current_track_gid = state["global_assignments"].get(track_id)
                                if current_track_gid in camera_occupied_gids:
                                    camera_occupied_gids.remove(current_track_gid)
                                    
                                consecutive_cands = state["pending_counters"].get(track_id, 1)
                                gid, sim, second_sim, raw_status = self.matcher.match(
                                    bank_embeddings, 
                                    exclude_gids=camera_occupied_gids,
                                    current_camera=camera_id,
                                    track_id=track_id,
                                    consecutive_candidate_count=consecutive_cands
                                )
                                
                                if track_id not in state["comparison_history"]:
                                    state["comparison_history"][track_id] = []
                                state["comparison_history"][track_id].append({
                                    "gid": gid,
                                    "sim": sim,
                                    "second_sim": second_sim,
                                    "raw_status": raw_status
                                })
                                if len(state["comparison_history"][track_id]) > 10:
                                    state["comparison_history"][track_id].pop(0)

                                update_min_q = float(self.config.get("gallery_update_min_quality", 0.55))
                                
                                # -------------------------------------------------------------
                                # Decision Handling (Phases 7, 12, 14)
                                # -------------------------------------------------------------
                                if raw_status == "MATCH" and gid is not None:
                                    if gid in active_global_ids and state["global_assignments"].get(track_id) != gid:
                                        logging.warning(f"Prevented double-assignment: {gid} is already active in camera {camera_id}.")
                                        state["match_status"][track_id] = "PENDING"
                                    else:
                                        state["global_assignments"][track_id] = gid
                                        state["global_ids_set"].add(gid)
                                        state["match_status"][track_id] = "MATCHED"
                                        active_global_ids.add(gid)
                                        self._sync_global_id_assignment(camera_id, track_id, gid)
                                        state["pending_candidate"][track_id] = None
                                        state["pending_counters"][track_id] = 0
                                        
                                        # Phase 14: Only update gallery when CONFIRMED
                                        if quality >= update_min_q:
                                            self.gallery.update_identity(gid, [{"camera_id": camera_id, "track_id": track_id}], bank_embeddings)
                                            
                                        print(f"\n[REID MATCH CONFIRMED]\ncamera={camera_id}\ntrack={track_id}\nevidence={len(bank_embeddings)}\nmatched={gid}\nsimilarity={sim:.2f}\nview={viewpoint}\n")
                                        logging.info(f"[GLOBAL ID CONFIRMED] Track {track_id} in {camera_id} confirmed as {gid} (sim={sim:.2f})")
                                        ident = getattr(self.gallery, "identities", {}).get(gid)
                                        if ident:
                                            p_prof = ident.par_profile if hasattr(ident, "par_profile") else ident.get("par_profile", {})
                                            logging.info(f"[PAR PROFILE] {gid} | Stable Shirt={p_prof.get('upper_color', 'unknown')}, Pants={p_prof.get('lower_color', 'unknown')}, Obs={p_prof.get('par_count', 0)}")
                                        
                                elif raw_status == "PENDING" and gid is not None:
                                    state["match_status"][track_id] = "PENDING"
                                    prev_cand = state["pending_candidate"].get(track_id)
                                    if prev_cand == gid:
                                        state["pending_counters"][track_id] = state["pending_counters"].get(track_id, 0) + 1
                                    else:
                                        state["pending_candidate"][track_id] = gid
                                        state["pending_counters"][track_id] = 1
                                        
                                    p_count = state["pending_counters"][track_id]
                                    pending_promote_frames = int(self.config.get("pending_confirm_frames", 5))
                                    
                                    # Promotion from PENDING -> CONFIRMED (Phase 7: CCTV1 BACK -> CCTV2 SIDE -> CCTV2 FRONT)
                                    if p_count >= pending_promote_frames or (viewpoint == "FRONT" and sim >= 0.72):
                                        if gid not in active_global_ids or state["global_assignments"].get(track_id) == gid:
                                            state["global_assignments"][track_id] = gid
                                            state["global_ids_set"].add(gid)
                                            state["match_status"][track_id] = "MATCHED"
                                            active_global_ids.add(gid)
                                            self._sync_global_id_assignment(camera_id, track_id, gid)
                                            if quality >= update_min_q:
                                                self.gallery.update_identity(gid, [{"camera_id": camera_id, "track_id": track_id}], bank_embeddings)
                                                
                                            print(f"\n[GLOBAL ID CONFIRMED FROM PENDING]\ncamera={camera_id}\ntrack={track_id}\nglobal_id={gid}\nsimilarity={sim:.2f}\nconsecutive_frames={p_count}\nview={viewpoint}\n")
                                            logging.info(f"[GLOBAL ID CONFIRMED] Track {track_id} in {camera_id} confirmed as {gid} from PENDING after {p_count} checks")
                                            ident = getattr(self.gallery, "identities", {}).get(gid)
                                            if ident:
                                                p_prof = ident.par_profile if hasattr(ident, "par_profile") else ident.get("par_profile", {})
                                                logging.info(f"[PAR PROFILE] {gid} | Stable Shirt={p_prof.get('upper_color', 'unknown')}, Pants={p_prof.get('lower_color', 'unknown')}, Obs={p_prof.get('par_count', 0)}")
                                    else:
                                        # Reset no_match_counters so a premature new ID is NEVER spawned while in PENDING
                                        state["no_match_counters"][track_id] = 0
                                        logging.info(f"[GLOBAL MATCH PENDING] Track {track_id} in {camera_id} -> Candidate {gid} (Score={sim:.2f}, Count={p_count}/{pending_promote_frames})")
                                        
                                elif raw_status == "NO_MATCH":
                                    state["no_match_counters"][track_id] = state["no_match_counters"].get(track_id, 0) + 1
                                    new_id_confirm = int(self.config.get("reid_new_id_confirm_frames", 25))
                                    
                                    if state["no_match_counters"][track_id] >= new_id_confirm:
                                        new_gid = self.gallery.add_identity([{"camera_id": camera_id, "track_id": track_id}], bank_embeddings)
                                        state["global_assignments"][track_id] = new_gid
                                        state["global_ids_set"].add(new_gid)
                                        state["match_status"][track_id] = "MATCHED"
                                        active_global_ids.add(new_gid)
                                        self._sync_global_id_assignment(camera_id, track_id, new_gid)
                                        state["pending_candidate"][track_id] = None
                                        state["pending_counters"][track_id] = 0
                                        
                                        print(f"\n[GLOBAL ID NEW]\ncamera={camera_id}\ntrack={track_id}\nevidence={len(bank_embeddings)}\nstatus=NEW\nglobal_id={new_gid}\n")
                                        logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, gid={new_gid}, sim={sim:.2f}, status=NEW")
                                        ident = getattr(self.gallery, "identities", {}).get(new_gid)
                                        if ident:
                                            p_prof = ident.par_profile if hasattr(ident, "par_profile") else ident.get("par_profile", {})
                                            logging.info(f"[PAR PROFILE] {new_gid} | Initial Shirt={p_prof.get('upper_color', 'unknown')}, Pants={p_prof.get('lower_color', 'unknown')}, Obs={p_prof.get('par_count', 0)}")
                                    else:
                                        state["match_status"][track_id] = "PENDING"
                                
                        elif m_status == "MATCHED":
                            current_gid = state["global_assignments"][track_id]
                            bank_embeddings = state["banks"][track_id].get_all_embeddings()
                            
                            hysteresis_exclude = set(active_global_ids)
                            if current_gid in hysteresis_exclude:
                                hysteresis_exclude.remove(current_gid)
                                
                            gid, sim, second_sim, status = self.matcher.match(bank_embeddings, exclude_gids=hysteresis_exclude)
                            margin = sim - second_sim
                            
                            h_thresh = float(self.config.get("hysteresis_switch_threshold", 0.85))
                            h_frames = int(self.config.get("hysteresis_consecutive_frames", 5))
                            ambig_margin = float(self.config.get("hysteresis_switch_margin", self.config.get("reid_ambiguous_margin", 0.04)))
                            
                            # Hungarian 1-to-1 constraint: cannot switch to an ID currently held by another track in this camera
                            if status == "MATCH" and gid != current_gid and gid not in active_global_ids and sim >= h_thresh and margin >= ambig_margin:
                                if state["hysteresis_candidate"].get(track_id) != gid:
                                    state["hysteresis_candidate"][track_id] = gid
                                    state["hysteresis_counters"][track_id] = 1
                                else:
                                    state["hysteresis_counters"][track_id] += 1
                                    
                                if state["hysteresis_counters"][track_id] >= h_frames:
                                    # Execute Switch!
                                    print(f"\n[HYSTERESIS SWITCH]\ncamera={camera_id}\ntrack={track_id}\nold_gid={current_gid}\nnew_gid={gid}\nsimilarity={sim:.2f}\n")
                                    logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, old_gid={current_gid}, new_gid={gid}, sim={sim:.2f}, status=HYSTERESIS_SWITCH")
                                    
                                    if current_gid in state["global_ids_set"]:
                                        state["global_ids_set"].remove(current_gid)
                                    state["global_assignments"][track_id] = gid
                                    state["global_ids_set"].add(gid)
                                    self._sync_global_id_assignment(camera_id, track_id, gid)
                                    
                                    if current_gid in active_global_ids:
                                        active_global_ids.remove(current_gid)
                                    active_global_ids.add(gid)
                                    
                                    state["hysteresis_counters"][track_id] = 0
                                    state["hysteresis_candidate"][track_id] = None
                                    current_gid = gid
                            else:
                                state["hysteresis_counters"][track_id] = 0
                                state["hysteresis_candidate"][track_id] = None

                            update_min_q = float(self.config.get("gallery_update_min_quality", 0.55))
                            if added and quality >= update_min_q:
                                self.gallery.update_identity(current_gid, [{"camera_id": camera_id, "track_id": track_id}], [entry])
                                logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, gid={current_gid}, status=RETAINED, reason={reason}")
                                
            # Pathway & Handwashing Validation Check
            if self.violation_enabled and camera_id in self.line_detectors:
                ld = self.line_detectors[camera_id]
                crossing_evt = ld.check_crossing(track_id, (x1, y1, x2, y2))
                if crossing_evt:
                    current_gid = state["global_assignments"].get(track_id)
                    resolved_gid = current_gid if (current_gid and current_gid not in ["PENDING", "AMBIGUOUS", "UNKNOWN"]) else None
                    if resolved_gid is None:
                        resolved_gid = state.get("pending_candidate", {}).get(track_id)

                    hw_status = "UNKNOWN"
                    if resolved_gid and self.person_state_manager:
                        hw_status = self.person_state_manager.get_handwash_status(resolved_gid)
                    elif self.handwash_detector:
                        hw_status = self.handwash_detector.get_status(track_id)

                    self.violation_engine.process_crossing(
                        crossing_event=crossing_evt,
                        global_id=resolved_gid,
                        handwash_status=hw_status,
                        frame=frame,
                        bbox=(x1, y1, x2, y2)
                    )

            m_status = state.get("match_status", {}).get(track_id, "PENDING")
            vio_evt = self.violation_engine.get_violation(camera_id, track_id) if (self.violation_enabled and self.violation_engine) else None
            val_info = self.violation_engine.get_validation(camera_id, track_id) if (self.violation_enabled and self.violation_engine) else None

            current_gid = state["global_assignments"].get(track_id)
            resolved_gid = current_gid if (current_gid and current_gid not in ["PENDING", "AMBIGUOUS", "UNKNOWN"]) else None
            gid_display = resolved_gid or (f"PENDING ({state.get('pending_candidate', {}).get(track_id)})" if state.get("pending_candidate", {}).get(track_id) else "PENDING")

            hw_status = "UNKNOWN"
            if resolved_gid and self.person_state_manager:
                hw_status = self.person_state_manager.get_handwash_status(resolved_gid)
            elif self.handwash_detector:
                hw_status = self.handwash_detector.get_status(track_id)

            hw_flag = "YES" if hw_status == "HANDWASHED" else ("NO" if hw_status == "NOT_HANDWASHED" else "UNKNOWN")
            current_path = self.route_policy.get_pathway(camera_id) if self.route_policy else camera_id.upper()

            is_vio = (vio_evt is not None) or (resolved_gid and self.person_state_manager and self.person_state_manager.is_in_violation(resolved_gid))
            if is_vio:
                # Violation Rendering (Red Box & Handwash / Pathway Metadata)
                color = (0, 0, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Top Violation Banner
                vio_label = f"VIOLATION | GID: {gid_display}"
                cv2.putText(frame, vio_label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Bottom Handwash + Pathway Badge
                path_name = vio_evt.pathway if vio_evt else current_path
                sub_label = f"Handwash: {hw_flag} | Path: {path_name} | VIOLATION"
                cv2.putText(frame, sub_label, (x1, min(h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            elif val_info:
                # Valid Movement Rendering (Green Box)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                valid_label = f"VALID | GID: {gid_display}"
                cv2.putText(frame, valid_label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                val_path = val_info.get("pathway", current_path)
                sub_label = f"Handwash: {hw_flag} | Path: {val_path} | VALID"
                cv2.putText(frame, sub_label, (x1, min(h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                if m_status == "MATCHED":
                    color = (0, 255, 0)
                    label = f"Trk: {track_id} | GID: {gid_display}"
                elif m_status in ["PENDING", "AMBIGUOUS"]:
                    color = (0, 165, 255) # Orange
                    cand = state.get("pending_candidate", {}).get(track_id)
                    label = f"Trk: {track_id} | GID: PENDING ({cand})" if cand else f"Trk: {track_id} | GID: PENDING"
                else:
                    color = (255, 255, 0)
                    label = f"Trk: {track_id} | GID: UNKNOWN"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                # Handwash status badge below box
                hw_color = (0, 255, 0) if hw_flag == "YES" else ((0, 0, 255) if hw_flag == "NO" else (0, 255, 255))
                sub_label = f"Handwash: {hw_flag} | Path: {current_path}"
                cv2.putText(frame, sub_label, (x1, min(h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, hw_color, 2)
            
        # Draw raw detections in yellow (if tracker dropped them)
        for d in raw_detections:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Det Only: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
        return frame

    def stop(self):
        self.running = False
