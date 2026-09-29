import cv2
import numpy as np
import os
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
        person_state_manager=None
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
                    self.line_detectors[c_id] = LineCrossingDetector(
                        camera_id=c_id,
                        line_start=tuple(start),
                        line_end=tuple(end),
                        enabled=enabled
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
        
        # State per camera
        self.state = {}
        for cam in cameras_info:
            self.state[cam["camera_id"]] = {
                "cap": cv2.VideoCapture(cam["video_path"]),
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
                "ambiguity_tracker": {}
            }
            
        self.running = True

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
                    # Never read a single frame, just put black
                    display_frames.append(np.zeros((h, w, 3), dtype=np.uint8))
            
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

        # Draw virtual line if configured for this camera
        if self.violation_enabled and camera_id in self.line_detectors:
            ld = self.line_detectors[camera_id]
            if ld.enabled:
                cv2.line(frame, ld.line_start, ld.line_end, (255, 255, 0), 2)
                cv2.circle(frame, ld.line_start, 4, (255, 255, 0), -1)
                cv2.circle(frame, ld.line_end, 4, (255, 255, 0), -1)
                cv2.putText(frame, "Zone Line", (ld.line_start[0], max(20, ld.line_start[1] - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

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
                        par_features = self.par_extractor.extract(crop)
                        viewpoint, vp_conf = self.pose_extractor.extract_viewpoint(crop)
                        metadata["viewpoint_confidence"] = float(vp_conf)
                        
                        if track_id not in state["banks"]:
                            state["banks"][track_id] = TrackEmbeddingBank(
                                camera_id, track_id, 
                                max_size_per_view=self.config.get("reid_bank_size", 5), # Actually max per view is e.g. 5
                                diversity_threshold=self.config["embedding_diversity_threshold"]
                            )
                            state["match_status"][track_id] = "PENDING"
                            
                        # Add embedding with all multi-evidence metadata
                        added, reason, entry = state["banks"][track_id].add_embedding(
                            embedding=embedding, 
                            quality=quality, 
                            frame_id=frame_idx,
                            viewpoint=viewpoint,
                            par_features=par_features,
                            quality_metadata=metadata
                        )
                        
                        m_status = state["match_status"].get(track_id, "PENDING")
                        
                        if m_status in ["PENDING", "AMBIGUOUS"]:
                            bank_embeddings = state["banks"][track_id].get_all_embeddings()
                            confirm_frames = self.config.get("reid_confirm_frames", 10)
                            
                            # Only attempt to match if we have collected enough evidence
                            if len(bank_embeddings) >= confirm_frames:
                                # Exclude GIDs currently held by other tracks in this camera
                                camera_occupied_gids = set(active_global_ids)
                                current_track_gid = state["global_assignments"].get(track_id)
                                if current_track_gid in camera_occupied_gids:
                                    camera_occupied_gids.remove(current_track_gid)
                                    
                                gid, sim, second_sim, raw_status = self.matcher.match(
                                    bank_embeddings, 
                                    exclude_gids=camera_occupied_gids
                                )
                                
                                # 1. Decide per track, not per frame: Average last 10 comparison scores
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

                                history = state["comparison_history"][track_id]
                                valid_gids = [e["gid"] for e in history if e["gid"] is not None]
                                
                                thresh_match = float(self.config.get("threshold_match", 0.83))
                                thresh_ambig = float(self.config.get("threshold_ambiguous", 0.80))
                                ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))
                                
                                if valid_gids:
                                    top_gid = max(set(valid_gids), key=valid_gids.count)
                                    cand_entries = [e for e in history if e["gid"] == top_gid]
                                    mean_sim = float(np.mean([e["sim"] for e in cand_entries]))
                                    mean_second_sim = float(np.mean([e["second_sim"] for e in cand_entries]))
                                    mean_margin = mean_sim - mean_second_sim
                                    
                                    if mean_sim >= thresh_match and (mean_margin >= ambig_margin or mean_second_sim == 0.0):
                                        status = "MATCH"
                                    elif mean_sim >= thresh_ambig:
                                        status = "AMBIGUOUS"
                                    else:
                                        status = "NO_MATCH"
                                        
                                    # 2. Resolve stuck ambiguity:
                                    # If a track stays ambiguous for about 20 comparisons with the same top candidate and gap >= 0.04, accept it
                                    if status == "AMBIGUOUS":
                                        prev_ambig = state["ambiguity_tracker"].get(track_id, {})
                                        if prev_ambig.get("gid") == top_gid:
                                            ambig_count = prev_ambig.get("count", 0) + 1
                                        else:
                                            ambig_count = 1
                                        state["ambiguity_tracker"][track_id] = {"gid": top_gid, "count": ambig_count}
                                        
                                        if ambig_count >= 20 and mean_margin >= ambig_margin:
                                            status = "MATCH"
                                            logging.info(
                                                f"[RESOLVED AMBIGUITY] camera={camera_id}, track={track_id}, accepted gid={top_gid} "
                                                f"after {ambig_count} ambiguous comparisons (mean_sim={mean_sim:.2f}, gap={mean_margin:.2f})"
                                            )
                                else:
                                    top_gid = None
                                    mean_sim = 0.0
                                    mean_second_sim = 0.0
                                    mean_margin = 0.0
                                    status = "NO_MATCH"
                                
                                if status == "MATCH" and top_gid is not None:
                                    # Hungarian/uniqueness rule: no two tracks in the same camera can share a global ID
                                    if top_gid in active_global_ids and state["global_assignments"].get(track_id) != top_gid:
                                        logging.warning(f"Prevented double-assignment: {top_gid} is already active in camera {camera_id}.")
                                        status = "AMBIGUOUS"
                                    else:
                                        is_recovery = track_id not in state["no_match_counters"] and len(bank_embeddings) > confirm_frames + 5
                                        rec_thresh = float(self.config.get("threshold_recovery", 0.82))
                                        if not is_recovery or (is_recovery and mean_sim >= rec_thresh):
                                            state["global_assignments"][track_id] = top_gid
                                            state["global_ids_set"].add(top_gid)
                                            state["match_status"][track_id] = "MATCHED"
                                            active_global_ids.add(top_gid)
                                            
                                            update_min_q = float(self.config.get("gallery_update_min_quality", 0.55))
                                            if quality >= update_min_q:
                                                self.gallery.update_identity(top_gid, [{"camera_id": camera_id, "track_id": track_id}], bank_embeddings)
                                            
                                            print(f"\n[REID MATCH]\ncamera={camera_id}\ntrack={track_id}\nevidence={len(bank_embeddings)}\nquality={quality:.2f}\nmean_similarity={mean_sim:.2f}\nmean_margin={mean_margin:.2f}\nmatched={top_gid}\n")
                                            logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, gid={top_gid}, mean_sim={mean_sim:.2f}, status=MATCH")
                                        else:
                                            status = "AMBIGUOUS"
                                            
                                if status == "NO_MATCH":
                                    state["no_match_counters"][track_id] = state["no_match_counters"].get(track_id, 0) + 1
                                    new_id_confirm = int(self.config.get("reid_new_id_confirm_frames", 20))
                                    
                                    if state["no_match_counters"][track_id] >= new_id_confirm:
                                        new_gid = self.gallery.add_identity([{"camera_id": camera_id, "track_id": track_id}], bank_embeddings)
                                        state["global_assignments"][track_id] = new_gid
                                        state["global_ids_set"].add(new_gid)
                                        state["match_status"][track_id] = "MATCHED"
                                        active_global_ids.add(new_gid)
                                        
                                        print(f"\n[GLOBAL ID NEW]\ncamera={camera_id}\ntrack={track_id}\nevidence={len(bank_embeddings)}\nstatus=NEW\nglobal_id={new_gid}\n")
                                        logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, gid={new_gid}, sim={mean_sim:.2f}, status=NEW")
                                    else:
                                        state["match_status"][track_id] = "AMBIGUOUS"
                                        print(f"\n[DELAY NEW ID]\ncamera={camera_id}\ntrack={track_id}\nno_match_count={state['no_match_counters'][track_id]}/{new_id_confirm}\naction=WAIT_FOR_MORE_EVIDENCE\n")
                                        
                                if status == "AMBIGUOUS":
                                    state["match_status"][track_id] = "AMBIGUOUS"
                                    logging.info(f"MATCH_LOG: camera={camera_id}, track={track_id}, gid=PENDING, mean_sim={mean_sim:.2f}, status=AMBIGUOUS")
                                
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
                            ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))
                            
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
                                
            # Route & Attire Violation Detection Check
            if self.violation_enabled and camera_id in self.line_detectors:
                ld = self.line_detectors[camera_id]
                crossing_evt = ld.check_crossing(track_id, (x1, y1, x2, y2))
                if crossing_evt:
                    stable_att = self.attire_smoother.get_stable_attire(track_id) if self.attire_smoother else {}
                    if not stable_att.get("final_attire") or stable_att["final_attire"] == "unknown":
                        if track_id in state["attire_assignments"]:
                            a_info = state["attire_assignments"][track_id]
                            stable_att = {"final_attire": a_info["label"].lower(), "confidence": a_info["conf"]}

                    current_gid = state["global_assignments"].get(track_id)
                    resolved_gid = current_gid if (current_gid and current_gid not in ["PENDING", "AMBIGUOUS", "UNKNOWN"]) else None

                    self.violation_engine.process_crossing(
                        crossing_event=crossing_evt,
                        global_id=resolved_gid,
                        stable_attire=stable_att,
                        frame=frame,
                        bbox=(x1, y1, x2, y2)
                    )

            m_status = state.get("match_status", {}).get(track_id, "PENDING")
            vio_evt = self.violation_engine.get_violation(camera_id, track_id) if (self.violation_enabled and self.violation_engine) else None
            
            if vio_evt:
                # Violation Rendering (Red Box & Violation Metadata)
                color = (0, 0, 255)
                gid_display = vio_evt.global_id or state["global_assignments"].get(track_id, "PENDING")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                
                # Top Violation Banner
                vio_label = f"VIOLATION | GID: {gid_display}"
                cv2.putText(frame, vio_label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Bottom Attire vs Expected Badge
                sub_label = f"Attire: {vio_evt.actual_attire.upper()} | Expected: {vio_evt.expected_attire.upper()}"
                cv2.putText(frame, sub_label, (x1, min(h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            else:
                if m_status == "MATCHED":
                    gid = state["global_assignments"].get(track_id, "UNKNOWN")
                    color = (0, 255, 0)
                    label = f"Trk: {track_id} | GID: {gid}"
                elif m_status in ["PENDING", "AMBIGUOUS"]:
                    color = (0, 165, 255) # Orange
                    label = f"Trk: {track_id} | GID: {m_status}"
                else:
                    color = (255, 255, 0)
                    label = f"Trk: {track_id} | GID: UNKNOWN"
                
                # Draw the colored bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(30, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Draw Attire Badge below the bounding box if available
                if track_id in state["attire_assignments"]:
                    attire_info = state["attire_assignments"][track_id]
                    attire_text = f"[{attire_info['label']}: {int(attire_info['conf'] * 100)}%]"
                    badge_color = (0, 220, 0) if attire_info["is_formal"] else (0, 140, 255)
                    cv2.putText(frame, attire_text, (x1, min(h - 10, y2 + 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, badge_color, 2)
            
        # Draw raw detections in yellow (if tracker dropped them)
        for d in raw_detections:
            x1, y1, x2, y2, conf, cls = d
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"Det Only: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            
        return frame

    def stop(self):
        self.running = False
