import cv2
import numpy as np
import threading
import time
import os
import datetime
import logging

from app.reid.quality import calculate_quality
from app.reid.embedding_bank import TrackEmbeddingBank
from ultralytics import YOLO

class CameraProcessor(threading.Thread):
    def __init__(self, camera_id, video_path, output_path, config, extractor, gallery, matcher, attire_extractor=None):
        super().__init__()
        self.camera_id = camera_id
        self.video_path = video_path
        
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
        self.attire_extractor = attire_extractor
        self.gallery = gallery
        self.matcher = matcher
        
        self.yolo = YOLO(config["model_path"])
        # We'll use ultralytics internal tracking
        self.banks = {}
        self.global_assignments = {} # track_id -> global_id
        self.attire_assignments = {}
        self.attire_history = {}
        
        self.running = True

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logging.error(f"[{self.camera_id}] Error opening video.")
            return
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        out = cv2.VideoWriter(self.output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w * 2, h))
        
        frame_idx = 0
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            original_frame = frame.copy()
            frame_idx += 1
            
            # 1 & 2. Detection and Tracking
            results = self.yolo.track(
                frame, 
                classes=[0], 
                tracker=self.config.get("tracker_config", "botsort.yaml"), 
                persist=True, 
                verbose=False,
                conf=0.25, # Lowered confidence threshold to detect harder-to-see people
                imgsz=self.config.get("imgsz", 640)
            )
            
            tracks = []
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                clss = results[0].boxes.cls.cpu().numpy()
                for b, tid, conf, cls in zip(boxes, track_ids, confs, clss):
                    tracks.append([*b, tid, conf, cls])
            
            # 3. Processing Tracks
            for t in tracks:
                x1, y1, x2, y2, track_id, conf, cls = t
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Trk: {int(track_id)}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
                
                # Formal Attire Detection
                if self.attire_extractor and self.config.get("attire_detection_enabled", True) and crop.size > 0:
                    if frame_idx % self.config.get("attire_interval", 2) == 0 or track_id not in self.attire_assignments:
                        attire_res = self.attire_extractor.extract_attire(crop)
                        if attire_res["label"] != "unknown":
                            if track_id not in self.attire_history:
                                self.attire_history[track_id] = []
                            self.attire_history[track_id].append(attire_res)
                            if len(self.attire_history[track_id]) > 10:
                                self.attire_history[track_id].pop(0)
                            
                            formal_count = sum(1 for a in self.attire_history[track_id] if a.get("is_formal") is True)
                            informal_count = sum(1 for a in self.attire_history[track_id] if a.get("is_formal") is False)
                            avg_conf = float(np.mean([a["conf"] for a in self.attire_history[track_id]]))
                            is_formal = formal_count >= informal_count
                            self.attire_assignments[track_id] = {
                                "label": "Formal" if is_formal else "Informal",
                                "conf": avg_conf,
                                "is_formal": is_formal
                            }
                
                if self.config["reid_enabled"] and frame_idx % self.config["reid_interval"] == 0:
                    if crop.size > 0:
                        quality, accepted, reason = calculate_quality(
                            crop, conf, 
                            self.config["min_reid_confidence"],
                            self.config["min_reid_width"],
                            self.config["min_reid_height"],
                            self.config["blur_threshold"],
                            w, h, (x1, y1, x2, y2)
                        )
                        
                        if accepted:
                            embedding = self.extractor.extract([crop])[0]
                            if track_id not in self.banks:
                                self.banks[track_id] = TrackEmbeddingBank(
                                    self.camera_id, track_id, 
                                    max_size_per_view=self.config["reid_bank_size"],
                                    diversity_threshold=self.config["embedding_diversity_threshold"]
                                )
                                
                            self.banks[track_id].add_embedding(embedding, quality, frame_idx)
                            
                            # Match logic
                            if track_id not in self.global_assignments:
                                bank_embeddings = self.banks[track_id].get_all_embeddings()
                                gid, sim = self.matcher.match(bank_embeddings)
                                if gid:
                                    self.global_assignments[track_id] = gid
                                    self.gallery.update_identity(gid, [{"camera_id": self.camera_id, "track_id": track_id}], bank_embeddings)
                                    logging.info(f"[{self.camera_id}] Track {track_id} MATCHED to Global ID {gid} (sim: {sim:.2f})")
                                elif len(bank_embeddings) >= self.config["reid_confirm_frames"]:
                                    new_gid = self.gallery.add_identity([{"camera_id": self.camera_id, "track_id": track_id}], bank_embeddings)
                                    self.global_assignments[track_id] = new_gid
                                    logging.info(f"[{self.camera_id}] Track {track_id} CONFIRMED as NEW Global ID {new_gid}")
                                    
                # Draw Global ID & Attire Badge
                gid = self.global_assignments.get(track_id, "UNKNOWN")
                cv2.putText(frame, f"GID: {gid}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                if track_id in self.attire_assignments:
                    attire_info = self.attire_assignments[track_id]
                    attire_badge = f"[{attire_info['label']}: {int(attire_info['conf'] * 100)}%]"
                    attire_col = (0, 255, 0) if attire_info["is_formal"] else (0, 140, 255)
                    cv2.putText(frame, attire_badge, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, attire_col, 2)
                
            side_by_side = np.hstack((original_frame, frame))
            out.write(side_by_side)
            
            # Display the frame visually
            cv2.imshow(self.camera_id, side_by_side)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
            
        cap.release()
        out.release()
        cv2.destroyWindow(self.camera_id)
        logging.info(f"[{self.camera_id}] Finished processing.")

    def stop(self):
        self.running = False
