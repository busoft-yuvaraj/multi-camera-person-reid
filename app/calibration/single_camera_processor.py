"""
Single-Camera Video Calibration and Tracking Processor
Implements:
- YOLO11 person detection
- ByteTrack / BoT-SORT local tracking
- Bottom-center footpoint extraction
- Homography-based ground-plane projection
- Common building coordinate mapping
- Spatial obstacle & walkable region validation
- Real-time dual-panel visualization (CCTV feed + Ground-plane trajectory)
- Video recording to calibration_result/
- JSON observation export to outputs/tracking_logs.json
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

from app.calibration.world_coordinate_mapper import WorldCoordinateMapper
from app.calibration.calibration_models import CameraCalibrationData

logger = logging.getLogger(__name__)


class SingleCameraCalibrationProcessor:
    """
    Processes a single CCTV video stream, tracks people, projects footpoints
    onto the calibrated ground plane, and generates synchronized dual-panel outputs.
    """

    def __init__(
        self,
        config_path: str = "config/camera_calibration.yaml",
        model_name: str = "yolo11n.pt",
        output_dir: str = "calibration_result",
        logs_dir: str = "outputs"
    ):
        self.config_path = config_path
        self.output_dir = output_dir
        self.logs_dir = logs_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # 1. Load Calibration Mapper
        self.mapper = WorldCoordinateMapper(config_path=self.config_path)

        # 2. Load YOLO11 Model
        logger.info(f"Loading YOLO11 detection model: {model_name}")
        self.model = YOLO(model_name)

        # Trajectory history: track_id -> list of (x_ground, y_ground, frame_idx, status)
        self.trajectories: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        # Tracking logs
        self.tracking_logs: List[Dict[str, Any]] = []

    def reset_trajectories(self):
        """Clears accumulated trajectory history."""
        self.trajectories.clear()
        self.tracking_logs.clear()

    def get_predefined_ground_plan_canvas(
        self,
        camera_id: str,
        canvas_size: Tuple[int, int] = (720, 720)
    ) -> np.ndarray:
        """
        Creates a crisp 2D ground-plane map for the camera showing walkable boundaries,
        obstacles, and calibration landmarks.
        """
        cam = self.mapper.cameras.get(camera_id)
        if not cam:
            canvas = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.uint8)
            cv2.putText(canvas, f"Unknown Camera: {camera_id}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return canvas

        # Base dark slate background
        w_can, h_can = canvas_size
        canvas = np.full((h_can, w_can, 3), 28, dtype=np.uint8)

        # Grid lines (every 20 units)
        grid_color = (45, 45, 52)
        margin = 60
        g_w = w_can - 2 * margin
        g_h = h_can - 2 * margin

        b = cam.boundaries
        x_min, x_max = b.get("x_min", 0.0), b.get("x_max", 100.0)
        y_min, y_max = b.get("y_min", 0.0), b.get("y_max", 100.0)

        def g2c(gx: float, gy: float) -> Tuple[int, int]:
            # Map ground (X, Y) to canvas (u, v)
            norm_x = (gx - x_min) / max(1.0, (x_max - x_min))
            norm_y = (gy - y_min) / max(1.0, (y_max - y_min))
            # Invert Y so 0 is at top or bottom depending on convention
            u = int(margin + norm_x * g_w)
            v = int(margin + (1.0 - norm_y) * g_h)
            return u, v

        # Draw grid
        for gx in np.linspace(x_min, x_max, 6):
            p1 = g2c(gx, y_min)
            p2 = g2c(gx, y_max)
            cv2.line(canvas, p1, p2, grid_color, 1)
            cv2.putText(canvas, f"{int(gx)}", (p1[0] - 10, h_can - margin + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 130), 1)

        for gy in np.linspace(y_min, y_max, 6):
            p1 = g2c(x_min, gy)
            p2 = g2c(x_max, gy)
            cv2.line(canvas, p1, p2, grid_color, 1)
            cv2.putText(canvas, f"{int(gy)}", (margin - 35, p1[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 130), 1)

        # Draw Walkable Regions
        surface_colors = {
            "tile": (80, 75, 45),     # muted cyan/blue
            "carpet": (60, 80, 50),   # muted green
            "wood": (40, 60, 95)      # muted bronze/wood
        }
        for wr in cam.walkable_regions:
            pts = np.array([g2c(p.x, p.y) for p in wr.polygon], dtype=np.int32)
            color = surface_colors.get(wr.surface_type, (60, 60, 60))
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.5, canvas, 0.5, 0, canvas)
            cv2.polylines(canvas, [pts], isClosed=True, color=(140, 180, 140), thickness=2)

        # Draw Obstacles
        for obs in cam.obstacles:
            if obs.obstacle_type == "circle" and obs.center and obs.radius:
                cu, cv = g2c(obs.center.x, obs.center.y)
                r_px = int(obs.radius / max(1.0, (x_max - x_min)) * g_w)
                cv2.circle(canvas, (cu, cv), r_px, (30, 30, 140), -1)
                cv2.circle(canvas, (cu, cv), r_px, (60, 60, 220), 2)
                cv2.putText(canvas, obs.id.upper(), (cu - 10, cv + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1)
            elif obs.obstacle_type == "polygon" and obs.polygon:
                pts = np.array([g2c(p.x, p.y) for p in obs.polygon], dtype=np.int32)
                cv2.fillPoly(canvas, [pts], (30, 30, 140))
                cv2.polylines(canvas, [pts], isClosed=True, color=(60, 60, 220), thickness=2)

        # Draw Landmarks
        for lm in cam.landmarks:
            lu, lv = g2c(lm.ground_point.x, lm.ground_point.y)
            cv2.circle(canvas, (lu, lv), 5, (0, 215, 255), -1)
            cv2.circle(canvas, (lu, lv), 7, (255, 255, 255), 1)
            cv2.putText(canvas, lm.id, (lu + 8, lv - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Header Title
        cv2.putText(canvas, f"GROUND-PLANE MAP: {cam.zone.upper()} ({cam.camera_id})",
                    (margin, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(canvas, f"Units: {cam.coordinate_convention.get('units', 'relative')} | Status: CALIBRATED",
                    (margin, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 220, 120), 1)

        return canvas

    def process_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        frame_idx: int,
        timestamp_sec: float,
        conf_threshold: float = 0.35,
        tracker_type: str = "bytetrack.yaml",
        active_track_filter: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """
        Processes a single video frame:
        1. Tracks persons using YOLO11 + tracker
        2. Projects footpoints to ground coordinates
        3. Renders Left Panel (annotated CCTV)
        4. Renders Right Panel (ground plane with moving markers and trajectory)
        5. Renders Synchronized Dual Panel (Left + Right combined)
        6. Logs tracking data
        
        Returns:
            (annotated_cctv, ground_plane_view, dual_synchronized_view, frame_observations)
        """
        h_orig, w_orig = frame.shape[:2]
        cctv_panel = frame.copy()

        # Predefined ground plan canvas
        gp_panel = self.get_predefined_ground_plan_canvas(camera_id, canvas_size=(720, 720))
        cam = self.mapper.cameras.get(camera_id)
        b = cam.boundaries if cam else {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100}
        margin = 60
        g_w = 720 - 2 * margin
        g_h = 720 - 2 * margin

        def g2c(gx: float, gy: float) -> Tuple[int, int]:
            norm_x = (gx - b.get("x_min", 0.0)) / max(1.0, (b.get("x_max", 100.0) - b.get("x_min", 0.0)))
            norm_y = (gy - b.get("y_min", 0.0)) / max(1.0, (b.get("y_max", 100.0) - b.get("y_min", 0.0)))
            u = int(margin + norm_x * g_w)
            v = int(margin + (1.0 - norm_y) * g_h)
            return u, v

        # 1. Run YOLO11 Tracking (classes=[0] for person)
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=tracker_type,
            classes=[0],
            conf=conf_threshold,
            verbose=False
        )

        frame_observations = []
        current_active_tracks = []

        # Distinct track colors (BGR)
        track_palette = [
            (0, 255, 128),   # Spring green
            (255, 165, 0),   # Orange
            (0, 215, 255),   # Gold
            (255, 0, 255),   # Magenta
            (147, 20, 255),  # Deep pink
            (255, 255, 0),   # Cyan
            (0, 128, 255),   # Coral
            (180, 105, 255)  # Purple
        ]

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                track_id = int(box.id[0]) if box.id is not None else -1
                conf = float(box.conf[0]) if box.conf is not None else 0.0
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                # Bottom-Center Footpoint
                u_foot = float((x1 + x2) / 2.0)
                v_foot = float(y2)

                # Homography & Coordinate Mapping
                mapping = self.mapper.validate_footpoint(
                    camera_id=camera_id,
                    u=u_foot,
                    v=v_foot
                )

                loc_g = mapping.get("local_ground")
                glob_g = mapping.get("global_building")
                ground_x = loc_g[0] if loc_g else None
                ground_y = loc_g[1] if loc_g else None
                global_x = glob_g[0] if glob_g else None
                global_y = glob_g[1] if glob_g else None

                obs = {
                    "camera_id": camera_id,
                    "frame_number": frame_idx,
                    "timestamp": round(timestamp_sec, 3),
                    "timestamp_str": time.strftime("%H:%M:%S", time.gmtime(timestamp_sec)) + f".{int((timestamp_sec % 1) * 1000):03d}",
                    "track_id": track_id,
                    "confidence": round(conf, 3),
                    "bounding_box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "footpoint_pixel": [round(u_foot, 1), round(v_foot, 1)],
                    "world_coordinates": {
                        "local_ground": [round(ground_x, 2), round(ground_y, 2)] if ground_x is not None else None,
                        "global_building": [round(global_x, 2), round(global_y, 2)] if global_x is not None else None
                    },
                    "calibration_validity": {
                        "is_valid": mapping["is_valid"],
                        "status": mapping["status"],
                        "region_name": mapping.get("region_name"),
                        "obstacle_name": mapping.get("obstacle_name")
                    }
                }
                frame_observations.append(obs)
                self.tracking_logs.append(obs)

                if track_id != -1 and ground_x is not None:
                    self.trajectories[track_id].append({
                        "ground_x": ground_x,
                        "ground_y": ground_y,
                        "global_x": global_x,
                        "global_y": global_y,
                        "frame_idx": frame_idx,
                        "timestamp": timestamp_sec,
                        "status": mapping["status"]
                    })
                    current_active_tracks.append(track_id)

                # --- ANNOTATE LEFT PANEL (CCTV) ---
                color = track_palette[track_id % len(track_palette)] if track_id != -1 else (0, 200, 255)
                cv2.rectangle(cctv_panel, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.circle(cctv_panel, (int(u_foot), int(v_foot)), 5, (0, 0, 255), -1)
                cv2.drawMarker(cctv_panel, (int(u_foot), int(v_foot)), (255, 255, 255),
                               markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)

                tag = f"ID:{track_id} ({conf:.2f})"
                if ground_x is not None:
                    tag += f" | G:[{ground_x:.1f}, {ground_y:.1f}]"
                
                (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(cctv_panel, (int(x1), int(y1) - 22), (int(x1) + tw + 6, int(y1)), color, -1)
                cv2.putText(cctv_panel, tag, (int(x1) + 3, int(y1) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1)

        # Overlay CCTV HUD info
        cv2.rectangle(cctv_panel, (15, 15), (420, 85), (20, 20, 20), -1)
        cv2.rectangle(cctv_panel, (15, 15), (420, 85), (60, 60, 60), 1)
        cv2.putText(cctv_panel, f"CAM: {camera_id.upper()}", (25, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 200), 2)
        cv2.putText(cctv_panel, f"Frame: {frame_idx:04d} | Time: {timestamp_sec:.2f}s | Active Tracks: {len(current_active_tracks)}",
                    (25, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(cctv_panel, "YOLO11 + ByteTrack | Homography G-Plane Gating",
                    (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 150), 1)

        # --- ANNOTATE RIGHT PANEL (Ground Plane) ---
        for tid, points in self.trajectories.items():
            if active_track_filter is not None and tid != active_track_filter:
                continue
            if len(points) < 2:
                continue

            color = track_palette[tid % len(track_palette)]
            canvas_pts = [g2c(pt["ground_x"], pt["ground_y"]) for pt in points]
            for i in range(1, len(canvas_pts)):
                cv2.line(gp_panel, canvas_pts[i - 1], canvas_pts[i], color, 2, cv2.LINE_AA)

            for pt in canvas_pts[::3]:
                cv2.circle(gp_panel, pt, 2, color, -1)

        for obs in frame_observations:
            tid = obs["track_id"]
            if active_track_filter is not None and tid != active_track_filter:
                continue

            loc_g = obs["world_coordinates"]["local_ground"]
            if loc_g is None:
                continue

            gx, gy = loc_g
            cu, cv = g2c(gx, gy)
            color = track_palette[tid % len(track_palette)] if tid != -1 else (0, 200, 255)

            cv2.circle(gp_panel, (cu, cv), 12, color, 2)
            cv2.circle(gp_panel, (cu, cv), 6, color, -1)
            cv2.circle(gp_panel, (cu, cv), 2, (255, 255, 255), -1)

            glob_g = obs["world_coordinates"]["global_building"]
            lbl = f"P{tid} ({gx:.1f}, {gy:.1f})"
            cv2.putText(gp_panel, lbl, (cu + 12, cv - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            if glob_g:
                cv2.putText(gp_panel, f"Bld:[{glob_g[0]:.1f}, {glob_g[1]:.1f}]", (cu + 12, cv + 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 200), 1)

        # --- COMPOSITE SYNCHRONIZED DUAL-PANEL VIEW ---
        cctv_resized = cv2.resize(cctv_panel, (1280, 720))
        dual_panel = np.hstack([cctv_resized, gp_panel])

        return cctv_panel, gp_panel, dual_panel, frame_observations

    def process_video(
        self,
        camera_id: str,
        video_path: str,
        start_frame: int = 0,
        max_frames: Optional[int] = None,
        conf_threshold: float = 0.35,
        save_videos: bool = True
    ) -> Dict[str, Any]:
        """
        Runs full video processing pipeline on a single CCTV video.
        Saves outputs to:
        - calibration_result/{camera_id}_live_cctv_processed.mp4
        - calibration_result/{camera_id}_ground_plane_trajectory.mp4
        - calibration_result/{camera_id}_synchronized_dual_view.mp4
        - outputs/tracking_logs.json
        - outputs/calibration_metadata.json
        """
        assert os.path.exists(video_path), f"Video file not found: {video_path}"
        self.reset_trajectories()

        cap = cv2.VideoCapture(video_path)
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(f"Processing video: {video_path} (CAM: {camera_id}, Start: {start_frame}, FPS: {fps:.2f}, Total: {total_frames})")

        # Output video writers
        cctv_out_path = os.path.join(self.output_dir, f"{camera_id}_live_cctv_processed.mp4")
        gp_out_path = os.path.join(self.output_dir, f"{camera_id}_ground_plane_trajectory.mp4")
        dual_out_path = os.path.join(self.output_dir, f"{camera_id}_synchronized_dual_view.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer_cctv = cv2.VideoWriter(cctv_out_path, fourcc, fps, (w_in, h_in)) if save_videos else None
        writer_gp = cv2.VideoWriter(gp_out_path, fourcc, fps, (720, 720)) if save_videos else None
        writer_dual = cv2.VideoWriter(dual_out_path, fourcc, fps, (1280 + 720, 720)) if save_videos else None

        frame_idx = 0
        start_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_frame = start_frame + frame_idx
            timestamp_sec = current_frame / fps

            cctv_p, gp_p, dual_p, obs = self.process_frame(
                camera_id=camera_id,
                frame=frame,
                frame_idx=current_frame,
                timestamp_sec=timestamp_sec,
                conf_threshold=conf_threshold
            )

            if save_videos:
                if writer_cctv:
                    writer_cctv.write(cctv_p)
                if writer_gp:
                    writer_gp.write(gp_p)
                if writer_dual:
                    writer_dual.write(dual_p)

            frame_idx += 1
            if max_frames and frame_idx >= max_frames:
                break

        cap.release()
        if writer_cctv:
            writer_cctv.release()
        if writer_gp:
            writer_gp.release()
        if writer_dual:
            writer_dual.release()

        elapsed = time.time() - start_time
        fps_proc = frame_idx / max(0.001, elapsed)

        # 3. Save JSON Tracking Logs
        json_log_path = os.path.join(self.logs_dir, "tracking_logs.json")
        with open(json_log_path, "w") as f:
            json.dump({
                "camera_id": camera_id,
                "video_file": os.path.basename(video_path),
                "total_frames_processed": frame_idx,
                "processing_fps": round(fps_proc, 2),
                "total_observations": len(self.tracking_logs),
                "unique_tracks": list(self.trajectories.keys()),
                "observations": self.tracking_logs
            }, f, indent=2)

        # 4. Save Calibration Metadata & Homography Matrices
        meta_log_path = os.path.join(self.logs_dir, "calibration_metadata.json")
        cal_res = self.mapper.calibration_results.get(camera_id)
        with open(meta_log_path, "w") as f:
            json.dump({
                "camera_id": camera_id,
                "is_calibrated": cal_res.is_valid if cal_res else False,
                "rms_reprojection_error_px": cal_res.rms_reprojection_error if cal_res else None,
                "max_reprojection_error_px": cal_res.max_reprojection_error if cal_res else None,
                "condition_number": cal_res.condition_number if cal_res else None,
                "H_ground2img": cal_res.H_ground2img.tolist() if cal_res else None,
                "H_img2ground": cal_res.H_img2ground.tolist() if cal_res else None,
                "landmarks": cal_res.per_point_errors if cal_res else [],
                "global_transform": {
                    "translation": [self.mapper.cameras[camera_id].global_transform.translation.x, self.mapper.cameras[camera_id].global_transform.translation.y],
                    "rotation_deg": self.mapper.cameras[camera_id].global_transform.rotation_deg,
                    "scale": self.mapper.cameras[camera_id].global_transform.scale
                } if camera_id in self.mapper.cameras else None
            }, f, indent=2)

        logger.info(
            f"Video calibration processing complete for {camera_id}: "
            f"{frame_idx} frames in {elapsed:.1f}s ({fps_proc:.1f} FPS). Logs saved to {json_log_path}"
        )

        return {
            "camera_id": camera_id,
            "frames_processed": frame_idx,
            "elapsed_seconds": round(elapsed, 2),
            "processing_fps": round(fps_proc, 2),
            "total_tracks": len(self.trajectories),
            "total_observations": len(self.tracking_logs),
            "output_videos": {
                "live_cctv": cctv_out_path,
                "ground_plane": gp_out_path,
                "synchronized_dual": dual_out_path
            },
            "output_logs": {
                "tracking_logs": json_log_path,
                "calibration_metadata": meta_log_path
            }
        }
