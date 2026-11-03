"""
Multi-Camera Person Tracking & Re-Identification Processor.
Production-ready, SOLID-compliant orchestrator supporting both synchronous and asynchronous execution.
"""

import os
import cv2
import copy
import time
import math
import asyncio
import logging
import datetime
from typing import Optional, Dict, List, Any, Set, Tuple

import numpy as np
from ultralytics import YOLO

from app.reid.quality import calculate_quality
from app.reid.embedding_bank import TrackEmbeddingBank
from app.events.transition_detector import TransitionDetector, TransitionEvent
from app.events.roi_detector import (
    RoiDetector,
    RoiEvent,
    FrameResolutionMismatchError,
    DEFAULT_CAMERA_RESOLUTIONS,
    parse_resolution,
)

logger = logging.getLogger("multi_camera_processor")


# =============================================================================
# Helper Component: Spatial Geometry & Duplicate Suppression (Single Responsibility)
# =============================================================================

class SpatialGeometryHelper:
    """Utility class for bounding box geometry and IoU calculations."""

    @staticmethod
    def compute_iou(box_a: Tuple[float, float, float, float], box_b: Tuple[float, float, float, float]) -> float:
        """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        x1 = max(float(box_a[0]), float(box_b[0]))
        y1 = max(float(box_a[1]), float(box_b[1]))
        x2 = min(float(box_a[2]), float(box_b[2]))
        y2 = min(float(box_a[3]), float(box_b[3]))

        inter_w = max(0.0, x2 - x1)
        inter_h = max(0.0, y2 - y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0

        area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
        area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
        union_area = area_a + area_b - inter_area
        return float(inter_area / union_area) if union_area > 0.0 else 0.0

    @staticmethod
    def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculates bottom-center foot contact point."""
        return float((bbox[0] + bbox[2]) / 2.0), float(bbox[3])


# =============================================================================
# Helper Component: Multi-Camera Frame Annotator & Visualizer (Single Responsibility)
# =============================================================================

class MultiCameraAnnotator:
    """Handles rendering of ROI polygons, transition tripwires, bounding boxes, and status badges."""

    @staticmethod
    def draw_phase1_annotations(
        camera_id: str,
        frame: np.ndarray,
        h: int,
        w: int,
        tracks: list,
        state: dict,
        cam_zone: str,
        roi_detectors: Dict[str, RoiDetector],
        transition_detectors: Dict[str, TransitionDetector],
        journey_manager: Optional[Any],
        association_manager: Optional[Any],
        config: Dict[str, Any]
    ) -> np.ndarray:
        # 1. Draw ROIs
        if camera_id in roi_detectors:
            for roi_id, poly in roi_detectors[camera_id].polygons.items():
                cv2.polylines(frame, [poly], isClosed=True, color=(255, 200, 0), thickness=2)
                M = cv2.moments(poly)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = int(poly[0][0][0]), int(poly[0][0][1])
                cv2.putText(
                    frame, f"ROI: {roi_id.upper()}", (max(10, cx - 40), max(20, cy)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 0), 2
                )

        # 2. Draw Transition Lines
        if camera_id in transition_detectors:
            for trans_id, line_info in transition_detectors[camera_id].lines.items():
                p1 = (int(line_info["start"][0]), int(line_info["start"][1]))
                p2 = (int(line_info["end"][0]), int(line_info["end"][1]))
                line_col = (0, 255, 255) if "yellow" in trans_id else ((0, 255, 0) if "green" in trans_id else (0, 165, 255))
                cv2.line(frame, p1, p2, line_col, 3)
                cv2.circle(frame, p1, 5, line_col, -1)
                cv2.circle(frame, p2, 5, line_col, -1)
                mid_x, mid_y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
                cv2.putText(
                    frame, trans_id, (mid_x - 30, max(20, mid_y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, line_col, 2
                )

        # 3. Draw Track Boxes & Journey Badges
        rendered_gids = set()
        for t in tracks:
            x1, y1, x2, y2, track_id, conf, cls = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            foot_pt = (int((x1 + x2) / 2), int(y2))
            cv2.circle(frame, foot_pt, 4, (0, 255, 255), -1)

            gid = state.get("global_assignments", {}).get(track_id)
            m_status = state.get("match_status", {}).get(track_id, "PENDING")

            # Guard against duplicate GID rendering on multiple bounding boxes in the same frame
            if gid and gid in rendered_gids:
                gid = None
                m_status = "PENDING"
            elif gid and m_status == "MATCHED":
                rendered_gids.add(gid)

            ident_state = journey_manager.get_identity(gid) if (gid and journey_manager) else None
            last_trans = ident_state.last_transition_id if ident_state else None
            curr_roi = ident_state.current_roi if ident_state else None

            if m_status == "MATCHED" and gid:
                color = (0, 255, 0)
                top_text = f"GID: {gid} | L{track_id} | CONFIRMED"
            elif m_status == "PENDING":
                color = (0, 165, 255)  # Orange
                cand = association_manager.pending_candidate.get((camera_id, track_id)) if association_manager else None
                if cand:
                    cnt = association_manager.pending_counters.get((camera_id, track_id), 1)
                    pending_thresh = int(config.get("pending_confirm_frames", 3))
                    top_text = f"PENDING ({cand} {cnt}/{pending_thresh}) | L{track_id}"
                else:
                    top_text = f"PENDING | L{track_id}"
            elif m_status == "DUPLICATE":
                color = (128, 128, 128)  # Gray for suppressed duplicate ghost tracks
                top_text = f"DUPLICATE | L{track_id}"
            else:
                color = (255, 255, 0)  # Cyan/Yellow
                top_text = f"NEW/UNKNOWN | L{track_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, top_text, (x1, max(25, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

            sub_text = f"Zone: {cam_zone} | ROI: {curr_roi or 'None'} | Last: {last_trans or 'None'}"
            cv2.putText(frame, sub_text, (x1, min(h - 10, y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 2)

        return frame


# =============================================================================
# Core Orchestrator: MultiCameraProcessor 
# =============================================================================

class MultiCameraProcessor:

    def __init__(
        self,
        cameras_info: List[Dict[str, Any]],
        output_path: str,
        config: Dict[str, Any],
        extractor: Any,
        par_extractor: Any,
        pose_extractor: Any,
        gallery: Optional[Any] = None,
        matcher: Optional[Any] = None,
        association_manager: Optional[Any] = None,
        journey_manager: Optional[Any] = None,
        store: Optional[Any] = None,
        **kwargs: Any
    ):
        self.cameras_info = cameras_info
        self.config = config
        self.extractor = extractor
        self.par_extractor = par_extractor
        self.pose_extractor = pose_extractor
        self.gallery = gallery
        self.matcher = matcher
        self.association_manager = association_manager
        self.journey_manager = journey_manager
        self.store = store

        # Format output video path with timestamp
        base_name, ext = os.path.splitext(output_path)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = f"{base_name}_{timestamp}{ext}"
        out_dir = os.path.dirname(self.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        self.phase1_mode = bool(config.get("phase1_mode", True) or (association_manager is not None))
        self.raise_on_resolution_error = bool(config.get("raise_on_resolution_error", True))
        self.running = True

        # Camera streams & state initialization
        self.transition_detectors: Dict[str, TransitionDetector] = {}
        self.roi_detectors: Dict[str, RoiDetector] = {}
        self.expected_resolutions: Dict[str, Tuple[int, int]] = {}
        self.state: Dict[str, Dict[str, Any]] = {}

        self._init_camera_states()

    def _init_camera_states(self):
        cam_cfgs = self.config.get("cameras", {})
        model_path = self.config.get("model_path", "yolo11n.pt")

        for cam in self.cameras_info:
            c_id = cam["camera_id"]
            c_cfg = cam_cfgs.get(c_id, {})
            c_zone = c_cfg.get("zone", c_id)
            v_path = cam.get("video_path", "")

            # Robust path validation (without hardcoded string hacks)
            if v_path and not os.path.exists(v_path):
                logger.warning(f"[{c_id}] Video source '{v_path}' not found at specified path.")

            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                logger.warning(f"[{c_id}] Could not open video capture for '{v_path}'.")

            c_fps = cap.get(cv2.CAP_PROP_FPS)
            if not c_fps or c_fps <= 0 or np.isnan(c_fps):
                c_fps = float(c_cfg.get("fps", 30.0))
            start_offset = float(c_cfg.get("start_offset", c_cfg.get("offset_seconds", 0.0)))
            c_crop = c_cfg.get("crop")

            # Setup ROI and transition geometry with coordinate auto-scaling if cropped
            roi_cfg = copy.deepcopy(c_cfg.get("roi", {}))
            trans_cfg = copy.deepcopy(c_cfg.get("transitions", {}))
            if c_crop:
                cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
                cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
                cx1, cy1, cx2, cy2 = c_crop
                cw = max(1, cx2 - cx1)
                ch = max(1, cy2 - cy1)
                sx = cap_w / float(cw)
                sy = cap_h / float(ch)

                for _, r_info in roi_cfg.items():
                    if "points" in r_info:
                        r_info["points"] = [
                            [int(round((pt[0] - cx1) * sx)), int(round((pt[1] - cy1) * sy))]
                            for pt in r_info["points"]
                        ]
                for _, t_info in trans_cfg.items():
                    if "points" in t_info:
                        pts = t_info["points"]
                        if "start" in pts and "end" in pts:
                            pts["start"] = [int(round((pts["start"][0] - cx1) * sx)), int(round((pts["start"][1] - cy1) * sy))]
                            pts["end"] = [int(round((pts["end"][0] - cx1) * sx)), int(round((pts["end"][1] - cy1) * sy))]

            c_expected_res = parse_resolution(
                c_cfg.get("expected_resolution") or c_cfg.get("resolution") or roi_cfg.get("expected_resolution")
            ) or DEFAULT_CAMERA_RESOLUTIONS.get(c_id, (1280, 720))
            self.expected_resolutions[c_id] = c_expected_res

            self.transition_detectors[c_id] = TransitionDetector(
                camera_id=c_id,
                zone=c_zone,
                transitions_config=trans_cfg,
                cooldown_frames=int(self.config.get("transition_cooldown_frames", 15))
            )
            self.roi_detectors[c_id] = RoiDetector(
                camera_id=c_id,
                zone=c_zone,
                roi_config=roi_cfg,
                debounce_frames=int(self.config.get("roi_debounce_frames", 3)),
                expected_resolution=c_expected_res
            )

            # Independent state tracking dictionary per camera
            self.state[c_id] = {
                "cap": cap,
                "video_path": v_path,
                "zone": c_zone,
                "crop": c_crop,
                "fps": float(c_fps),
                "start_offset": float(start_offset),
                "frame_count": 0,
                "current_presentation_time": 0.0,
                "current_shared_time": float(start_offset),
                "expected_resolution": c_expected_res,
                "expected_original_resolution": c_expected_res,
                "resolution_error": None,
                "model": YOLO(model_path),
                "banks": {},
                "global_assignments": {},
                "match_status": {},
                "duplicate_tracks": {},
                "last_frame": None,
                "finished": False,
                "detections": 0,
                "active_tracks": 0,
                "reid_extractions": 0,
                "global_ids_set": set(),
                "last_seen_frame": {},
                "last_seen_time": {},
                "local_id_map": {},
                "next_local_id": 1
            }

    # =========================================================================
    # Frame Resolution Validation
    # =========================================================================

    def validate_camera_frame(self, camera_id: str, frame: Any, stage: str = "evaluation") -> bool:
        """
        Validates frame dimensions against the expected resolution for the given camera.
        Raises FrameResolutionMismatchError if the frame is invalid or dimensions do not match.
        """
        expected_res = self.expected_resolutions.get(camera_id) or DEFAULT_CAMERA_RESOLUTIONS.get(camera_id, (1280, 720))
        if frame is None:
            raise FrameResolutionMismatchError(
                camera_id=camera_id,
                expected_resolution=expected_res,
                actual_resolution=(0, 0),
                message=f"[{camera_id}] Received None frame at {stage} stage: frame is unreadable or empty."
            )
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise FrameResolutionMismatchError(
                camera_id=camera_id,
                expected_resolution=expected_res,
                actual_resolution=(0, 0),
                message=f"[{camera_id}] Received invalid frame with shape {getattr(frame, 'shape', None)} at {stage} stage."
            )
        act_h, act_w = int(frame.shape[0]), int(frame.shape[1])
        if act_h <= 0 or act_w <= 0:
            raise FrameResolutionMismatchError(
                camera_id=camera_id,
                expected_resolution=expected_res,
                actual_resolution=(act_w, act_h),
                message=f"[{camera_id}] Received empty frame with zero dimensions ({act_w}x{act_h}) at {stage} stage."
            )
        if (act_w, act_h) != expected_res:
            raise FrameResolutionMismatchError(
                camera_id=camera_id,
                expected_resolution=expected_res,
                actual_resolution=(act_w, act_h),
                message=(
                    f"Frame resolution mismatch for camera '{camera_id}' at {stage} stage: "
                    f"expected {expected_res[0]}x{expected_res[1]} (WxH), but got {act_w}x{act_h} (WxH). "
                    f"ROI coordinates cannot be safely applied."
                )
            )
        return True

    # =========================================================================
    # Duplicate Bounding Box Suppression
    # =========================================================================

    def _suppress_duplicate_tracks(
        self,
        camera_id: str,
        tracks: list,
        state: dict,
        frame_idx: int,
        current_time: float,
        iou_threshold: float = 0.60
    ) -> Tuple[list, list]:
        """
        Suppresses duplicate / ghost bounding boxes on the same physical person in a single camera.
        Prevents multiple local tracks from claiming distinct Global IDs for the same individual.
        """
        if "duplicate_tracks" not in state:
            state["duplicate_tracks"] = {}

        if len(tracks) <= 1:
            return tracks, []

        primary_tracks = []
        suppressed_tracks = []
        suppressed_ids = set()

        for i in range(len(tracks)):
            t_i = tracks[i]
            tid_i = int(t_i[4])
            if tid_i in suppressed_ids:
                continue

            gid_i = state.get("global_assignments", {}).get(tid_i)
            status_i = state.get("match_status", {}).get(tid_i)

            for j in range(i + 1, len(tracks)):
                t_j = tracks[j]
                tid_j = int(t_j[4])
                if tid_j in suppressed_ids:
                    continue

                gid_j = state.get("global_assignments", {}).get(tid_j)
                status_j = state.get("match_status", {}).get(tid_j)

                # If both are confirmed to DIFFERENT global IDs, they are two distinct physical people
                if gid_i and gid_j and gid_i != gid_j and status_i == "MATCHED" and status_j == "MATCHED":
                    continue

                iou = SpatialGeometryHelper.compute_iou(t_i[:4], t_j[:4])
                if iou >= iou_threshold:
                    # Select primary track
                    if status_i == "MATCHED" and status_j != "MATCHED":
                        supp_track, prim_track = t_j, t_i
                    elif status_j == "MATCHED" and status_i != "MATCHED":
                        supp_track, prim_track = t_i, t_j
                    elif float(t_i[5]) >= float(t_j[5]):
                        supp_track, prim_track = t_j, t_i
                    else:
                        supp_track, prim_track = t_i, t_j

                    s_id = int(supp_track[4])
                    p_id = int(prim_track[4])
                    suppressed_ids.add(s_id)
                    suppressed_tracks.append(supp_track)
                    state["duplicate_tracks"][s_id] = p_id
                    state["match_status"][s_id] = "DUPLICATE"

        for t in tracks:
            tid = int(t[4])
            if tid not in suppressed_ids:
                primary_tracks.append(t)

        return primary_tracks, suppressed_tracks

    # =========================================================================
    # Frame Processing Engine
    # =========================================================================

    def _process_frame(self, camera_id: str, frame: np.ndarray, frame_idx: int, w: int, h: int) -> np.ndarray:
        """Processes a single camera frame through tracking, event detection, ReID, and association."""
        state = self.state[camera_id]

        # Resolution check before applying ROIs
        if camera_id in self.roi_detectors:
            self.roi_detectors[camera_id].validate_frame(frame)

        # 1. YOLOv11 Detection + BoT-SORT Tracking
        results = state["model"].track(
            frame,
            classes=[0],
            tracker=self.config.get("tracker_config", "botsort.yaml"),
            persist=True,
            verbose=False,
            conf=self.config.get("detection_confidence", 0.25),
            imgsz=self.config.get("imgsz", 960)
        )

        tracks = []
        raw_detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy()

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
                for b, conf, cls in zip(boxes, confs, clss):
                    raw_detections.append([*b, conf, cls])

        # 2. Duplicate Ghost Bounding Box Suppression
        tracks, _ = self._suppress_duplicate_tracks(camera_id, tracks, state, frame_idx, state.get("current_shared_time", 0.0))

        # 3. Exclusion Zone Filtering
        exclusion_zones = self.config.get("exclusion_zones", {}).get(camera_id, [])
        if exclusion_zones:
            valid_tracks = []
            for t in tracks:
                cx, cy = (t[0] + t[2]) / 2.0 / float(w), (t[1] + t[3]) / 2.0 / float(h)
                if not any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in exclusion_zones):
                    valid_tracks.append(t)
            tracks = valid_tracks

            valid_raw = []
            for d in raw_detections:
                cx, cy = (d[0] + d[2]) / 2.0 / float(w), (d[1] + d[3]) / 2.0 / float(h)
                if not any(zx1 <= cx <= zx2 and zy1 <= cy <= zy2 for zx1, zy1, zx2, zy2 in exclusion_zones):
                    valid_raw.append(d)
            raw_detections = valid_raw

        # 4. track_only_inside_roi Gating
        allowed_rois = self.config.get("cameras", {}).get(camera_id, {}).get("track_only_inside_roi", [])
        if allowed_rois and camera_id in self.roi_detectors:
            roi_det = self.roi_detectors[camera_id]
            roi_det.validate_frame(frame)
            valid_tracks = []
            for t in tracks:
                fx, fy = (t[0] + t[2]) / 2.0, t[3]
                if any(roi_det.is_point_in_roi((fx, fy), r_id) for r_id in allowed_rois):
                    valid_tracks.append(t)
            tracks = valid_tracks

            valid_raw = []
            for d in raw_detections:
                fx, fy = (d[0] + d[2]) / 2.0, d[3]
                if any(roi_det.is_point_in_roi((fx, fy), r_id) for r_id in allowed_rois):
                    valid_raw.append(d)
            raw_detections = valid_raw

        state["detections"] += len(tracks) + len(raw_detections)
        state["active_tracks"] += len(tracks)

        # 5. Presentation Timeline Update
        state["frame_count"] += 1
        fps = state.get("fps", 30.0)
        presentation_time = float(state["frame_count"] - 1) / float(fps)
        shared_time = presentation_time + state.get("start_offset", 0.0)
        current_time = shared_time
        state["current_presentation_time"] = presentation_time
        state["current_shared_time"] = shared_time

        current_track_ids = {int(t[4]) for t in tracks}
        for tid in current_track_ids:
            state["last_seen_frame"][tid] = frame_idx
            state["last_seen_time"][tid] = current_time

        # 6. Active Global ID Occupancy Across Cameras
        cam_zone = state.get("zone", camera_id)
        other_active_gids = set()
        for other_cid, other_st in self.state.items():
            if other_cid != camera_id:
                for o_tid, o_last_time in other_st.get("last_seen_time", {}).items():
                    if abs(current_time - o_last_time) <= 0.50:
                        o_gid = other_st.get("global_assignments", {}).get(o_tid)
                        if o_gid:
                            other_active_gids.add(o_gid)

        # 7. Disappearance & Track Loss Handling
        track_lost_frames = int(self.config.get("track_lost_frames", 60))
        for tid, l_frame in list(state["last_seen_frame"].items()):
            if tid not in current_track_ids and (frame_idx - l_frame > track_lost_frames):
                if camera_id in self.roi_detectors:
                    self.roi_detectors[camera_id].on_track_lost(tid, frame_idx, current_time)
                if self.journey_manager:
                    self.journey_manager.on_track_lost(camera_id, cam_zone, tid, current_time, frame_idx)
                state["global_assignments"].pop(tid, None)
                state["last_seen_frame"].pop(tid, None)
                state["last_seen_time"].pop(tid, None)

        if self.journey_manager and frame_idx % 30 == 0:
            self.journey_manager.check_expirations(current_time, frame_idx)

        # 8. Principal Sorting of Tracks
        confirm_frames = int(self.config.get("reid_confirm_frames", 3))
        def track_sort_key(t):
            tid = int(t[4])
            if tid in state["global_assignments"]:
                return (2, 1.0)
            if tid in state["banks"]:
                embs = state["banks"][tid].get_all_embeddings()
                if len(embs) >= confirm_frames:
                    return (1, 1.0)
            return (0, 0.0)

        sorted_tracks = sorted(tracks, key=track_sort_key, reverse=True)

        # 9. Batch ReID Evaluation & Feature Extraction Optimization
        reid_enabled = self.config.get("reid_enabled", True)
        reid_interval = int(self.config.get("reid_interval", 3))
        should_extract_reid = reid_enabled and (frame_idx % reid_interval == 0)

        tracks_for_reid = []
        if should_extract_reid:
            for t in sorted_tracks:
                x1, y1, x2, y2, track_id, conf, cls = t
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                track_id = int(track_id)
                crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
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
                        tracks_for_reid.append((track_id, crop, quality, metadata))

        # Single batched forward pass for all accepted crops in this frame (2x-3x speedup on CPU)
        batched_embeddings = {}
        if tracks_for_reid:
            state["reid_extractions"] += len(tracks_for_reid)
            crops_batch = [item[1] for item in tracks_for_reid]
            try:
                embeddings = self.extractor.extract(crops_batch)
                for (tid, crop, q, meta), emb in zip(tracks_for_reid, embeddings):
                    batched_embeddings[tid] = (emb, crop, q, meta)
            except Exception as e:
                logger.warning(f"Batched extraction fallback to sequential: {e}")
                for tid, crop, q, meta in tracks_for_reid:
                    emb = self.extractor.extract([crop])[0]
                    batched_embeddings[tid] = (emb, crop, q, meta)

        # 10. Iterate Each Active Track for Events, Bank Buffering, and Association
        for t in sorted_tracks:
            x1, y1, x2, y2, track_id, conf, cls = t
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            track_id = int(track_id)
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

            # ROI Events
            if camera_id in self.roi_detectors:
                roi_events = self.roi_detectors[camera_id].check_rois(
                    local_track_id=track_id,
                    bbox=(x1, y1, x2, y2),
                    frame_idx=frame_idx,
                    timestamp=current_time,
                    global_id=state["global_assignments"].get(track_id),
                    frame=frame
                )
                for revt in roi_events:
                    if self.journey_manager:
                        self.journey_manager.record_roi_event(revt)

            # Transition Line Events
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

            # ReID Embedding Bank Buffering
            added = False
            quality = 0.0
            if track_id in batched_embeddings:
                embedding, crop, quality, metadata = batched_embeddings[track_id]

                # Gated PAR Feature extraction: reuse existing PAR profile if track is already established
                bank = state["banks"].get(track_id)
                if bank is not None and getattr(bank, "par_attributes", None):
                    par_attr = bank.par_attributes
                    par_features = getattr(bank, "par_features", None)
                else:
                    par_attr = self.par_extractor.extract_attributes(crop) if hasattr(self.par_extractor, "extract_attributes") else {}
                    par_features = par_attr.get("features", self.par_extractor.extract(crop))

                viewpoint, vp_conf = self.pose_extractor.extract_viewpoint(crop)
                metadata.update({
                    "viewpoint_confidence": float(vp_conf),
                    "viewpoint": viewpoint,
                    "camera_id": camera_id,
                    "zone": cam_zone,
                    "track_id": track_id,
                    "timestamp": current_time
                })

                if track_id not in state["banks"]:
                    state["banks"][track_id] = TrackEmbeddingBank(
                        camera_id, track_id,
                        max_size_per_view=self.config.get("reid_bank_size", 15),
                        diversity_threshold=self.config.get("embedding_diversity_threshold", 0.90)
                    )
                    state["match_status"][track_id] = "PENDING"

                added, _, _ = state["banks"][track_id].add_embedding(
                    embedding=embedding,
                    quality=quality,
                    frame_id=frame_idx,
                    viewpoint=viewpoint,
                    par_features=par_features,
                    quality_metadata=metadata,
                    par_attributes=par_attr
                )

            # Two-Stage Association
            if track_id in state["banks"] and self.association_manager is not None:
                bank_embeddings = state["banks"][track_id].get_all_embeddings()
                if bank_embeddings:
                    same_cam_occupied = {
                        existing_gid for other_tid, existing_gid in state["global_assignments"].items()
                        if other_tid in current_track_ids and other_tid != track_id
                    }
                    active_camera_gids = same_cam_occupied.union(other_active_gids)

                    assigned_gid, assoc_status, _, _ = self.association_manager.associate_track(
                        camera_id=camera_id,
                        zone=cam_zone,
                        local_track_id=track_id,
                        bank_embeddings=bank_embeddings,
                        current_time=current_time,
                        frame_idx=frame_idx,
                        active_camera_gids=active_camera_gids
                    )

                    if assoc_status == "MATCHED" and assigned_gid:
                        # Reacquisition / cleanup: remove previous tracks in this camera mapped to assigned_gid
                        for other_tid, existing_gid in list(state["global_assignments"].items()):
                            if existing_gid == assigned_gid and other_tid != track_id:
                                state["global_assignments"].pop(other_tid, None)
                                state["match_status"].pop(other_tid, None)

                        # Clean up from other cameras if reassigned
                        for other_cid, other_st in self.state.items():
                            if other_cid != camera_id:
                                for other_tid, existing_gid in list(other_st.get("global_assignments", {}).items()):
                                    if existing_gid == assigned_gid:
                                        other_st["global_assignments"].pop(other_tid, None)
                                        other_st.get("match_status", {}).pop(other_tid, None)

                        state["global_assignments"][track_id] = assigned_gid
                        state["global_ids_set"].add(assigned_gid)
                        state["match_status"][track_id] = "MATCHED"

                        update_min_q = float(self.config.get("gallery_update_min_quality", 0.60))
                        if added and quality >= update_min_q and self.gallery is not None:
                            self.gallery.update_identity(
                                assigned_gid,
                                [{"camera_id": camera_id, "track_id": track_id}],
                                bank_embeddings
                            )
                    elif assoc_status == "PENDING":
                        if state["match_status"].get(track_id) != "MATCHED":
                            state["match_status"][track_id] = "PENDING"

        # 11. Visual Annotations
        if self.phase1_mode:
            self._draw_phase1_annotations(camera_id, frame, h, w, tracks, state, cam_zone)
        else:
            for t in tracks:
                x1, y1, x2, y2, track_id, conf, cls = map(int, t[:5]) + list(t[5:])
                gid = state["global_assignments"].get(int(track_id), "UNKNOWN")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Trk: {track_id} | GID: {gid}", (x1, max(25, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

        return frame

    def _draw_phase1_annotations(self, camera_id: str, frame: np.ndarray, h: int, w: int, tracks: list, state: dict, cam_zone: str):
        """Delegates overlay rendering to MultiCameraAnnotator."""
        return MultiCameraAnnotator.draw_phase1_annotations(
            camera_id=camera_id,
            frame=frame,
            h=h,
            w=w,
            tracks=tracks,
            state=state,
            cam_zone=cam_zone,
            roi_detectors=self.roi_detectors,
            transition_detectors=self.transition_detectors,
            journey_manager=self.journey_manager,
            association_manager=self.association_manager,
            config=self.config
        )

    # =========================================================================
    # Synchronous Execution Engine
    # =========================================================================

    def run(self, max_frames: Optional[int] = None):
        """Synchronously processes all camera streams in synchronized lockstep."""
        active_cameras = [cam_id for cam_id, st in self.state.items() if st["cap"].isOpened()]
        if not active_cameras:
            logger.error("No valid active camera captures available to process.")
            return

        base_cap = self.state[active_cameras[0]]["cap"]
        w = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(base_cap, "get") else 1280
        w = w if w > 0 else 1280
        h = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(base_cap, "get") else 720
        h = h if h > 0 else 720
        fps = int(base_cap.get(cv2.CAP_PROP_FPS)) if hasattr(base_cap, "get") else 30
        out_fps = fps if fps > 0 else 30

        out_w = w * len(self.cameras_info)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, out_fps, (out_w, h))
        if not out.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(self.output_path, fourcc, out_fps, (out_w, h))

        if max_frames is None:
            max_frames = self.config.get("max_frames")

        # Setup resizable GUI display window that fits desktop screen bounds
        window_name = "Multi-Camera Tracker"
        gui_enabled = True
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            screen_w, screen_h = 1920, 1080
            try:
                import ctypes
                user32 = ctypes.windll.user32
                screen_w = int(user32.GetSystemMetrics(0)) or 1920
                screen_h = int(user32.GetSystemMetrics(1)) or 1080
            except Exception:
                pass

            # Fit initial window within 90% of screen width and 85% of screen height
            avail_w = max(640, int(screen_w * 0.90))
            avail_h = max(360, int(screen_h * 0.85))

            aspect = float(out_w) / float(max(1, h))
            init_w = avail_w
            init_h = int(init_w / aspect)
            if init_h > avail_h:
                init_h = avail_h
                init_w = int(init_h * aspect)

            cv2.resizeWindow(window_name, max(320, init_w), max(180, init_h))
        except Exception as e:
            logger.warning(f"Could not initialize GUI display window: {e}")
            gui_enabled = False

        frame_idx = 0
        try:
            while self.running:
                if max_frames and frame_idx >= max_frames:
                    logger.info(f"Reached max_frames limit: {max_frames}. Ending processing.")
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
                            try:
                                # Stage 1: Validate original resolution before cropping
                                self.validate_camera_frame(cam_id, frame, stage="original")

                                # Stage 2: Camera crop and zoom
                                if state.get("crop"):
                                    cx1, cy1, cx2, cy2 = state["crop"]
                                    frame = frame[cy1:cy2, cx1:cx2]
                                    frame = cv2.resize(frame, (w, h))

                                # Stage 3: Validate post-crop evaluation resolution
                                self.validate_camera_frame(cam_id, frame, stage="evaluation")

                                # Process frame
                                frame = self._process_frame(cam_id, frame, frame_idx, w, h)
                                state["last_frame"] = frame.copy()

                                # Zone overlay banner
                                c_zone_name = state.get("zone", cam_id).upper()
                                cv2.putText(
                                    state["last_frame"], f"Camera: {cam_id} | Zone: {c_zone_name}", (20, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2
                                )
                            except FrameResolutionMismatchError as e:
                                logger.error(f"[{cam_id}] Resolution validation failed: {e}. Stopping entire tracking program.")
                                self.running = False
                                for s in self.state.values():
                                    s["finished"] = True
                                state["resolution_error"] = str(e)
                                if self.raise_on_resolution_error:
                                    raise
                                break

                    if state["last_frame"] is not None:
                        display_frames.append(state["last_frame"])
                    else:
                        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                        cv2.putText(placeholder, f"CAMERA OFFLINE: {cam_id}", (50, h // 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        display_frames.append(placeholder)

                if all_finished or not self.running:
                    break

                frame_idx += 1

                # Write stitched horizontal output
                for i in range(len(display_frames)):
                    if display_frames[i].shape[:2] != (h, w):
                        display_frames[i] = cv2.resize(display_frames[i], (w, h))

                combined_frame = np.hstack(display_frames)
                out.write(combined_frame)

                if gui_enabled:
                    try:
                        # Check if user closed the window
                        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                            logger.info("Display window closed by user. Stopping processing.")
                            self.running = False
                            break

                        # Query current window dimensions to fit frame accurately with aspect ratio preservation
                        rect = cv2.getWindowImageRect(window_name)
                        if rect and len(rect) >= 4 and rect[2] > 10 and rect[3] > 10:
                            win_w, win_h = rect[2], rect[3]
                            cf_h, cf_w = combined_frame.shape[:2]
                            scale = min(win_w / float(cf_w), win_h / float(cf_h))
                            disp_w = max(1, int(cf_w * scale))
                            disp_h = max(1, int(cf_h * scale))

                            resized_disp = cv2.resize(combined_frame, (disp_w, disp_h), interpolation=cv2.INTER_AREA)

                            canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                            y_off = (win_h - disp_h) // 2
                            x_off = (win_w - disp_w) // 2
                            canvas[y_off:y_off + disp_h, x_off:x_off + disp_w] = resized_disp
                            cv2.imshow(window_name, canvas)
                        else:
                            cv2.imshow(window_name, combined_frame)

                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord('q'), 27):
                            self.running = False
                            break
                    except Exception:
                        pass

        finally:
            self._cleanup(out)

    # =========================================================================
    # Asynchronous Execution Engine (Production Async Support)
    # =========================================================================

    async def run_async(self, max_frames: Optional[int] = None):
        """
        Asynchronously processes all camera streams.
        Cooperatively yields execution with asyncio.sleep(0) between frames,
        allowing seamless integration with FastAPI, asyncio websockets, or event loops.
        """
        active_cameras = [cam_id for cam_id, st in self.state.items() if st["cap"].isOpened()]
        if not active_cameras:
            logger.error("No valid active camera captures available to process.")
            return

        base_cap = self.state[active_cameras[0]]["cap"]
        w = int(base_cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if hasattr(base_cap, "get") else 1280
        w = w if w > 0 else 1280
        h = int(base_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if hasattr(base_cap, "get") else 720
        h = h if h > 0 else 720
        fps = int(base_cap.get(cv2.CAP_PROP_FPS)) if hasattr(base_cap, "get") else 30
        out_fps = fps if fps > 0 else 30

        out_w = w * len(self.cameras_info)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, out_fps, (out_w, h))

        if max_frames is None:
            max_frames = self.config.get("max_frames")

        frame_idx = 0
        try:
            while self.running:
                if max_frames and frame_idx >= max_frames:
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
                            self.validate_camera_frame(cam_id, frame, stage="original")

                            if state.get("crop"):
                                cx1, cy1, cx2, cy2 = state["crop"]
                                frame = frame[cy1:cy2, cx1:cx2]
                                frame = cv2.resize(frame, (w, h))

                            self.validate_camera_frame(cam_id, frame, stage="evaluation")
                            frame = self._process_frame(cam_id, frame, frame_idx, w, h)
                            state["last_frame"] = frame.copy()

                    if state["last_frame"] is not None:
                        display_frames.append(state["last_frame"])
                    else:
                        placeholder = np.zeros((h, w, 3), dtype=np.uint8)
                        display_frames.append(placeholder)

                if all_finished or not self.running:
                    break

                frame_idx += 1

                for i in range(len(display_frames)):
                    if display_frames[i].shape[:2] != (h, w):
                        display_frames[i] = cv2.resize(display_frames[i], (w, h))

                combined = np.hstack(display_frames)
                out.write(combined)

                # Cooperative async yield
                await asyncio.sleep(0)

        finally:
            self._cleanup(out)

    def _cleanup(self, out: Optional[cv2.VideoWriter]):
        """Releases video captures, writers, and UI windows."""
        for state in self.state.values():
            if state.get("cap") and hasattr(state["cap"], "release"):
                state["cap"].release()
        if out is not None and hasattr(out, "release"):
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("MultiCameraProcessor cleanup complete.")

    def stop(self):
        """Signals the processing loop to gracefully stop."""
        self.running = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self._cleanup(None)
