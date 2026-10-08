import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class RoiEvent:
    event: str  # "ROI_ENTER" or "ROI_EXIT"
    camera_id: str
    zone: str
    roi_id: str
    local_track_id: int
    timestamp: float
    frame_idx: int
    foot_point: Tuple[float, float] = (0.0, 0.0)
    global_id: Optional[str] = None
    duration_seconds: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RoiDetector:
    """
    Detects person entry and exit within defined camera ROIs (e.g., Pantry sink area).
    Uses the bottom-center foot point of the bounding box.
    Features:
        - Arbitrary polygon ROI checking via OpenCV pointPolygonTest.
        - Temporal debouncing to eliminate edge jitter.
        - Tracks entry timestamp and computes duration upon exit.
    """
    def __init__(
        self,
        camera_id: str,
        zone: str,
        roi_config: Dict[str, Any],
        debounce_frames: int = 3
    ):
        self.camera_id = camera_id
        self.zone = zone
        self.debounce_frames = debounce_frames
        
        # Parsed polygon arrays: roi_id -> np.ndarray of shape (N, 1, 2)
        self.polygons: Dict[str, np.ndarray] = {}
        for roi_id, cfg in roi_config.items():
            if not cfg.get("enabled", True):
                continue
            pts = cfg.get("points", [])
            if len(pts) >= 3:
                self.polygons[roi_id] = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

        # State tracking:
        # (track_id, roi_id) -> bool (currently inside)
        self.is_inside: Dict[Tuple[int, str], bool] = {}
        # (track_id, roi_id) -> count of consecutive frames inside
        self.inside_consecutive: Dict[Tuple[int, str], int] = {}
        # (track_id, roi_id) -> count of consecutive frames outside
        self.outside_consecutive: Dict[Tuple[int, str], int] = {}
        # (track_id, roi_id) -> enter timestamp
        self.enter_timestamps: Dict[Tuple[int, str], float] = {}

    @staticmethod
    def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox[:4]
        return float((x1 + x2) / 2.0), float(y2)

    def is_point_in_roi(self, point: Tuple[float, float], roi_id: str) -> bool:
        poly = self.polygons.get(roi_id)
        if poly is None:
            return False
        # pointPolygonTest returns >= 0 if inside or on edge
        res = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
        return res >= 0

    def check_rois(
        self,
        local_track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        timestamp: float,
        global_id: Optional[str] = None
    ) -> List[RoiEvent]:
        events: List[RoiEvent] = []
        foot_point = self.get_foot_point(bbox)

        for roi_id in self.polygons.keys():
            key = (local_track_id, roi_id)
            currently_in = self.is_point_in_roi(foot_point, roi_id)
            was_in = self.is_inside.get(key, False)

            if currently_in:
                self.outside_consecutive[key] = 0
                self.inside_consecutive[key] = self.inside_consecutive.get(key, 0) + 1

                if not was_in and self.inside_consecutive[key] >= self.debounce_frames:
                    # Confirmed ROI_ENTER
                    self.is_inside[key] = True
                    self.enter_timestamps[key] = timestamp
                    events.append(RoiEvent(
                        event="ROI_ENTER",
                        camera_id=self.camera_id,
                        zone=self.zone,
                        roi_id=roi_id,
                        local_track_id=local_track_id,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        foot_point=foot_point,
                        global_id=global_id
                    ))
            else:
                self.inside_consecutive[key] = 0
                self.outside_consecutive[key] = self.outside_consecutive.get(key, 0) + 1

                if was_in and self.outside_consecutive[key] >= self.debounce_frames:
                    # Confirmed ROI_EXIT
                    self.is_inside[key] = False
                    enter_ts = self.enter_timestamps.pop(key, timestamp)
                    duration = max(0.0, timestamp - enter_ts)
                    events.append(RoiEvent(
                        event="ROI_EXIT",
                        camera_id=self.camera_id,
                        zone=self.zone,
                        roi_id=roi_id,
                        local_track_id=local_track_id,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        foot_point=foot_point,
                        global_id=global_id,
                        duration_seconds=round(duration, 2)
                    ))

        return events

    def on_track_lost(
        self,
        local_track_id: int,
        frame_idx: int,
        timestamp: float,
        global_id: Optional[str] = None
    ) -> List[RoiEvent]:
        """
        If a track disappears while marked as inside an ROI, emit ROI_EXIT and clean up.
        """
        events: List[RoiEvent] = []
        for roi_id in list(self.polygons.keys()):
            key = (local_track_id, roi_id)
            if self.is_inside.get(key, False):
                self.is_inside[key] = False
                enter_ts = self.enter_timestamps.pop(key, timestamp)
                duration = max(0.0, timestamp - enter_ts)
                events.append(RoiEvent(
                    event="ROI_EXIT",
                    camera_id=self.camera_id,
                    zone=self.zone,
                    roi_id=roi_id,
                    local_track_id=local_track_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    foot_point=(0.0, 0.0),
                    global_id=global_id,
                    duration_seconds=round(duration, 2)
                ))
            self.inside_consecutive.pop(key, None)
            self.outside_consecutive.pop(key, None)
        return events

    def reset_track(self, local_track_id: int):
        for roi_id in self.polygons.keys():
            key = (local_track_id, roi_id)
            self.is_inside.pop(key, None)
            self.inside_consecutive.pop(key, None)
            self.outside_consecutive.pop(key, None)
            self.enter_timestamps.pop(key, None)
