import cv2
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

# Camera default expected resolutions based on the coordinate system of existing ROI configurations
DEFAULT_CAMERA_RESOLUTIONS: Dict[str, Tuple[int, int]] = {
    "waiting_lobby": (1280, 720),
    "pantry": (1280, 720),
    "passage": (1280, 720),
    "passage_1": (1280, 720),
}


def parse_resolution(res_cfg: Any) -> Optional[Tuple[int, int]]:
    """
    Parses a resolution configuration into a (width, height) tuple.
    Supports:
        - [width, height] or (width, height)
        - {'width': W, 'height': H} or {'w': W, 'h': H}
        - 'WIDTHxHEIGHT' (e.g. '1280x720')
    """
    if res_cfg is None:
        return None
    if isinstance(res_cfg, (list, tuple)) and len(res_cfg) >= 2:
        return (int(res_cfg[0]), int(res_cfg[1]))
    if isinstance(res_cfg, dict):
        w = res_cfg.get("width", res_cfg.get("w"))
        h = res_cfg.get("height", res_cfg.get("h"))
        if w is not None and h is not None:
            return (int(w), int(h))
    if isinstance(res_cfg, str) and "x" in res_cfg.lower():
        parts = res_cfg.lower().split("x")
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), int(parts[1].strip()))
            except ValueError:
                pass
    return None


class FrameResolutionMismatchError(ValueError):
    """
    Raised when a camera frame's actual resolution does not match the expected
    resolution for which the camera's ROI system was configured.
    """
    def __init__(
        self,
        camera_id: str,
        expected_resolution: Tuple[int, int],
        actual_resolution: Tuple[int, int],
        message: Optional[str] = None
    ):
        self.camera_id = camera_id
        self.expected_resolution = expected_resolution  # (width, height)
        self.actual_resolution = actual_resolution      # (width, height)
        exp_w, exp_h = expected_resolution
        act_w, act_h = actual_resolution
        if not message:
            message = (
                f"Camera '{camera_id}' frame resolution mismatch: "
                f"expected {exp_w}x{exp_h} (width x height), "
                f"but got {act_w}x{act_h}. "
                f"ROI coordinates cannot be safely applied."
            )
        super().__init__(message)


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
        - Strict frame-resolution validation before ROI evaluation.
        - Arbitrary polygon ROI checking via OpenCV pointPolygonTest.
        - Temporal debouncing to eliminate edge jitter.
        - Tracks entry timestamp and computes duration upon exit.
    """
    def __init__(
        self,
        camera_id: str,
        zone: str,
        roi_config: Dict[str, Any],
        debounce_frames: int = 3,
        expected_resolution: Optional[Tuple[int, int]] = None
    ):
        self.camera_id = camera_id
        self.zone = zone
        self.debounce_frames = debounce_frames
        
        # Expected resolution definition
        # Priority: explicit argument -> roi_config["expected_resolution"] -> camera default -> (1280, 720)
        parsed_res = parse_resolution(expected_resolution)
        if parsed_res is None:
            parsed_res = parse_resolution(
                roi_config.get("expected_resolution") or roi_config.get("resolution")
            )
        if parsed_res is None:
            parsed_res = DEFAULT_CAMERA_RESOLUTIONS.get(camera_id, (1280, 720))

        self.expected_width: int = int(parsed_res[0])
        self.expected_height: int = int(parsed_res[1])
        self.expected_resolution: Tuple[int, int] = (self.expected_width, self.expected_height)

        # Parsed polygon arrays: roi_id -> np.ndarray of shape (N, 1, 2)
        self.polygons: Dict[str, np.ndarray] = {}
        for roi_id, cfg in roi_config.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled", True):
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

    def validate_resolution(self, actual_width: int, actual_height: int) -> bool:
        """
        Validates frame dimensions against the expected resolution.
        Raises FrameResolutionMismatchError if dimensions differ.
        """
        if actual_width != self.expected_width or actual_height != self.expected_height:
            raise FrameResolutionMismatchError(
                camera_id=self.camera_id,
                expected_resolution=self.expected_resolution,
                actual_resolution=(actual_width, actual_height)
            )
        return True

    def validate_frame(self, frame: Any) -> bool:
        """
        Safely inspects a frame and validates its resolution against expected resolution.
        Handles None, unreadable, or invalid frames safely without crashing.
        """
        if frame is None:
            raise FrameResolutionMismatchError(
                camera_id=self.camera_id,
                expected_resolution=self.expected_resolution,
                actual_resolution=(0, 0),
                message=f"Camera '{self.camera_id}' received None frame: frame is unreadable or empty."
            )
        if not hasattr(frame, "shape") or len(frame.shape) < 2:
            raise FrameResolutionMismatchError(
                camera_id=self.camera_id,
                expected_resolution=self.expected_resolution,
                actual_resolution=(0, 0),
                message=f"Camera '{self.camera_id}' received invalid frame with shape {getattr(frame, 'shape', None)}."
            )
        actual_h, actual_w = int(frame.shape[0]), int(frame.shape[1])
        if actual_h <= 0 or actual_w <= 0:
            raise FrameResolutionMismatchError(
                camera_id=self.camera_id,
                expected_resolution=self.expected_resolution,
                actual_resolution=(actual_w, actual_h),
                message=f"Camera '{self.camera_id}' received empty frame with zero dimensions ({actual_w}x{actual_h})."
            )
        return self.validate_resolution(actual_width=actual_w, actual_height=actual_h)

    @staticmethod
    def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox[:4]
        return float((x1 + x2) / 2.0), float(y2)

    def is_point_in_roi(
        self,
        point: Tuple[float, float],
        roi_id: str,
        frame_resolution: Optional[Tuple[int, int]] = None
    ) -> bool:
        if frame_resolution is not None:
            self.validate_resolution(actual_width=frame_resolution[0], actual_height=frame_resolution[1])
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
        global_id: Optional[str] = None,
        frame: Optional[Any] = None,
        frame_resolution: Optional[Tuple[int, int]] = None
    ) -> List[RoiEvent]:
        # Validate resolution before applying ROI if frame or frame_resolution is provided
        if frame is not None:
            self.validate_frame(frame)
        elif frame_resolution is not None:
            self.validate_resolution(actual_width=frame_resolution[0], actual_height=frame_resolution[1])
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
