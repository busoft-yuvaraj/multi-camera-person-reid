import math
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

@dataclass
class TransitionEvent:
    event: str
    camera_id: str
    zone: str
    local_track_id: int
    transition_id: str
    timestamp: float
    frame_idx: int
    direction: str
    foot_point: Tuple[float, float]
    global_id: Optional[str] = None
    target_camera: Optional[str] = None
    target_zone: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class TransitionDetector:
    """
    Detects transition line crossings per camera for multi-camera tracking.
    Uses bottom-center foot points:
        foot_x = (x1 + x2) / 2
        foot_y = y2
    Features:
        - Multi-line support per camera.
        - Signed cross-product side determination.
        - Multi-frame confirmation & minimum movement to eliminate boundary jitter.
        - Cooldown / debounce frames per track to prevent duplicate triggering.
    """
    def __init__(
        self,
        camera_id: str,
        zone: str,
        transitions_config: Dict[str, Any],
        cooldown_frames: int = 15,
        min_movement_px: float = 5.0
    ):
        self.camera_id = camera_id
        self.zone = zone
        self.cooldown_frames = cooldown_frames
        self.min_movement_px = min_movement_px
        
        # Parsed line configs: transition_id -> dict
        self.lines: Dict[str, Dict[str, Any]] = {}
        for trans_id, cfg in transitions_config.items():
            if not cfg.get("enabled", True):
                continue
            pts = cfg.get("points", {})
            start = pts.get("start", [0, 0])
            end = pts.get("end", [0, 0])
            self.lines[trans_id] = {
                "start": (float(start[0]), float(start[1])),
                "end": (float(end[0]), float(end[1])),
                "target_camera": cfg.get("target_camera"),
                "target_zone": cfg.get("target_zone")
            }

        # State tracking:
        # (track_id, trans_id) -> last observed side ("SIDE_A", "SIDE_B")
        self.track_sides: Dict[Tuple[int, str], str] = {}
        # (track_id, trans_id) -> last observed foot point
        self.last_points: Dict[Tuple[int, str], Tuple[float, float]] = {}
        # (track_id, trans_id) -> frame_idx when transition last triggered
        self.last_trigger_frame: Dict[Tuple[int, str], int] = {}
        # (track_id, trans_id) -> consecutive frames on current side
        self.side_history_count: Dict[Tuple[int, str], int] = {}

    @staticmethod
    def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        x1, y1, x2, y2 = bbox[:4]
        return float((x1 + x2) / 2.0), float(y2)

    @staticmethod
    def calculate_side(point: Tuple[float, float], line_start: Tuple[float, float], line_end: Tuple[float, float]) -> str:
        """
        Cross-product: d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        """
        x1, y1 = line_start
        x2, y2 = line_end
        px, py = point
        d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if d > 1e-4:
            return "SIDE_A"
        elif d < -1e-4:
            return "SIDE_B"
        return "COLINEAR"

    def check_transitions(
        self,
        local_track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        timestamp: float,
        global_id: Optional[str] = None
    ) -> List[TransitionEvent]:
        """
        Checks all configured transition lines for crossings by the given track.
        Returns a list of TransitionEvents (usually 0 or 1).
        """
        events: List[TransitionEvent] = []
        foot_point = self.get_foot_point(bbox)

        for trans_id, line_info in self.lines.items():
            start = line_info["start"]
            end = line_info["end"]
            key = (local_track_id, trans_id)

            current_side = self.calculate_side(foot_point, start, end)
            if current_side == "COLINEAR":
                continue

            last_side = self.track_sides.get(key)
            last_pt = self.last_points.get(key)
            last_trig = self.last_trigger_frame.get(key, -999)

            # First time observing track on this line
            if last_side is None:
                self.track_sides[key] = current_side
                self.last_points[key] = foot_point
                self.side_history_count[key] = 1
                continue

            # Update side history
            if current_side == last_side:
                self.side_history_count[key] = self.side_history_count.get(key, 0) + 1
                self.last_points[key] = foot_point
                continue

            # Side has flipped: verify conditions
            # 1. Cooldown check
            if frame_idx - last_trig < self.cooldown_frames:
                self.last_points[key] = foot_point
                continue

            # 2. Minimum movement check
            dist = 0.0
            if last_pt is not None:
                dist = math.hypot(foot_point[0] - last_pt[0], foot_point[1] - last_pt[1])
            if dist < self.min_movement_px:
                self.last_points[key] = foot_point
                continue

            # Valid crossing confirmed!
            direction = f"{last_side}_TO_{current_side}"
            self.track_sides[key] = current_side
            self.last_points[key] = foot_point
            self.last_trigger_frame[key] = frame_idx
            self.side_history_count[key] = 1

            event = TransitionEvent(
                event="TRANSITION_CROSSED",
                camera_id=self.camera_id,
                zone=self.zone,
                local_track_id=local_track_id,
                transition_id=trans_id,
                timestamp=timestamp,
                frame_idx=frame_idx,
                direction=direction,
                foot_point=foot_point,
                global_id=global_id,
                target_camera=line_info.get("target_camera"),
                target_zone=line_info.get("target_zone")
            )
            events.append(event)

        return events

    def reset_track(self, local_track_id: int):
        for trans_id in self.lines.keys():
            key = (local_track_id, trans_id)
            self.track_sides.pop(key, None)
            self.last_points.pop(key, None)
            self.last_trigger_frame.pop(key, None)
            self.side_history_count.pop(key, None)
