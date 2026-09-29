from dataclasses import dataclass, asdict
from typing import Tuple, Optional, Dict, Any
import datetime

@dataclass
class CrossingEvent:
    camera_id: str
    local_track_id: int
    timestamp: str
    previous_side: str
    current_side: str
    crossing_direction: str
    foot_point: Tuple[float, float]
    line_start: Tuple[int, int]
    line_end: Tuple[int, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LineCrossingDetector:
    """
    Detects when a tracked person's foot point crosses an arbitrary 2D virtual line.
    Line coordinates are configured via start (X1, Y1) and end (X2, Y2).
    """
    def __init__(self, camera_id: str, line_start: Tuple[int, int], line_end: Tuple[int, int], enabled: bool = True):
        self.camera_id = camera_id
        self.line_start = (int(line_start[0]), int(line_start[1]))
        self.line_end = (int(line_end[0]), int(line_end[1]))
        self.enabled = enabled
        
        # Mapping: local_track_id -> previous side ("SIDE_A" or "SIDE_B")
        self.track_sides: Dict[int, str] = {}
        # Mapping: local_track_id -> last CrossingEvent
        self.last_crossings: Dict[int, CrossingEvent] = {}

    @staticmethod
    def calculate_side(point: Tuple[float, float], line_start: Tuple[int, int], line_end: Tuple[int, int]) -> str:
        """
        Determines which side of the directed 2D line (line_start -> line_end) the point lies on.
        Uses the signed 2D cross product:
            d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        Returns:
            "SIDE_A" if d > 0 (Left / Positive half-plane)
            "SIDE_B" if d < 0 (Right / Negative half-plane)
            "COLINEAR" if d == 0
        """
        x1, y1 = line_start
        x2, y2 = line_end
        px, py = point

        d = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if d > 0:
            return "SIDE_A"
        elif d < 0:
            return "SIDE_B"
        return "COLINEAR"

    @staticmethod
    def get_foot_point(bbox) -> Tuple[float, float]:
        """
        Computes the bottom-center (foot point) from bounding box:
            foot_x = (x1 + x2) / 2
            foot_y = y2
        """
        x1, y1, x2, y2 = bbox[:4]
        return float((x1 + x2) / 2.0), float(y2)

    def check_crossing(self, local_track_id: int, bbox, timestamp: Optional[str] = None) -> Optional[CrossingEvent]:
        """
        Evaluates the track's bounding box against the virtual line.
        Returns a CrossingEvent if a transition from previous_side != current_side occurred,
        otherwise None.
        """
        if not self.enabled:
            return None

        foot_point = self.get_foot_point(bbox)
        current_side = self.calculate_side(foot_point, self.line_start, self.line_end)

        # Ignore colinear points to prevent flickering on boundary
        if current_side == "COLINEAR":
            return None

        if local_track_id not in self.track_sides:
            # First observation of this track; register starting side without triggering crossing
            self.track_sides[local_track_id] = current_side
            return None

        previous_side = self.track_sides[local_track_id]

        if previous_side != current_side:
            # Side changed: valid crossing detected
            self.track_sides[local_track_id] = current_side
            crossing_dir = f"{previous_side}_TO_{current_side}"
            ts = timestamp or datetime.datetime.now().isoformat()

            event = CrossingEvent(
                camera_id=self.camera_id,
                local_track_id=local_track_id,
                timestamp=ts,
                previous_side=previous_side,
                current_side=current_side,
                crossing_direction=crossing_dir,
                foot_point=foot_point,
                line_start=self.line_start,
                line_end=self.line_end
            )
            self.last_crossings[local_track_id] = event
            return event

        return None

    def get_last_crossing(self, local_track_id: int) -> Optional[CrossingEvent]:
        return self.last_crossings.get(local_track_id)

    def reset_track(self, local_track_id: int):
        self.track_sides.pop(local_track_id, None)
        self.last_crossings.pop(local_track_id, None)
