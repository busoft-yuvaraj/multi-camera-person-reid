from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any

class IdentityStatus(str, Enum):
    NEW = "NEW"
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    ACTIVE = "ACTIVE"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"
    REACQUIRABLE = "REACQUIRABLE"
    EXPIRED = "EXPIRED"

@dataclass
class JourneyEvent:
    event: str
    global_id: Optional[str]
    camera_id: str
    zone: str
    local_track_id: int
    timestamp: float
    frame_idx: int
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 3),
            "frame_idx": self.frame_idx,
            "global_id": self.global_id,
            "camera_id": self.camera_id,
            "zone": self.zone,
            "local_track_id": self.local_track_id,
            "event": self.event,
            **self.details
        }

@dataclass
class GlobalIdentityState:
    global_id: str
    status: IdentityStatus = IdentityStatus.NEW
    
    # Location state
    current_camera: Optional[str] = None
    current_zone: Optional[str] = None
    current_local_track_id: Optional[int] = None
    
    previous_camera: Optional[str] = None
    previous_zone: Optional[str] = None
    previous_local_track_id: Optional[int] = None
    
    # Timestamps
    first_seen_timestamp: float = 0.0
    last_seen_timestamp: float = 0.0
    
    # Transition metadata
    last_transition_id: Optional[str] = None
    last_transition_timestamp: Optional[float] = None
    last_transition_direction: Optional[str] = None
    last_transition_target_camera: Optional[str] = None
    last_transition_target_zone: Optional[str] = None
    
    # ROI metadata
    current_roi: Optional[str] = None
    last_roi: Optional[str] = None
    last_roi_enter_timestamp: Optional[float] = None
    last_roi_exit_timestamp: Optional[float] = None
    
    # Chronological history
    journey_history: List[JourneyEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def last_camera(self) -> Optional[str]:
        return self.current_camera

    @property
    def last_zone(self) -> Optional[str]:
        return self.current_zone

    def time_since_last_seen(self, current_time: float) -> float:
        if self.last_seen_timestamp <= 0.0:
            return 0.0
        return max(0.0, current_time - self.last_seen_timestamp)

    def add_event(self, event: JourneyEvent):
        self.journey_history.append(event)

    def to_dict(self, current_time: Optional[float] = None) -> Dict[str, Any]:
        t_since = self.time_since_last_seen(current_time) if current_time is not None else 0.0
        return {
            "global_id": self.global_id,
            "status": self.status.value if isinstance(self.status, IdentityStatus) else str(self.status),
            "current_camera": self.current_camera,
            "current_zone": self.current_zone,
            "current_local_track_id": self.current_local_track_id,
            "previous_camera": self.previous_camera,
            "previous_zone": self.previous_zone,
            "first_seen_timestamp": round(self.first_seen_timestamp, 3),
            "last_seen_timestamp": round(self.last_seen_timestamp, 3),
            "time_since_last_seen": round(t_since, 3),
            "last_transition_id": self.last_transition_id,
            "last_transition_timestamp": round(self.last_transition_timestamp, 3) if self.last_transition_timestamp else None,
            "last_transition_direction": self.last_transition_direction,
            "current_roi": self.current_roi,
            "journey_event_count": len(self.journey_history)
        }
