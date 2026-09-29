from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Set, Tuple
import datetime

@dataclass
class ViolationEvent:
    event_id: str
    global_id: Optional[str]
    local_track_id: int
    camera_id: str
    zone: str
    actual_attire: str
    attire_confidence: float
    expected_attire: str
    violation_type: str
    timestamp: str
    crossing_direction: str
    evidence_image: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ViolationEngine:
    """
    Coordinates crossing events, stable attire evaluations, duplicate prevention,
    person state updates, and evidence persistence.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        route_policy,
        evidence_manager,
        person_state_manager
    ):
        self.config = config or {}
        self.route_policy = route_policy
        self.evidence_manager = evidence_manager
        self.person_state_manager = person_state_manager

        self.event_counter = 0
        self.events: List[ViolationEvent] = []

        # Duplicate prevention:
        # Tracks (camera_id, local_track_id, crossing_direction) to ensure a single crossing creates 1 event
        self.recorded_crossings: Set[Tuple[str, int, str]] = set()

        # Active violations per track: (camera_id, local_track_id) -> ViolationEvent
        # Used for real-time visualization on video frames
        self.active_violations: Dict[Tuple[str, int], ViolationEvent] = {}

    def process_crossing(
        self,
        crossing_event,
        global_id: Optional[str],
        stable_attire: Dict[str, Any],
        frame,
        bbox
    ) -> Optional[ViolationEvent]:
        """
        Processes a line crossing event.
        Returns a ViolationEvent if a route violation is confirmed, else None.
        """
        camera_id = crossing_event.camera_id
        track_id = crossing_event.local_track_id
        direction = crossing_event.crossing_direction

        # 1. Duplicate Prevention: ensure this crossing event hasn't already fired
        crossing_sig = (camera_id, track_id, direction)
        if crossing_sig in self.recorded_crossings:
            return None

        # Also link track to global_id in person_state_manager if resolved
        if global_id:
            self.person_state_manager.link_track(camera_id, track_id, global_id)

        attire_name = stable_attire.get("final_attire", "unknown")
        attire_conf = stable_attire.get("confidence", 0.0)
        min_conf = self.config.get("attire", {}).get("minimum_confidence", 0.70)

        # 2. Evaluate violation via RoutePolicy
        eval_res = self.route_policy.evaluate_violation(
            camera_id=camera_id,
            actual_attire=attire_name,
            attire_conf=attire_conf,
            min_conf=min_conf
        )

        zone = self.route_policy.get_zone(camera_id)

        if not eval_res["is_violation"]:
            # Legitimate / allowed movement
            if global_id:
                self.person_state_manager.update_location(global_id, camera_id, zone)
                self.person_state_manager.update_attire(global_id, attire_name, attire_conf)
            return None

        # 3. Violation confirmed -> Generate event
        self.event_counter += 1
        event_id = f"VIO_{self.event_counter:04d}"

        event = ViolationEvent(
            event_id=event_id,
            global_id=global_id,
            local_track_id=track_id,
            camera_id=camera_id,
            zone=zone,
            actual_attire=eval_res["actual_attire"],
            attire_confidence=attire_conf,
            expected_attire=eval_res["expected_attire"],
            violation_type="wrong_route",
            timestamp=crossing_event.timestamp,
            crossing_direction=direction,
            evidence_image=None
        )

        # Mark this specific crossing as handled for duplicate prevention
        self.recorded_crossings.add(crossing_sig)
        self.active_violations[(camera_id, track_id)] = event

        # Update in-memory PersonStateManager
        if global_id:
            self.person_state_manager.update_location(global_id, camera_id, zone)
            self.person_state_manager.update_attire(global_id, attire_name, attire_conf)
            self.person_state_manager.set_violation(global_id, True)

        # 4. Save evidence image and metadata
        if self.evidence_manager:
            img_path = self.evidence_manager.save_evidence(
                event=event,
                frame=frame,
                bbox=bbox,
                line_coords=(crossing_event.line_start, crossing_event.line_end)
            )
            event.evidence_image = img_path

        self.events.append(event)
        return event

    def get_violation(self, camera_id: str, track_id: int) -> Optional[ViolationEvent]:
        """
        Returns active violation for this track if present.
        """
        return self.active_violations.get((camera_id, track_id))

    def reset_track(self, camera_id: str, track_id: int):
        """
        Removes track state when track is lost or completed.
        """
        self.active_violations.pop((camera_id, track_id), None)
        # Clear recorded crossings for this track
        self.recorded_crossings = {
            sig for sig in self.recorded_crossings if not (sig[0] == camera_id and sig[1] == track_id)
        }
