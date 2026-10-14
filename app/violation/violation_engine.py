from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Set, Tuple
import datetime
import logging

@dataclass
class ViolationEvent:
    event_id: str
    global_id: Optional[str]
    local_track_id: int
    camera_id: str
    zone: str
    pathway: str
    handwash_status: str
    validation_result: str
    violation_type: str
    timestamp: str
    crossing_direction: str
    reason: str
    evidence_image: Optional[str] = None
    # Backwards-compatibility fields for legacy attire integrations
    actual_attire: str = ""
    expected_attire: str = ""
    attire_confidence: float = 1.0

    def __post_init__(self):
        if not self.actual_attire:
            self.actual_attire = self.handwash_status
        if not self.expected_attire:
            self.expected_attire = "HANDWASHED" if self.pathway == "PASSAGE_1" else "NOT_HANDWASHED"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class ViolationEngine:
    """
    Coordinates pathway crossing events, handwashing validation, duplicate prevention,
    logging of decision chains, person state updates, and evidence persistence.
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

        # Duplicate prevention sets
        self.recorded_crossings: Set[Tuple[str, Any, str]] = set()

        # Active violations per track: (camera_id, local_track_id) -> ViolationEvent
        self.active_violations: Dict[Tuple[str, int], ViolationEvent] = {}

        # Active validations per track: (camera_id, local_track_id) -> dict
        self.active_validations: Dict[Tuple[str, int], Dict[str, Any]] = {}

    def process_crossing(
        self,
        crossing_event,
        global_id: Optional[str] = None,
        handwash_status: Optional[str] = None,
        frame=None,
        bbox=None,
        stable_attire: Optional[Dict[str, Any]] = None
    ) -> Optional[ViolationEvent]:
        """
        Processes a line crossing event against handwashing pathway compliance rules.
        Decision Chain:
            Global ID -> Current pathway -> Stored handwash status -> VALID / VIOLATION
        """
        camera_id = crossing_event.camera_id
        track_id = crossing_event.local_track_id
        direction = crossing_event.crossing_direction

        # 1. Duplicate Prevention: ensure this crossing event hasn't already fired
        track_sig = (camera_id, track_id, direction)
        gid_sig = (camera_id, str(global_id), direction) if global_id else None

        if track_sig in self.recorded_crossings:
            return None
        if gid_sig and gid_sig in self.recorded_crossings:
            return None

        # Link local track to Global ID in person_state_manager
        if global_id and self.person_state_manager:
            self.person_state_manager.link_track(camera_id, track_id, global_id)

        # 2. Retrieve Stored Handwash Status
        if not handwash_status or handwash_status == "UNKNOWN":
            if global_id and self.person_state_manager:
                stored_hw = self.person_state_manager.get_handwash_status(global_id)
                if stored_hw and stored_hw != "UNKNOWN":
                    handwash_status = stored_hw

        # Check stable_attire fallback if provided (e.g. from tests passing attire)
        if (not handwash_status or handwash_status == "UNKNOWN") and stable_attire:
            final_att = stable_attire.get("final_attire", "").upper()
            if final_att in ["HANDWASHED", "NOT_HANDWASHED", "UNKNOWN"]:
                handwash_status = final_att
            elif final_att == "FORMAL":
                handwash_status = "HANDWASHED"
            elif final_att == "INFORMAL":
                handwash_status = "NOT_HANDWASHED"

        handwash_status = (handwash_status or "UNKNOWN").upper().strip()

        # 3. Determine Target Pathway
        pathway = self.route_policy.get_pathway(camera_id) if self.route_policy else camera_id.upper()
        zone = self.route_policy.get_zone(camera_id) if self.route_policy else camera_id

        # 4. Evaluate Pathway Compliance
        eval_res = self.route_policy.evaluate_compliance(
            camera_id=camera_id,
            handwash_status=handwash_status,
            pathway=pathway
        ) if self.route_policy else {
            "is_violation": False,
            "is_valid": False,
            "result": "UNKNOWN",
            "reason": "no_policy"
        }

        result = eval_res.get("result")

        # 5. Mandatory Logging of Decision Chain (Section 15)
        gid_label = str(global_id) if global_id else f"Track_{track_id}"
        path_label = pathway or zone.upper()

        logging.info(f"[HANDWASH] Global ID {gid_label} → {handwash_status}")
        logging.info(f"[PATHWAY] Global ID {gid_label} → {path_label}")
        logging.info(f"[VALIDATION] Global ID {gid_label} → {result or 'UNKNOWN'}")

        print(f"[HANDWASH] Global ID {gid_label} → {handwash_status}")
        print(f"[PATHWAY] Global ID {gid_label} → {path_label}")
        print(f"[VALIDATION] Global ID {gid_label} → {result or 'UNKNOWN'}")

        # 6. UNKNOWN State Handling: do not immediately classify or flag violations
        if result == "UNKNOWN" or result is None:
            return None

        # 7. Outcome: VALID
        if result == "VALID":
            self.recorded_crossings.add(track_sig)
            if gid_sig:
                self.recorded_crossings.add(gid_sig)

            self.active_validations[(camera_id, track_id)] = {
                "global_id": global_id,
                "local_track_id": track_id,
                "pathway": pathway,
                "handwash_status": handwash_status,
                "status": "VALID",
                "timestamp": crossing_event.timestamp
            }

            if global_id and self.person_state_manager:
                self.person_state_manager.update_location(global_id, camera_id, zone)
                self.person_state_manager.update_validation(global_id, "VALID", pathway)

            return None

        # 8. Outcome: VIOLATION
        self.event_counter += 1
        event_id = f"VIO_{self.event_counter:04d}"

        event = ViolationEvent(
            event_id=event_id,
            global_id=global_id,
            local_track_id=track_id,
            camera_id=camera_id,
            zone=zone,
            pathway=pathway,
            handwash_status=handwash_status,
            validation_result="VIOLATION",
            violation_type="handwash_pathway_violation",
            timestamp=crossing_event.timestamp,
            crossing_direction=direction,
            reason=eval_res.get("reason", "pathway_violation"),
            evidence_image=None
        )

        self.recorded_crossings.add(track_sig)
        if gid_sig:
            self.recorded_crossings.add(gid_sig)

        self.active_violations[(camera_id, track_id)] = event

        if global_id and self.person_state_manager:
            self.person_state_manager.update_location(global_id, camera_id, zone)
            self.person_state_manager.update_validation(global_id, "VIOLATION", pathway)
            self.person_state_manager.set_violation(global_id, True)

        # 9. Save Evidence Image and Metadata
        if self.evidence_manager:
            line_coords = (crossing_event.line_start, crossing_event.line_end) if hasattr(crossing_event, "line_start") else None
            img_path = self.evidence_manager.save_evidence(
                event=event,
                frame=frame,
                bbox=bbox,
                line_coords=line_coords
            )
            event.evidence_image = img_path

        self.events.append(event)
        return event

    def get_violation(self, camera_id: str, track_id: int) -> Optional[ViolationEvent]:
        """
        Returns active violation for this track if present.
        """
        return self.active_violations.get((camera_id, track_id))

    def get_validation(self, camera_id: str, track_id: int) -> Optional[Dict[str, Any]]:
        """
        Returns active validation for this track if present.
        """
        return self.active_validations.get((camera_id, track_id))

    def reset_track(self, camera_id: str, track_id: int):
        """
        Removes track state when track is lost or completed.
        """
        self.active_violations.pop((camera_id, track_id), None)
        self.active_validations.pop((camera_id, track_id), None)
        self.recorded_crossings = {
            sig for sig in self.recorded_crossings if not (sig[0] == camera_id and sig[1] == track_id)
        }
