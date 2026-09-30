from typing import Dict, Optional, List, Any
import datetime

class PersonStateManager:
    """
    Maintains active in-memory state for individuals keyed by Global_ID.
    Tracks entry zone, current location, movement history, handwashing status,
    pathway compliance, and violation status.
    """
    def __init__(self, camera_configs: Optional[Dict[str, Any]] = None):
        self.camera_configs = camera_configs or {}
        # Mapping: global_id -> person_state dict
        self.person_state: Dict[str, Dict[str, Any]] = {}
        # Mapping: (camera_id, local_track_id) -> global_id
        self.track_to_global: Dict[tuple, str] = {}

    def get_or_create(self, global_id: str) -> Dict[str, Any]:
        """
        Retrieves or initializes the state dictionary for a Global_ID.
        """
        if global_id not in self.person_state:
            self.person_state[global_id] = {
                "global_id": global_id,
                "handwash_status": "UNKNOWN",  # UNKNOWN, HANDWASHED, NOT_HANDWASHED
                "handwash_confidence": 0.0,
                "handwash_timestamp": None,
                "pathway": None,
                "line_crossed": False,
                "validation_status": None,     # "VALID", "VIOLATION", None
                "entry_camera": None,
                "entry_zone": None,
                "entry_status": "allowed",
                "current_camera": None,
                "current_zone": None,
                "allowed_route": self._compute_allowed_route("unknown"),
                "route_history": [],
                "violation_status": False,
                # Backwards-compatibility fields
                "attire": "unknown",
                "attire_confidence": 0.0,
            }
        return self.person_state[global_id]

    def update_handwash_status(
        self,
        global_id: str,
        status: str,
        confidence: float = 1.0,
        timestamp: Optional[str] = None
    ):
        """
        Updates the handwash status for a Global_ID.
        Protects against ID-switching and state regression:
        Once confirmed as HANDWASHED, a person cannot be downgraded to
        NOT_HANDWASHED or UNKNOWN by a temporary glitch or camera transition.
        """
        if not global_id or str(global_id).upper() in ["PENDING", "AMBIGUOUS", "UNKNOWN", "NONE"]:
            return

        state = self.get_or_create(global_id)
        current_status = state.get("handwash_status", "UNKNOWN")
        status_norm = str(status).upper().strip()

        # Protection: do not downgrade HANDWASHED
        if current_status == "HANDWASHED" and status_norm != "HANDWASHED":
            return

        state["handwash_status"] = status_norm
        state["handwash_confidence"] = float(confidence)
        if timestamp:
            state["handwash_timestamp"] = timestamp
        elif state["handwash_timestamp"] is None and status_norm != "UNKNOWN":
            state["handwash_timestamp"] = datetime.datetime.now().isoformat()

    def get_handwash_status(self, global_id: str) -> str:
        """
        Returns the current handwash status for a Global_ID, defaulting to UNKNOWN.
        """
        if not global_id:
            return "UNKNOWN"
        state = self.person_state.get(global_id)
        if not state:
            return "UNKNOWN"
        return state.get("handwash_status", "UNKNOWN")

    def update_validation(
        self,
        global_id: str,
        validation_status: str,
        pathway: Optional[str] = None
    ):
        """
        Updates pathway validation outcome ("VALID" or "VIOLATION").
        """
        if not global_id:
            return
        state = self.get_or_create(global_id)
        state["validation_status"] = validation_status
        state["line_crossed"] = True
        if pathway:
            state["pathway"] = pathway
        if validation_status == "VIOLATION":
            state["violation_status"] = True
            state["entry_status"] = "violation"
        elif validation_status == "VALID":
            state["violation_status"] = False
            state["entry_status"] = "allowed"

    def _compute_allowed_route(self, attire: str) -> List[str]:
        """
        Maintained for backwards compatibility.
        """
        allowed_zones = []
        attire_lower = (attire or "").lower()
        for cam_name, cam_cfg in self.camera_configs.items():
            expected = cam_cfg.get("allowed_attire")
            zone = cam_cfg.get("zone", cam_name)
            if expected is None:
                allowed_zones.append(zone)
            elif expected.lower() == attire_lower:
                allowed_zones.append(zone)
        return allowed_zones

    def update_location(self, global_id: str, camera_id: str, zone: str):
        """
        Updates current location and maintains chronological route history.
        """
        if not global_id:
            return
        state = self.get_or_create(global_id)
        
        # Set entry if not yet recorded
        if state["entry_camera"] is None:
            state["entry_camera"] = camera_id
            state["entry_zone"] = zone

        if state["current_zone"] != zone:
            state["route_history"].append({
                "camera": camera_id,
                "zone": zone,
                "timestamp": datetime.datetime.now().isoformat()
            })

        state["current_camera"] = camera_id
        state["current_zone"] = zone

    def update_attire(self, global_id: str, attire: str, confidence: float):
        """
        Maintained for backwards compatibility.
        """
        if not global_id:
            return
        state = self.get_or_create(global_id)
        state["attire"] = attire
        state["attire_confidence"] = float(confidence)
        state["allowed_route"] = self._compute_allowed_route(attire)

    def set_violation(self, global_id: str, status: bool = True):
        """
        Sets violation flag and entry status.
        """
        if not global_id:
            return
        state = self.get_or_create(global_id)
        state["violation_status"] = status
        if status:
            state["entry_status"] = "violation"
            state["validation_status"] = "VIOLATION"

    def link_track(self, camera_id: str, local_track_id: int, global_id: str):
        """
        Associates a camera-local track ID with a Global_ID.
        """
        self.track_to_global[(camera_id, int(local_track_id))] = global_id

    def get_global_id(self, camera_id: str, local_track_id: int) -> Optional[str]:
        """
        Retrieves the Global_ID mapped to a local track, if resolved.
        """
        return self.track_to_global.get((camera_id, int(local_track_id)))

    def is_in_violation(self, global_id: str) -> bool:
        """
        Returns True if the specified Global_ID has an active violation recorded.
        """
        if not global_id:
            return False
        state = self.person_state.get(global_id)
        if not state:
            return False
        return bool(state.get("violation_status", False))

    def get_state(self, global_id: str) -> Optional[Dict[str, Any]]:
        return self.person_state.get(global_id)
