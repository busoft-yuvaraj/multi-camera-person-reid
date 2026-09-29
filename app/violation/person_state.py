from typing import Dict, Optional, List, Any
import datetime

class PersonStateManager:
    """
    Maintains active in-memory state for individuals keyed by Global_ID.
    Tracks entry zone, current location, movement history, attire, and violation status.
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
                "attire": "unknown",
                "attire_confidence": 0.0,
                "entry_camera": None,
                "entry_zone": None,
                "entry_status": "allowed",
                "current_camera": None,
                "current_zone": None,
                "allowed_route": self._compute_allowed_route("unknown"),
                "route_history": [],
                "violation_status": False
            }
        return self.person_state[global_id]

    def _compute_allowed_route(self, attire: str) -> List[str]:
        """
        Determines the list of allowed zones for this person based on attire.
        If attire is null or unknown, returns all zones with no restriction or matching.
        """
        allowed_zones = []
        attire_lower = (attire or "").lower()
        for cam_name, cam_cfg in self.camera_configs.items():
            expected = cam_cfg.get("allowed_attire")
            zone = cam_cfg.get("zone", cam_name)
            if expected is None:
                # No restriction in this zone (e.g. pantry)
                allowed_zones.append(zone)
            elif expected.lower() == attire_lower:
                allowed_zones.append(zone)
        return allowed_zones

    def update_location(self, global_id: str, camera_id: str, zone: str):
        """
        Updates current location and maintains chronological route history.
        """
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
        Updates person's stable attire and recalculates permitted route.
        """
        state = self.get_or_create(global_id)
        state["attire"] = attire
        state["attire_confidence"] = float(confidence)
        state["allowed_route"] = self._compute_allowed_route(attire)

    def set_violation(self, global_id: str, status: bool = True):
        """
        Sets violation flag and entry status.
        """
        state = self.get_or_create(global_id)
        state["violation_status"] = status
        if status:
            state["entry_status"] = "violation"

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

    def get_state(self, global_id: str) -> Optional[Dict[str, Any]]:
        return self.person_state.get(global_id)
