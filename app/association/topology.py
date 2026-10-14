from typing import Dict, List, Optional, Any

class TopologyGate:
    """
    Topology constraint checker for multi-camera transitions.
    Enforces physically possible camera/zone transitions as a hard constraint.
    """
    def __init__(
        self,
        topology_config: Optional[Dict[str, Any]] = None,
        transition_rules: Optional[List[Dict[str, Any]]] = None
    ):
        self.topology: Dict[str, Any] = topology_config or {
            "pantry": ["waiting_lobby", "passage"],
            "waiting_lobby": ["pantry", "passage"],
            "passage": ["pantry", "waiting_lobby"]
        }
        self.transition_rules = transition_rules or []

    def is_transition_valid(self, source_zone: Optional[str], target_zone: str) -> bool:
        """
        Hard constraint: returns True if movement from source_zone to target_zone is possible.
        Same zone/camera is always valid.
        """
        if not source_zone or source_zone == target_zone:
            return True

        allowed_zones = self.topology.get(source_zone, [])
        # Extract zone list if configured as dict with next_zones
        if isinstance(allowed_zones, dict):
            allowed_zones = allowed_zones.get("next_zones", [])

        return target_zone in allowed_zones

    def filter_candidates(self, candidates: List[Any], current_zone: str) -> List[Any]:
        """
        Filters candidate identities, removing those whose last_zone cannot physically reach current_zone.
        """
        valid = []
        for cand in candidates:
            last_zone = getattr(cand, "current_zone", None) or getattr(cand, "last_camera", None)
            if isinstance(cand, dict):
                last_zone = cand.get("current_zone") or cand.get("last_camera")

            if self.is_transition_valid(last_zone, current_zone):
                valid.append(cand)
        return valid
