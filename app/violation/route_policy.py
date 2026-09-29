from typing import Dict, Any, Optional

class RoutePolicy:
    """
    Evaluates route compliance rules based on camera/zone configurations and person attire.
    """
    def __init__(self, camera_configs: Dict[str, Any]):
        """
        camera_configs: Dictionary mapping camera IDs to their configuration:
            {
                "waiting_lobby": {"zone": "waiting_lobby", "allowed_attire": "informal", ...},
                "passage_1": {"zone": "passage_1", "allowed_attire": "formal", ...},
                "pantry": {"zone": "pantry", "allowed_attire": None, ...}
            }
        """
        self.camera_configs = camera_configs or {}

    def get_expected_attire(self, camera_id: str) -> Optional[str]:
        cam_cfg = self.camera_configs.get(camera_id, {})
        allowed = cam_cfg.get("allowed_attire")
        if allowed is not None and str(allowed).lower() != "none" and str(allowed).lower() != "null":
            return str(allowed).lower().strip()
        return None

    def get_zone(self, camera_id: str) -> str:
        cam_cfg = self.camera_configs.get(camera_id, {})
        return cam_cfg.get("zone", camera_id)

    def is_line_enabled(self, camera_id: str) -> bool:
        cam_cfg = self.camera_configs.get(camera_id, {})
        line_cfg = cam_cfg.get("line", {})
        return bool(line_cfg.get("enabled", False))

    def evaluate_violation(
        self,
        camera_id: str,
        actual_attire: Optional[str],
        attire_conf: float = 0.0,
        min_conf: float = 0.70
    ) -> Dict[str, Any]:
        """
        Evaluates whether an observed person's attire violates the route rule for this camera/zone.

        Returns dict:
            {
                "is_violation": bool,
                "expected_attire": str or None,
                "actual_attire": str,
                "reason": str
            }
        """
        expected = self.get_expected_attire(camera_id)

        # 1. Zone has no rule (e.g., pantry) -> always allowed
        if expected is None:
            return {
                "is_violation": False,
                "expected_attire": None,
                "actual_attire": actual_attire or "unknown",
                "reason": "no_rule_for_zone"
            }

        # 2. Actual attire unknown
        if not actual_attire or actual_attire.lower() == "unknown":
            return {
                "is_violation": False,
                "expected_attire": expected,
                "actual_attire": "unknown",
                "reason": "attire_unknown"
            }

        # 3. Confidence below minimum threshold -> do not generate false positive
        if attire_conf < min_conf:
            return {
                "is_violation": False,
                "expected_attire": expected,
                "actual_attire": actual_attire,
                "reason": "confidence_below_threshold"
            }

        actual = actual_attire.lower().strip()

        # 4. Compare actual attire with expected attire
        if actual != expected:
            return {
                "is_violation": True,
                "expected_attire": expected,
                "actual_attire": actual,
                "reason": f"attire_mismatch_{actual}_vs_{expected}"
            }

        return {
            "is_violation": False,
            "expected_attire": expected,
            "actual_attire": actual,
            "reason": "attire_allowed"
        }
