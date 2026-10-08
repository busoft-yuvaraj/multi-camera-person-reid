from typing import Dict, List, Tuple, Optional, Any

class TemporalGate:
    """
    Temporal constraint checker for candidate identities.
    Validates that the elapsed time since a candidate was last seen falls within
    realistic physical bounds.
    """
    def __init__(
        self,
        temporal_config: Optional[Dict[str, Any]] = None,
        default_min_seconds: float = 0.5,
        default_max_seconds: float = 15.0,
        same_camera_max_seconds: float = 10.0
    ):
        self.config = temporal_config or {}
        self.default_min_seconds = default_min_seconds
        self.default_max_seconds = default_max_seconds
        self.same_camera_max_seconds = float(self.config.get("same_camera_reacquisition_max_seconds", same_camera_max_seconds))

    def evaluate(
        self,
        source_camera: Optional[str],
        target_camera: str,
        elapsed_seconds: float
    ) -> Tuple[bool, float, str]:
        """
        Evaluates elapsed time against constraints.
        Returns:
            (is_valid: bool, temporal_score: float, reason: str)
        """
        # First observation or no timestamp
        if source_camera is None or elapsed_seconds < 0.0:
            return True, 0.50, "NO_PRIOR_TIMESTAMP"

        # Same camera reacquisition / occlusion
        if source_camera == target_camera:
            if elapsed_seconds <= self.same_camera_max_seconds:
                # Score degrades gradually with occlusion duration
                score = max(0.50, 1.0 - (elapsed_seconds / self.same_camera_max_seconds) * 0.50)
                return True, float(score), f"SAME_CAMERA_VALID ({elapsed_seconds:.1f}s <= {self.same_camera_max_seconds}s)"
            else:
                return False, 0.0, f"SAME_CAMERA_EXPIRED ({elapsed_seconds:.1f}s > {self.same_camera_max_seconds}s)"

        # Cross-camera transition
        pair_key = f"{source_camera}_to_{target_camera}"
        reverse_pair_key = f"{target_camera}_to_{source_camera}"
        constraint = self.config.get(pair_key) or self.config.get(reverse_pair_key) or {}

        min_s = float(constraint.get("min_seconds", self.default_min_seconds))
        max_s = float(constraint.get("max_seconds", self.default_max_seconds))

        if elapsed_seconds < min_s:
            return False, 0.0, f"TOO_FAST ({elapsed_seconds:.1f}s < {min_s}s)"

        if elapsed_seconds > max_s:
            return False, 0.0, f"TOO_SLOW ({elapsed_seconds:.1f}s > {max_s}s)"

        # Normal valid transit window: ideal transit gets ~1.0
        # Linear decay towards edges of the window
        midpoint = (min_s + max_s) / 2.0
        spread = max(1.0, (max_s - min_s) / 2.0)
        dist = abs(elapsed_seconds - midpoint)
        score = max(0.50, 1.0 - (dist / spread) * 0.40)

        return True, float(score), f"TEMPORAL_VALID ({min_s}s <= {elapsed_seconds:.1f}s <= {max_s}s)"
