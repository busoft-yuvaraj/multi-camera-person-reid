from typing import Dict, List, Tuple, Optional, Any

class SpatialGate:
    """
    Spatial & transition evidence evaluator.
    Principle:
      Transition/spatial information is strong evidence, but NOT an absolute requirement,
      because a transition may be missed due to occlusion, detection failure, or camera edge crop.
    """
    def __init__(self, transition_rules: Optional[List[Dict[str, Any]]] = None):
        self.rules = transition_rules or []

    def evaluate(
        self,
        source_camera: Optional[str],
        current_camera: str,
        last_transition_id: Optional[str],
        last_transition_direction: Optional[str],
        time_since_transition: float,
        target_camera: Optional[str] = None,
        source_last_seen: Optional[float] = None,
        trans_timestamp: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Evaluates spatial evidence for candidate.
        Returns:
            (is_valid: bool, spatial_score: float, reason: str)
        """
        # Same camera continuation
        if source_camera == current_camera:
            return True, 1.0, "SAME_CAMERA_SPATIAL"

        # No transition was recorded for this candidate (e.g. line was missed)
        if not last_transition_id:
            # Valid as evidence allows, neutral score
            return True, 0.65, "NO_TRANSITION_EVENT (Valid, line may be missed/occluded)"

        # Check for internal movement in source camera after transition:
        # If person continued being observed in source_camera substantially after crossing the line,
        # the crossing was internal room movement, not an exit handoff.
        if trans_timestamp is not None and source_last_seen is not None:
            if (source_last_seen - trans_timestamp) > 2.0:
                return True, 0.65, f"TRANSITION_INTERNAL ({last_transition_id} followed by active presence in {source_camera})"

        # Stale transition check (> 45s)
        if time_since_transition > 45.0:
            return True, 0.60, f"TRANSITION_STALE ({last_transition_id} elapsed {time_since_transition:.1f}s > 45s)"

        # Check configured transition rules and target camera
        matched_rule = False
        reverse_direction = False
        expected_zones = []

        # If explicit target_camera metadata exists on candidate and matches
        if target_camera and target_camera == current_camera:
            matched_rule = True

        for rule in self.rules:
            from_cam = rule.get("from_camera")
            trans_name = rule.get("transition")
            cand_zones = rule.get("candidate_zones", [])
            expected_dir = rule.get("direction")

            if from_cam == source_camera and trans_name == last_transition_id:
                expected_zones.extend(cand_zones)
                if current_camera in cand_zones:
                    if expected_dir and last_transition_direction and expected_dir != last_transition_direction:
                        reverse_direction = True
                    else:
                        matched_rule = True

        if matched_rule:
            return True, 1.0, f"TRANSITION_MATCH ({last_transition_id} -> {current_camera})"

        if reverse_direction:
            # Reverse-direction crossing (e.g. stepping inward):
            # Not an exit handoff, but does not hard-reject candidate
            return True, 0.60, f"TRANSITION_REVERSE_DIRECTION ({last_transition_id} {last_transition_direction})"

        # Transition event was towards another camera (or not targeting this camera).
        # Soft penalty: does NOT hard-reject an identity whose topology allows movement.
        return True, 0.50, f"TRANSITION_UNMATCHED ({last_transition_id} expected {expected_zones or target_camera}, observed in {current_camera})"
