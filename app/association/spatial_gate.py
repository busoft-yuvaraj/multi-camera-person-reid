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
        time_since_transition: float
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

        # Check configured transition rules
        # e.g.: from_camera: pantry, transition: pantry_yellow -> target_zones: [waiting_lobby]
        matched_rule = False
        contradictory = False

        for rule in self.rules:
            from_cam = rule.get("from_camera")
            trans_name = rule.get("transition")
            cand_zones = rule.get("candidate_zones", [])

            if from_cam == source_camera and trans_name == last_transition_id:
                if current_camera in cand_zones:
                    matched_rule = True
                else:
                    # Transition was recently towards a completely different camera
                    if time_since_transition < 5.0:
                        contradictory = True

        if matched_rule:
            return True, 1.0, f"TRANSITION_MATCH ({last_transition_id} -> {current_camera})"

        if contradictory:
            # Penalize, but don't hard reject unless time is tiny
            return True, 0.25, f"CONTRADICTORY_TRANSITION ({last_transition_id} expected other camera)"

        return True, 0.60, f"TRANSITION_NEUTRAL ({last_transition_id})"
