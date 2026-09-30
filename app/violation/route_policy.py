from typing import Dict, Any, Optional

class RoutePolicy:
    """
    Evaluates pathway compliance rules based on camera/zone configurations and person handwash status.

    Business Rules:
    - HANDWASHED + PASSAGE_1 = VALID
    - HANDWASHED + EXIT = VIOLATION
    - NOT_HANDWASHED + PASSAGE_1 = VIOLATION
    - NOT_HANDWASHED + EXIT = VALID
    - UNKNOWN + Any pathway = Do not immediately classify (UNKNOWN)
    """
    def __init__(self, camera_configs: Dict[str, Any]):
        self.camera_configs = camera_configs or {}

    def get_pathway(self, camera_id: str) -> Optional[str]:
        """
        Resolves the logical pathway for a camera ("PASSAGE_1", "EXIT", "PANTRY", or custom).
        """
        cam_cfg = self.camera_configs.get(camera_id, {})
        if "pathway" in cam_cfg and cam_cfg["pathway"]:
            return str(cam_cfg["pathway"]).upper().strip()

        zone = str(cam_cfg.get("zone", camera_id)).lower().strip()
        cam_id_lower = str(camera_id).lower().strip()

        if "passage" in zone or "passage" in cam_id_lower:
            return "PASSAGE_1"
        elif "exit" in zone or "lobby" in zone or "waiting_lobby" in cam_id_lower:
            return "EXIT"
        elif "pantry" in zone or "pantry" in cam_id_lower:
            return "PANTRY"

        # Check allowed_attire fallback for legacy configurations
        legacy_attire = str(cam_cfg.get("allowed_attire", "")).lower()
        if legacy_attire == "formal":
            return "PASSAGE_1"
        elif legacy_attire == "informal":
            return "EXIT"

        return zone.upper()

    def get_zone(self, camera_id: str) -> str:
        cam_cfg = self.camera_configs.get(camera_id, {})
        return cam_cfg.get("zone", camera_id)

    def is_line_enabled(self, camera_id: str) -> bool:
        cam_cfg = self.camera_configs.get(camera_id, {})
        line_cfg = cam_cfg.get("line", {})
        return bool(line_cfg.get("enabled", False))

    def evaluate_compliance(
        self,
        camera_id: str,
        handwash_status: Optional[str],
        pathway: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates pathway compliance for a person's handwash status against the target pathway.

        Returns dict:
            {
                "is_violation": bool,
                "is_valid": bool,
                "result": "VALID" | "VIOLATION" | "UNKNOWN" | None,
                "pathway": str,
                "handwash_status": str,
                "reason": str
            }
        """
        pw = (pathway or self.get_pathway(camera_id) or "").upper().strip()
        hw = (handwash_status or "UNKNOWN").upper().strip()

        # 1. Pantry or unmonitored zone -> no pathway violation evaluation
        if not pw or pw == "PANTRY":
            return {
                "is_violation": False,
                "is_valid": False,
                "result": None,
                "pathway": pw or "PANTRY",
                "handwash_status": hw,
                "reason": "no_validation_for_zone"
            }

        # 2. Status is UNKNOWN -> wait for evidence, do not classify prematurely
        if hw == "UNKNOWN":
            return {
                "is_violation": False,
                "is_valid": False,
                "result": "UNKNOWN",
                "pathway": pw,
                "handwash_status": hw,
                "reason": "handwash_status_unknown"
            }

        # 3. PASSAGE_1 Pathway
        if pw in ["PASSAGE_1", "PASSAGE1"]:
            if hw == "HANDWASHED":
                return {
                    "is_violation": False,
                    "is_valid": True,
                    "result": "VALID",
                    "pathway": "PASSAGE_1",
                    "handwash_status": hw,
                    "reason": "handwashed_passage_1_valid"
                }
            elif hw == "NOT_HANDWASHED":
                return {
                    "is_violation": True,
                    "is_valid": False,
                    "result": "VIOLATION",
                    "pathway": "PASSAGE_1",
                    "handwash_status": hw,
                    "reason": "not_handwashed_passage_1_violation"
                }

        # 4. EXIT Pathway
        elif pw in ["EXIT", "WAITING_LOBBY", "EXIT_PATHWAY"]:
            if hw == "HANDWASHED":
                return {
                    "is_violation": True,
                    "is_valid": False,
                    "result": "VIOLATION",
                    "pathway": "EXIT",
                    "handwash_status": hw,
                    "reason": "handwashed_exit_violation"
                }
            elif hw == "NOT_HANDWASHED":
                return {
                    "is_violation": False,
                    "is_valid": True,
                    "result": "VALID",
                    "pathway": "EXIT",
                    "handwash_status": hw,
                    "reason": "not_handwashed_exit_valid"
                }

        return {
            "is_violation": False,
            "is_valid": False,
            "result": None,
            "pathway": pw,
            "handwash_status": hw,
            "reason": f"unknown_pathway_rule_{pw}"
        }

    def evaluate_violation(
        self,
        camera_id: str,
        actual_attire: Optional[str] = None,
        attire_conf: float = 0.0,
        min_conf: float = 0.70,
        handwash_status: Optional[str] = None,
        pathway: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backwards-compatible interface. If handwash_status is provided or actual_attire maps
        to handwash states (HANDWASHED / NOT_HANDWASHED), evaluates handwashing compliance.
        Otherwise evaluates legacy attire if required.
        """
        status_to_check = handwash_status
        if not status_to_check and actual_attire:
            attire_norm = actual_attire.upper().strip()
            if attire_norm in ["HANDWASHED", "NOT_HANDWASHED", "UNKNOWN"]:
                status_to_check = attire_norm
            elif attire_norm == "FORMAL":
                status_to_check = "HANDWASHED"
            elif attire_norm == "INFORMAL":
                status_to_check = "NOT_HANDWASHED"

        if status_to_check:
            comp = self.evaluate_compliance(camera_id, status_to_check, pathway)
            expected = "HANDWASHED" if comp["pathway"] == "PASSAGE_1" else "NOT_HANDWASHED"
            return {
                "is_violation": comp["is_violation"],
                "is_valid": comp["is_valid"],
                "expected_attire": expected,
                "actual_attire": comp["handwash_status"],
                "pathway": comp["pathway"],
                "reason": comp["reason"]
            }

        return {
            "is_violation": False,
            "is_valid": False,
            "expected_attire": None,
            "actual_attire": "unknown",
            "pathway": self.get_pathway(camera_id),
            "reason": "no_status_provided"
        }
