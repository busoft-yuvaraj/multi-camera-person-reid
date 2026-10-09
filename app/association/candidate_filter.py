from typing import Dict, List, Tuple, Optional, Set, Any
from .topology import TopologyGate
from .temporal_gate import TemporalGate
from .spatial_gate import SpatialGate
from app.journey.models import GlobalIdentityState

class CandidateFilter:
    """
    Candidate filtering engine for Phase 1.
    Applies the multi-stage filter pipeline:
      All Global IDs
            ↓
      [Topology Gate]  (Hard constraint: physical camera connection)
            ↓
      [Temporal Gate]  (Time bounds: transit min/max seconds)
            ↓
      [Spatial Gate]   (Evidence: line transition support/penalty)
            ↓
      Valid Candidate Global IDs
    """
    def __init__(
        self,
        topology_gate: TopologyGate,
        temporal_gate: TemporalGate,
        spatial_gate: SpatialGate
    ):
        self.topology_gate = topology_gate
        self.temporal_gate = temporal_gate
        self.spatial_gate = spatial_gate

    def filter_candidates(
        self,
        identities: Dict[str, Any],
        current_camera: str,
        current_zone: str,
        current_time: float,
        exclude_gids: Optional[Set[str]] = None
    ) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, str]]:
        """
        Returns:
            valid_gids: List of GIDs that passed all gates
            evaluations: GID -> dict of scores and details
            rejected: GID -> rejection reason
        """
        exclude_gids = exclude_gids or set()
        valid_gids: List[str] = []
        evaluations: Dict[str, Dict[str, Any]] = {}
        rejected: Dict[str, str] = {}

        for gid, identity in identities.items():
            if gid in exclude_gids:
                rejected[gid] = "EXCLUDED_OCCUPIED_IN_CAMERA"
                continue

            # Extract identity metadata
            if isinstance(identity, GlobalIdentityState):
                last_cam = identity.current_camera
                last_zone = identity.current_zone or last_cam
                last_seen = identity.last_seen_timestamp
                last_trans = identity.last_transition_id
                last_dir = identity.last_transition_direction
                last_trans_time = identity.last_transition_timestamp or last_seen
                status = identity.status
            elif isinstance(identity, dict):
                last_cam = identity.get("current_camera") or identity.get("last_camera")
                last_zone = identity.get("current_zone") or last_cam
                last_seen = float(identity.get("last_seen_timestamp", 0.0))
                last_trans = identity.get("last_transition_id")
                last_dir = identity.get("last_transition_direction")
                last_trans_time = float(identity.get("last_transition_timestamp", last_seen))
                status = identity.get("status", "ACTIVE")
            else:
                last_cam = getattr(identity, "last_camera", None)
                last_zone = last_cam
                last_seen = getattr(identity, "last_seen", 0.0)
                last_trans = None
                last_dir = None
                last_trans_time = last_seen
                status = "ACTIVE"

            # Reacquisition timing constraints are evaluated downstream by TemporalGate

            # -------------------------------------------------------------
            # 1. TOPOLOGY GATE (Hard Constraint)
            # -------------------------------------------------------------
            if last_cam and not self.topology_gate.is_transition_valid(last_zone, current_zone):
                reason = f"TOPOLOGY_REJECT ({last_zone} -> {current_zone} impossible)"
                rejected[gid] = reason
                continue

            # -------------------------------------------------------------
            # 2. TEMPORAL GATE (Transit & Occlusion Bounds)
            # -------------------------------------------------------------
            elapsed = max(0.0, current_time - last_seen) if last_seen > 0.0 else -1.0
            is_temp_valid, temp_score, temp_reason = self.temporal_gate.evaluate(
                source_camera=last_cam,
                target_camera=current_camera,
                elapsed_seconds=elapsed
            )

            if not is_temp_valid:
                rejected[gid] = f"TEMPORAL_REJECT ({temp_reason})"
                continue

            # -------------------------------------------------------------
            # 3. SPATIAL GATE (Evidence Weighting)
            # -------------------------------------------------------------
            time_since_trans = max(0.0, current_time - last_trans_time) if last_trans_time > 0.0 else 999.0
            is_spatial_valid, spatial_score, spatial_reason = self.spatial_gate.evaluate(
                source_camera=last_cam,
                current_camera=current_camera,
                last_transition_id=last_trans,
                last_transition_direction=last_dir,
                time_since_transition=time_since_trans
            )

            if not is_spatial_valid:
                rejected[gid] = f"SPATIAL_REJECT ({spatial_reason})"
                continue

            # Candidate PASSED!
            valid_gids.append(gid)
            evaluations[gid] = {
                "gid": gid,
                "last_camera": last_cam,
                "elapsed_seconds": round(elapsed, 2) if elapsed >= 0 else None,
                "temporal_score": temp_score,
                "temporal_reason": temp_reason,
                "spatial_score": spatial_score,
                "spatial_reason": spatial_reason,
                "last_transition_id": last_trans
            }

        return valid_gids, evaluations, rejected
