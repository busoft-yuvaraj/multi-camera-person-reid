import pytest
import numpy as np
from typing import Dict, Any

from app.association.spatial_gate import SpatialGate
from app.association.temporal_gate import TemporalGate
from app.association.topology import TopologyGate
from app.association.candidate_filter import CandidateFilter
from app.journey.models import GlobalIdentityState, IdentityStatus
from app.journey.journey_manager import JourneyManager
from app.journey.journey_store import JourneyStore
from app.reid.global_gallery import GlobalGallery
from app.reid.matcher import GlobalMatcher
from app.reid.extractors.pose_extractor import PoseExtractor
from app.reid.extractors.par_extractor import ParExtractor


def create_unit_vector(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def create_similar_vector(base: np.ndarray, target_sim: float, seed: int = 101) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base)).astype(np.float32)
    noise = noise - np.dot(noise, base) * base
    noise = noise / np.linalg.norm(noise)
    theta = np.arccos(np.clip(target_sim, -1.0, 1.0))
    res = np.cos(theta) * base + np.sin(theta) * noise
    return (res / np.linalg.norm(res)).astype(np.float32)


# ---------------------------------------------------------------------------
# Test 1: Stale pantry-to-passage event does not reject pantry-to-waiting-lobby
# ---------------------------------------------------------------------------
def test_stale_transition_does_not_reject_valid_pantry_to_waiting_lobby():
    rules = [
        {"from_camera": "pantry", "transition": "pantry_yellow", "candidate_zones": ["waiting_lobby"]},
        {"from_camera": "pantry", "transition": "pantry_green", "candidate_zones": ["passage"]},
    ]
    gate = SpatialGate(transition_rules=rules)

    # Person was in pantry, briefly stepped near green line (passage) at t=10.0,
    # but remained in pantry until t=15.0, then is evaluated entering waiting_lobby at t=16.0.
    is_valid, score, reason = gate.evaluate(
        source_camera="pantry",
        current_camera="waiting_lobby",
        last_transition_id="pantry_green",
        last_transition_direction="EXIT",
        time_since_transition=6.0,
        source_last_seen=15.0,
        trans_timestamp=10.0
    )

    # MUST NOT be hard-rejected (is_valid must be True)
    assert is_valid is True
    assert score == pytest.approx(0.65, abs=0.1)
    assert "TRANSITION_INTERNAL" in reason or "NO_REJECT" in reason


# ---------------------------------------------------------------------------
# Test 2: Incorrect transition direction cannot authorize false identity match
# ---------------------------------------------------------------------------
def test_incorrect_transition_direction_does_not_authorize_match():
    rules = [
        {"from_camera": "pantry", "transition": "pantry_yellow", "candidate_zones": ["waiting_lobby"], "direction": "EXIT"},
    ]
    gate = SpatialGate(transition_rules=rules)

    # Transition pantry_yellow towards waiting_lobby requires EXIT direction.
    # An ENTER event indicates movement into pantry, not exit to lobby.
    is_valid, score, reason = gate.evaluate(
        source_camera="pantry",
        current_camera="waiting_lobby",
        last_transition_id="pantry_yellow",
        last_transition_direction="ENTER",
        time_since_transition=2.0,
        source_last_seen=10.0,
        trans_timestamp=10.0
    )

    # Incorrect direction must not give full 1.0 authorization
    assert score < 0.90


# ---------------------------------------------------------------------------
# Test 3: Lockstep frame processing (elapsed ~0s) accepted with validated transition,
#         rejected without validated transition.
# ---------------------------------------------------------------------------
def test_temporal_gate_lockstep_and_validated_transition():
    config = {
        "pantry_to_waiting_lobby": {"min_seconds": 0.5, "max_seconds": 30.0},
        "default_min_seconds": 0.5,
        "default_max_seconds": 60.0
    }
    t_gate = TemporalGate(temporal_config=config)

    # Case A: Validated transition event present (boundary handoff in synchronized feeds)
    is_valid, score, reason = t_gate.evaluate(
        source_camera="pantry",
        target_camera="waiting_lobby",
        elapsed_seconds=0.05,
        has_validated_transition=True
    )
    assert is_valid is True
    assert score >= 0.85
    assert "TRANSITION_HANDOFF" in reason

    # Case B: No validated transition event (spontaneous near-zero elapsed time across cameras)
    is_valid_no_trans, score_no_trans, reason_no_trans = t_gate.evaluate(
        source_camera="pantry",
        target_camera="waiting_lobby",
        elapsed_seconds=0.05,
        has_validated_transition=False
    )
    assert is_valid_no_trans is False
    assert score_no_trans == 0.0
    assert "TOO_FAST" in reason_no_trans


# ---------------------------------------------------------------------------
# Test 4: Assigning Global ID in Camera B does not delete Camera A's mapping
# ---------------------------------------------------------------------------
def test_assigning_gid_in_camera_b_preserves_camera_a_mapping(tmp_path):
    store = JourneyStore(
        journey_log_path=str(tmp_path / "journey.jsonl"),
        association_log_path=str(tmp_path / "assoc.jsonl")
    )
    mgr = JourneyManager(store=store)

    # 1. Register Person_001 in waiting_lobby (track 1)
    mgr.register_new_identity(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        timestamp=1.0,
        frame_idx=10
    )
    assert mgr.get_active_global_id("waiting_lobby", 1) == "Person_001"

    # 2. Confirm association for Person_001 in pantry (track 2)
    confirmed = mgr.confirm_association(
        global_id="Person_001",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        timestamp=3.0,
        frame_idx=60,
        decision_reason="CROSS_CAMERA_MATCH"
    )
    assert confirmed is True

    # 3. Verify BOTH cameras have active mappings for Person_001
    assert mgr.get_active_global_id("pantry", 2) == "Person_001"
    assert mgr.get_active_global_id("waiting_lobby", 1) == "Person_001"


# ---------------------------------------------------------------------------
# Test 5: Track ID regeneration after occlusion recovers previous Global ID
# ---------------------------------------------------------------------------
def test_track_id_regeneration_after_occlusion_recovers_global_id(tmp_path):
    store = JourneyStore(
        journey_log_path=str(tmp_path / "journey.jsonl"),
        association_log_path=str(tmp_path / "assoc.jsonl")
    )
    mgr = JourneyManager(store=store)

    # 1. Register Person_001 in waiting_lobby (track 1)
    mgr.register_new_identity(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        timestamp=1.0,
        frame_idx=10
    )

    # 2. Track 1 is lost due to occlusion
    mgr.on_track_lost(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        timestamp=2.5,
        frame_idx=50
    )
    state = mgr.get_identity("Person_001")
    assert state.status == IdentityStatus.TEMPORARILY_LOST

    # 3. New local track 5 appears for the same person and confirms association
    confirmed = mgr.confirm_association(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        timestamp=3.0,
        frame_idx=60,
        decision_reason="REACQUISITION"
    )
    assert confirmed is True
    assert mgr.get_active_global_id("waiting_lobby", 5) == "Person_001"
    assert state.status == IdentityStatus.ACTIVE


# ---------------------------------------------------------------------------
# Test 6: Two different people in same camera cannot silently share a Global ID
# ---------------------------------------------------------------------------
def test_two_people_in_same_camera_cannot_share_global_id(tmp_path):
    store = JourneyStore(
        journey_log_path=str(tmp_path / "journey.jsonl"),
        association_log_path=str(tmp_path / "assoc.jsonl")
    )
    mgr = JourneyManager(store=store)

    # Track 1 is actively assigned Person_001
    mgr.register_new_identity(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        timestamp=1.0,
        frame_idx=10
    )

    # Candidate filter excludes Person_001 for Track 2 because Track 1 is concurrently active
    candidate_filter = CandidateFilter(
        topology_gate=TopologyGate({"waiting_lobby": {"next_zones": ["pantry"]}}),
        temporal_gate=TemporalGate({}),
        spatial_gate=SpatialGate([])
    )

    identities = mgr.get_all_identities()
    active_gids_for_track_2 = {"Person_001"}  # Track 1 currently holds Person_001 in this frame

    valid_gids, evals, rejections = candidate_filter.filter_candidates(
        identities=identities,
        current_camera="waiting_lobby",
        current_zone="waiting_lobby",
        current_time=1.5,
        exclude_gids=active_gids_for_track_2
    )

    # Person_001 MUST be excluded
    assert "Person_001" not in valid_gids
    assert "Person_001" in rejections
    assert "EXCLUDED_OCCUPIED_IN_CAMERA" in rejections["Person_001"]


# ---------------------------------------------------------------------------
# Test 7: Cross-view score adjustment does not inflate weak similarity (0.635 -> 0.976)
# ---------------------------------------------------------------------------
def test_cross_view_does_not_inflate_raw_similarity(tmp_path):
    gallery = GlobalGallery(storage_path=str(tmp_path / "gallery.pkl"), clear_on_start=True)
    matcher_cfg = {
        "weight_reid": 0.60,
        "weight_par": 0.25,
        "weight_context": 0.15,
        "threshold_match": 0.70,
        "threshold_ambiguous": 0.60,
        "threshold_cross_view": 0.55,
        "reid_ambiguous_margin": 0.04
    }
    matcher = GlobalMatcher(gallery=gallery, config=matcher_cfg)

    # Create prototype vector and a candidate with raw cosine sim ~0.635
    proto = create_unit_vector(seed=1)
    cand_vec = create_similar_vector(proto, target_sim=0.635, seed=2)

    # Add identity with BACK view
    gid = gallery.add_identity(
        tracks=[{"camera_id": "pantry", "track_id": 1}],
        embeddings=[{"embedding": cand_vec, "viewpoint": "BACK", "quality": 0.85}]
    )
    ident = gallery.get_identity(gid)

    # Candidate prototype is SIDE view (cross-view with BACK)
    score, details = matcher.compute_candidate_score(
        t_prototype=proto,
        t_par_data={},
        dom_t_view="SIDE",
        candidate_identity=ident,
        current_camera="waiting_lobby"
    )

    # In the old code, raw 0.635 was divided by 0.65 -> 0.976
    # In the fixed code, reid_effective must equal raw reid (not inflated)
    assert details["reid_raw"] == pytest.approx(0.635, abs=0.01)
    assert details["reid_effective"] == pytest.approx(0.635, abs=0.01)
    # The final score must not be inflated to >= 0.85
    assert score < 0.75


# ---------------------------------------------------------------------------
# Test 8: Existing topology, temporal, and spatial constraints remain effective
# ---------------------------------------------------------------------------
def test_existing_topology_temporal_spatial_constraints_effective():
    topo = TopologyGate({"waiting_lobby": {"next_zones": ["pantry"]}})
    # pantry -> waiting_lobby is NOT in this restricted topology
    assert topo.is_transition_valid("pantry", "waiting_lobby") is False
    assert topo.is_transition_valid("waiting_lobby", "pantry") is True

    # Temporal gate rejects > max_seconds
    temp = TemporalGate({"waiting_lobby_to_pantry": {"min_seconds": 1.0, "max_seconds": 10.0}})
    is_valid, _, reason = temp.evaluate("waiting_lobby", "pantry", elapsed_seconds=25.0)
    assert is_valid is False
    assert "TOO_SLOW" in reason


# ---------------------------------------------------------------------------
# Test 9: track_only_inside_roi filters out tracks outside specified ROI
# ---------------------------------------------------------------------------
def test_track_only_inside_roi_filtering():
    from app.events.roi_detector import RoiDetector
    roi_config = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[31, 17], [528, 5], [530, 521], [49, 532]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_config=roi_config)

    # Track at sink counter: foot=(210, 365) -> Inside
    inside_track = [150, 100, 270, 365, 1, 0.9, 0]
    fx, fy = (inside_track[0] + inside_track[2]) / 2.0, inside_track[3]
    assert detector.is_point_in_roi((fx, fy), "sink") is True

    # Pink person at doorway: foot=(537, 284) -> Outside
    pink_person_track = [500, 80, 574, 284, 2, 0.9, 0]
    fx_pink, fy_pink = (pink_person_track[0] + pink_person_track[2]) / 2.0, pink_person_track[3]
    assert detector.is_point_in_roi((fx_pink, fy_pink), "sink") is False

    # Sitting person at table: foot=(535, 603) -> Outside
    sitting_track = [490, 200, 580, 603, 3, 0.9, 0]
    fx_sit, fy_sit = (sitting_track[0] + sitting_track[2]) / 2.0, sitting_track[3]
    assert detector.is_point_in_roi((fx_sit, fy_sit), "sink") is False


# ---------------------------------------------------------------------------
# Test 10: CandidateFilter excludes candidate when active in another camera
# ---------------------------------------------------------------------------
def test_cross_camera_simultaneous_assignment_strictly_excluded():
    topo = TopologyGate({"pantry": {"next_zones": ["passage"]}, "passage": {"next_zones": ["pantry"]}})
    temp = TemporalGate({"pantry_to_passage": {"min_seconds": 0.5, "max_seconds": 45.0}})
    spatial = SpatialGate()
    c_filter = CandidateFilter(topo, temp, spatial)

    identities = {
        "Person_001": {
            "current_camera": "pantry",
            "last_seen_timestamp": 10.0,
            "last_transition_id": "pantry_green",
            "last_transition_target_camera": "passage",
            "last_transition_timestamp": 9.8,
            "status": "ACTIVE"
        }
    }

    # If Person_001 is actively detected in pantry, candidate filter with exclude_gids={'Person_001'}
    # MUST reject Person_001
    valid_gids, evals, rejected = c_filter.filter_candidates(
        identities=identities,
        current_camera="passage",
        current_zone="passage",
        current_time=10.0,
        exclude_gids={"Person_001"}
    )
    assert "Person_001" not in valid_gids
    assert rejected["Person_001"] == "EXCLUDED_OCCUPIED_IN_CAMERA"

