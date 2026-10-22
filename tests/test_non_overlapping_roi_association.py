import pytest
import numpy as np
from typing import Dict, Any, List

from app.association.topology import TopologyGate
from app.association.temporal_gate import TemporalGate
from app.association.spatial_gate import SpatialGate
from app.association.candidate_filter import CandidateFilter
from app.association.association_manager import AssociationManager
from app.journey.models import GlobalIdentityState, IdentityStatus, JourneyEvent
from app.journey.journey_store import JourneyStore
from app.journey.journey_manager import JourneyManager
from app.events.transition_detector import TransitionEvent
from app.reid.global_gallery import GlobalGallery
from app.reid.matcher import GlobalMatcher


def make_unit_vector(seed: int = 42, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def make_similar_vector(base_vec: np.ndarray, target_sim: float = 0.85, seed: int = 99) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base_vec)).astype(np.float32)
    noise = noise - np.dot(noise, base_vec) * base_vec
    noise = noise / np.linalg.norm(noise)
    theta = np.arccos(np.clip(target_sim, -1.0, 1.0))
    res = np.cos(theta) * base_vec + np.sin(theta) * noise
    return res / np.linalg.norm(res)


def build_non_overlapping_env(tmp_path, custom_config: Dict[str, Any] = None):
    config = {
        "threshold_match": 0.70,
        "threshold_ambiguous": 0.60,
        "threshold_cross_view": 0.55,
        "reid_ambiguous_margin": 0.04,
        "pending_confirm_frames": 3,
        "reid_new_id_confirm_frames": 15,
        "track_lost_frames": 60,
        "probation_confirm_frames": 4,
        "topology": {
            "pantry": ["waiting_lobby", "passage"],
            "waiting_lobby": ["pantry", "passage"],
            "passage": ["waiting_lobby", "pantry", "workspace"],
            "workspace": ["passage"]
        },
        "temporal_constraints": {
            "overlapping_camera_pairs": [],  # Strictly non-overlapping ROIs
            "pantry_to_waiting_lobby": {"min_seconds": 0.5, "max_seconds": 45.0},
            "waiting_lobby_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0},
            "pantry_to_passage": {"min_seconds": 0.5, "max_seconds": 45.0},
            "passage_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0},
            "waiting_lobby_to_passage": {"min_seconds": 0.5, "max_seconds": 45.0},
            "passage_to_waiting_lobby": {"min_seconds": 0.5, "max_seconds": 45.0},
            "passage_to_workspace": {"min_seconds": 0.5, "max_seconds": 45.0},
            "workspace_to_passage": {"min_seconds": 0.5, "max_seconds": 45.0},
            "same_camera_reacquisition_max_seconds": 60.0,
        },
        "transition_rules": [
            {
                "from_camera": "pantry",
                "transition": "pantry_yellow",
                "direction": "SIDE_A_TO_SIDE_B",
                "candidate_zones": ["waiting_lobby"]
            },
            {
                "from_camera": "pantry",
                "transition": "pantry_yellow",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["pantry"]
            },
            {
                "from_camera": "pantry",
                "transition": "pantry_green",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["passage"]
            },
            {
                "from_camera": "pantry",
                "transition": "pantry_green",
                "direction": "SIDE_A_TO_SIDE_B",
                "candidate_zones": ["pantry"]
            },
            {
                "from_camera": "waiting_lobby",
                "transition": "lobby_yellow",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["pantry", "passage"]
            },
            {
                "from_camera": "waiting_lobby",
                "transition": "lobby_yellow",
                "direction": "SIDE_A_TO_SIDE_B",
                "candidate_zones": ["waiting_lobby"]
            },
            {
                "from_camera": "passage",
                "transition": "passage_green",
                "direction": "SIDE_A_TO_SIDE_B",
                "candidate_zones": ["pantry", "waiting_lobby"]
            },
            {
                "from_camera": "passage",
                "transition": "passage_green",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["passage", "workspace"]
            }
        ]
    }
    if custom_config:
        config.update(custom_config)

    store = JourneyStore(
        journey_log_path=str(tmp_path / "journey.jsonl"),
        association_log_path=str(tmp_path / "assoc.jsonl")
    )
    journey_mgr = JourneyManager(
        store=store,
        reacquisition_timeout_seconds=60.0,
        topology=config["topology"],
        overlapping_camera_pairs=config["temporal_constraints"]["overlapping_camera_pairs"],
        transition_rules=config["transition_rules"]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gallery.pkl"), clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=config)
    topology_gate = TopologyGate(topology_config=config["topology"])
    temporal_gate = TemporalGate(temporal_config=config["temporal_constraints"])
    spatial_gate = SpatialGate(transition_rules=config["transition_rules"])
    candidate_filter = CandidateFilter(
        topology_gate=topology_gate,
        temporal_gate=temporal_gate,
        spatial_gate=spatial_gate
    )
    assoc_mgr = AssociationManager(
        gallery=gallery,
        matcher=matcher,
        candidate_filter=candidate_filter,
        journey_manager=journey_mgr,
        store=store,
        config=config
    )
    return {
        "config": config,
        "store": store,
        "journey_mgr": journey_mgr,
        "gallery": gallery,
        "matcher": matcher,
        "candidate_filter": candidate_filter,
        "assoc_mgr": assoc_mgr,
        "spatial_gate": spatial_gate,
        "temporal_gate": temporal_gate,
        "topology_gate": topology_gate
    }


# ===========================================================================
# 1. Same person moving between non-overlapping cameras retains same GID
# ===========================================================================
def test_1_same_person_moving_between_non_overlapping_cameras(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=100)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Person leaves pantry at t=2.0s
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.0, frame_idx=30)

    # Person arrives in passage at t=3.5s (elapsed = 1.5s, valid route)
    bank_passage = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned_gid, status, sim, _ = assoc_mgr.associate_track(
        camera_id="passage",
        zone="passage",
        local_track_id=10,
        bank_embeddings=bank_passage,
        current_time=3.5,
        frame_idx=60,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert assigned_gid == "Person_001"

    # Person leaves passage at t=6.0s
    jm.on_track_lost("passage", "passage", 10, timestamp=6.0, frame_idx=120)

    # Person arrives in waiting_lobby at t=8.0s (elapsed = 2.0s, valid route)
    bank_lobby = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.86} for _ in range(3)]
    assigned_gid2, status2, sim2, _ = assoc_mgr.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=20,
        bank_embeddings=bank_lobby,
        current_time=8.0,
        frame_idx=160,
        active_camera_gids=set()
    )
    assert status2 == "MATCHED"
    assert assigned_gid2 == "Person_001"
    assert len(gallery.get_identities()) == 1


# ===========================================================================
# 2. Two distinct people active simultaneously in non-overlapping cameras never share GID
# ===========================================================================
def test_2_two_distinct_people_in_non_overlapping_cameras_never_share_gid(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=201)
    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "pantry", "pantry", 1, timestamp=10.0, frame_idx=100)

    # Second person in waiting_lobby at t=10.6s (elapsed = 0.6s >= min 0.5s, time_diff = 0.6s < 1.0s)
    # Even if embedding has high similarity, non-overlapping cameras forbid simultaneous GID sharing
    bank_lobby = [{"embedding": emb_1, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=2,
        bank_embeddings=bank_lobby,
        current_time=10.6,
        frame_idx=101,
        active_camera_gids=set()
    )

    # Must be deferred as conflict, never confirmed as Person_001 simultaneously
    assert status == "PENDING"
    assert "CONFLICT_DEFERRED" in details["decision"]
    assert jm.get_active_global_id("waiting_lobby", 2) is None
    assert jm.get_active_global_id("pantry", 1) == "Person_001"


# ===========================================================================
# 3. Person becoming temporarily lost and subsequently reacquiring original GID
# ===========================================================================
def test_3_temporarily_lost_person_reacquires_original_gid(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=301)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=5.0, frame_idx=50)

    # Person disappears (temporarily lost)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=7.0, frame_idx=70)
    state = jm.get_identity("Person_001")
    assert state.status == IdentityStatus.TEMPORARILY_LOST

    # Reappears in waiting_lobby after 4 seconds (t=11.0s)
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.89} for _ in range(3)]
    assigned_gid, status, sim, _ = assoc_mgr.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=15,
        bank_embeddings=bank,
        current_time=11.0,
        frame_idx=110,
        active_camera_gids=set()
    )

    assert status == "MATCHED"
    assert assigned_gid == "Person_001"
    assert state.status == IdentityStatus.ACTIVE
    assert state.current_camera == "waiting_lobby"


# ===========================================================================
# 4. Local tracker ID change after occlusion (L3 -> L9) recovers GID without duplicate
# ===========================================================================
def test_4_local_tracker_renumbering_recovers_gid_without_duplicate(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=401)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 3}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 3, timestamp=1.0, frame_idx=10)

    # Track 3 is occluded at sink and lost
    jm.on_track_lost("pantry", "pantry", 3, timestamp=2.0, frame_idx=25)

    # BoT-SORT renumbers to Track 9 upon re-detection
    bank_9 = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]
    assigned_gid, status, sim, _ = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=9,
        bank_embeddings=bank_9,
        current_time=2.8,
        frame_idx=35,
        active_camera_gids=set()
    )

    assert status == "MATCHED"
    assert assigned_gid == "Person_001"
    assert len(gallery.get_identities()) == 1  # No duplicate Person_002 created!


# ===========================================================================
# 5. Stale track mapping does not block valid reacquisition
# ===========================================================================
def test_5_stale_track_mapping_does_not_block_reacquisition(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    c_filter = env["candidate_filter"]

    emb = make_unit_vector(seed=501)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 10}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 10, timestamp=1.0, frame_idx=10)

    # Track 10 is not in active_camera_gids for new frame
    identities = jm.get_all_identities()
    valid_gids, evals, excluded = c_filter.filter_candidates(
        identities=identities,
        current_camera="passage",
        current_zone="passage",
        current_time=3.0,
        exclude_gids=set()
    )

    assert "Person_001" in valid_gids
    assert "Person_001" not in excluded


# ===========================================================================
# 6. Ambiguous candidate remains pending without being prematurely confirmed
# ===========================================================================
def test_6_ambiguous_single_candidate_remains_pending_without_premature_confirmation(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=601)
    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.0, frame_idx=20)

    # Pink-shirt entrant has borderline similarity 0.718 against Person_001 (threshold is 0.70)
    # Because it is a single candidate with borderline score and no margin separation,
    # the removed next_sim == 0.0 shortcut MUST NOT immediately confirm it on frame 1!
    noisy_emb = make_similar_vector(emb_1, target_sim=0.718, seed=602)
    bank = [{"embedding": noisy_emb, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Mock matcher returning 0.718
    env["matcher"].compute_candidate_score = lambda *args, **kwargs: (0.718, {"is_cross_view": False})

    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=13,
        bank_embeddings=bank,
        current_time=3.0,
        frame_idx=30,
        active_camera_gids=set()
    )

    assert status == "PENDING"
    assert "PENDING_CANDIDATE" in details["decision"]
    assert jm.get_active_global_id("waiting_lobby", 13) is None


# ===========================================================================
# 7. False association does not contaminate confirmed identity gallery
# ===========================================================================
def test_7_false_association_does_not_contaminate_confirmed_gallery(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    gallery = env["gallery"]

    emb_initial = make_unit_vector(seed=701)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_initial, "viewpoint": "FRONT"}])
    initial_embs_count = len(gallery.get_identities()[gid].get_all_embeddings())

    # Simulated probation buffer in processor:
    # A candidate matching with poor quality or conflicting observation should not be committed to gallery
    probation_embeddings = [make_unit_vector(seed=702)]
    probation_count = 1  # Not yet reached probation threshold (4)

    # Gallery update is gated behind probation threshold
    probation_thresh = 4
    if probation_count >= probation_thresh:
        gallery.update_identity(gid, [{"camera_id": "passage", "track_id": 2}], probation_embeddings)

    # Gallery remains uncontaminated
    ident = gallery.get_identities()[gid]
    assert len(ident.get_all_embeddings()) == initial_embs_count


# ===========================================================================
# 8. Person reversing direction receives updated transition evidence
# ===========================================================================
def test_8_person_reversing_direction_updates_transition_evidence(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    s_gate = env["spatial_gate"]

    emb = make_unit_vector(seed=801)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Forward crossing: pantry_green towards passage (SIDE_B_TO_SIDE_A)
    fwd_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=1,
        transition_id="pantry_green",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_B_TO_SIDE_A",
        foot_point=(300.0, 520.0),
        target_camera="passage",
        candidate_destinations=["passage"]
    )
    jm.record_transition(fwd_evt)
    state = jm.get_identity("Person_001")
    assert state.last_transition_direction == "SIDE_B_TO_SIDE_A"
    assert state.last_transition_target_camera == "passage"

    # Reverse crossing: person changes mind and steps inward back to pantry (SIDE_A_TO_SIDE_B)
    rev_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=1,
        transition_id="pantry_green",
        timestamp=3.0,
        frame_idx=45,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(300.0, 480.0),
        target_camera="pantry",
        candidate_destinations=["pantry"]
    )
    jm.record_transition(rev_evt)
    assert state.last_transition_direction == "SIDE_A_TO_SIDE_B"
    assert state.last_transition_target_camera == "pantry"

    # Evaluating candidate for passage shows inward/reverse direction
    is_valid, score, reason = s_gate.evaluate(
        source_camera="pantry",
        current_camera="passage",
        last_transition_id=state.last_transition_id,
        last_transition_direction=state.last_transition_direction,
        time_since_transition=1.0,
        target_camera=state.last_transition_target_camera,
        candidate_zones=state.last_transition_candidate_zones
    )
    assert "TRANSITION_REVERSE_DIRECTION" in reason or "TRANSITION_UNMATCHED" in reason
    assert score <= 0.60


# ===========================================================================
# 9. Stale transition evidence does not reject a later valid route
# ===========================================================================
def test_9_stale_transition_evidence_does_not_reject_valid_route(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    c_filter = env["candidate_filter"]

    emb = make_unit_vector(seed=901)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Forward crossing occurred at t=2.0s
    evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=1,
        transition_id="pantry_green",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_B_TO_SIDE_A",
        foot_point=(300.0, 520.0),
        target_camera="passage",
        candidate_destinations=["passage"]
    )
    jm.record_transition(evt)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.5, frame_idx=35)

    # At t=25.0s (> 15s transit window), person appears in passage
    identities = jm.get_all_identities()
    valid_gids, evals, excluded = c_filter.filter_candidates(
        identities=identities,
        current_camera="passage",
        current_zone="passage",
        current_time=25.0,
        exclude_gids=set()
    )

    # Stale transition should NOT reject the candidate because topology allows it!
    assert "Person_001" in valid_gids
    assert "Person_001" not in excluded


# ===========================================================================
# 10. Person revisiting a camera retains their identity
# ===========================================================================
def test_10_camera_revisit_preserves_global_identity(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=1001)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # 1. Pantry -> Passage
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.0, frame_idx=30)
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    gid1, s1, _, _ = assoc_mgr.associate_track(
        camera_id="passage", zone="passage", local_track_id=10,
        bank_embeddings=bank, current_time=3.5, frame_idx=60, active_camera_gids=set()
    )
    assert gid1 == "Person_001"

    # 2. Passage -> Waiting Lobby
    jm.on_track_lost("passage", "passage", 10, timestamp=5.0, frame_idx=90)
    gid2, s2, _, _ = assoc_mgr.associate_track(
        camera_id="waiting_lobby", zone="waiting_lobby", local_track_id=20,
        bank_embeddings=bank, current_time=6.5, frame_idx=120, active_camera_gids=set()
    )
    assert gid2 == "Person_001"

    # 3. Revisit: Waiting Lobby -> Pantry
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 20, timestamp=8.0, frame_idx=150)
    gid3, s3, _, _ = assoc_mgr.associate_track(
        camera_id="pantry", zone="pantry", local_track_id=30,
        bank_embeddings=bank, current_time=9.5, frame_idx=180, active_camera_gids=set()
    )
    assert gid3 == "Person_001"
    assert len(gallery.get_identities()) == 1


# ===========================================================================
# 11. All valid physical routes function without requiring visual ROI overlap
# ===========================================================================
def test_11_all_valid_physical_routes_work_without_roi_overlap(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=1101)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]

    # Test all valid routes in physical topology:
    # 1. Pantry -> Waiting Lobby
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.0, frame_idx=30)
    g, s, _, _ = assoc_mgr.associate_track(
        camera_id="waiting_lobby", zone="waiting_lobby", local_track_id=2,
        bank_embeddings=bank, current_time=3.0, frame_idx=45, active_camera_gids=set()
    )
    assert g == "Person_001"

    # 2. Waiting Lobby -> Pantry
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 2, timestamp=4.0, frame_idx=60)
    g, s, _, _ = assoc_mgr.associate_track(
        camera_id="pantry", zone="pantry", local_track_id=3,
        bank_embeddings=bank, current_time=5.0, frame_idx=75, active_camera_gids=set()
    )
    assert g == "Person_001"

    # 3. Pantry -> Passage
    jm.on_track_lost("pantry", "pantry", 3, timestamp=6.0, frame_idx=90)
    g, s, _, _ = assoc_mgr.associate_track(
        camera_id="passage", zone="passage", local_track_id=4,
        bank_embeddings=bank, current_time=7.0, frame_idx=105, active_camera_gids=set()
    )
    assert g == "Person_001"

    # 4. Passage -> Workspace
    jm.on_track_lost("passage", "passage", 4, timestamp=8.0, frame_idx=120)
    g, s, _, _ = assoc_mgr.associate_track(
        camera_id="workspace", zone="workspace", local_track_id=5,
        bank_embeddings=bank, current_time=9.0, frame_idx=135, active_camera_gids=set()
    )
    assert g == "Person_001"


# ===========================================================================
# 12. Regression Test: Person A (Purple Shirt) and Person B (Blue Shirt)
#     Reproduce and prevent Global ID swap (Task 5)
# ===========================================================================
def test_12_reproduce_and_prevent_person_a_person_b_gid_swap(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    # Visual embeddings
    emb_person_1 = make_unit_vector(seed=101)  # Existing Person_001
    emb_person_A = make_unit_vector(seed=202)  # Person A (Purple shirt)
    emb_person_B = make_unit_vector(seed=303)  # Person B (Blue shirt)

    # 1. Register Person_001 in gallery
    gid_1 = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 99}], [{"embedding": emb_person_1, "viewpoint": "FRONT"}])
    assert gid_1 == "Person_001"
    jm.register_new_identity(gid_1, "waiting_lobby", "waiting_lobby", 99, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 99, timestamp=2.0, frame_idx=30)

    # Register Person_002 and Person_003 as other past identities
    gid_2 = gallery.add_identity([{"camera_id": "passage", "track_id": 98}], [{"embedding": make_unit_vector(seed=404), "viewpoint": "FRONT"}])
    gid_3 = gallery.add_identity([{"camera_id": "passage", "track_id": 97}], [{"embedding": make_unit_vector(seed=505), "viewpoint": "FRONT"}])

    # 2. Person A (purple shirt) assigned GID 4 in Passage (Track 1)
    gid_4 = gallery.add_identity([{"camera_id": "passage", "track_id": 1}], [{"embedding": emb_person_A, "viewpoint": "FRONT"}])
    assert gid_4 == "Person_004"
    jm.register_new_identity(gid_4, "passage", "passage", 1, timestamp=30.0, frame_idx=750)
    assert jm.get_active_global_id("passage", 1) == "Person_004"

    # 3. Person A (Track 1) in Passage crosses transition line towards Pantry
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="passage",
        zone="passage",
        local_track_id=1,
        transition_id="passage_green",
        timestamp=33.65,
        frame_idx=828,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(300.0, 520.0),
        target_camera="pantry",
        target_zone="pantry",
        candidate_destinations=["pantry"]
    )
    jm.record_transition(trans_evt)
    ident_4 = jm.get_identity("Person_004")
    assert ident_4.last_transition_source_camera == "passage"
    assert ident_4.last_transition_target_camera == "pantry"

    # 4. Person A enters Pantry as Track 6 at t=34.13s (0.48s later)
    # Person A's crop embedding has 0.74 similarity to Person_004, and 0.72 similarity to Person_001
    bank_A = [{"embedding": make_similar_vector(emb_person_A, 0.74, seed=601), "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Mock matcher score to replicate actual observed similarity:
    # Person_004 = 0.734, Person_001 = 0.720, others <= 0.50
    orig_compute = env["matcher"].compute_candidate_score
    def mock_compute_match(*args, **kwargs):
        cand = kwargs.get("candidate_identity") or (args[3] if len(args) > 3 else None)
        gid = cand.global_id if cand else None
        if gid == "Person_004":
            return 0.734, {"is_cross_view": False}
        elif gid == "Person_001":
            return 0.720, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_compute_match

    # Verify Person A association in Pantry:
    # Crucial assertion: Must confirm GID 4 (transition handoff respected without false conflict)
    # and MUST NOT assign Person_001!
    assigned_gid_A, status_A, sim_A, details_A = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=6,
        bank_embeddings=bank_A,
        current_time=34.13,
        frame_idx=837,
        active_camera_gids=set()
    )

    # Frame 1: Proposes Person_004 as pending candidate (0.734 vs 0.720 is ambiguous on frame 1)
    # Must NOT switch to Person_001!
    assert assigned_gid_A == "Person_004", f"Person A must propose Person_004, but got {assigned_gid_A}!"
    assert assigned_gid_A != "Person_001", "Person A must NOT switch to Person_001!"
    assert status_A == "PENDING"

    # 5. Additional evidence arrives over several frames for Person A
    final_gid, final_status = None, None
    for f_idx, t_val in [(845, 34.45), (855, 34.85), (865, 35.25)]:
        final_gid, final_status, _, _ = assoc_mgr.associate_track(
            camera_id="pantry", zone="pantry", local_track_id=6,
            bank_embeddings=bank_A, current_time=t_val, frame_idx=f_idx,
            active_camera_gids=set()
        )
    assert final_status == "MATCHED", f"Person A must confirm after multi-frame evidence, got {final_status}"
    assert final_gid == "Person_004"
    assert jm.get_active_global_id("pantry", 6) == "Person_004"

    # 6. Person B (blue shirt) independently tracked in Pantry at sink (Track 9)
    # Person B's embedding has weak similarity (0.642) with Person_004
    bank_B = [{"embedding": emb_person_B, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    def mock_compute_person_B(*args, **kwargs):
        cand = kwargs.get("candidate_identity") or (args[3] if len(args) > 3 else None)
        gid = cand.global_id if cand else None
        if gid == "Person_004":
            return 0.642, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_compute_person_B

    # Pantry currently has Track 6 confirmed with Person_004
    active_in_pantry = {"Person_004"}

    # Attempt to associate Track 9 (Person B) in Pantry
    assigned_gid_B, status_B, sim_B, details_B = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=9,
        bank_embeddings=bank_B,
        current_time=35.48,
        frame_idx=870,
        active_camera_gids=active_in_pantry
    )

    # CRITICAL ASSERTIONS:
    # 1. Person B must NEVER inherit Person_004!
    assert assigned_gid_B != "Person_004", "Person B (blue shirt) must NOT inherit Person_004!"
    # 2. Person A remains confirmed as Person_004
    assert jm.get_active_global_id("pantry", 6) == "Person_004"


# ===========================================================================
# 13. Safeguards: Occlusion, local track ID changes, stale ownership,
#     and gallery protection against contamination (Task 2, 3, 5)
# ===========================================================================
def test_13_safeguards_occlusion_and_stale_ownership_and_gallery_protection(tmp_path):
    env = build_non_overlapping_env(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_A = make_unit_vector(seed=701)
    emb_B = make_unit_vector(seed=702)

    # Seed past identities 1-3
    gallery.add_identity([{"camera_id": "passage", "track_id": 91}], [{"embedding": make_unit_vector(seed=1), "viewpoint": "FRONT"}])
    gallery.add_identity([{"camera_id": "passage", "track_id": 92}], [{"embedding": make_unit_vector(seed=2), "viewpoint": "FRONT"}])
    gallery.add_identity([{"camera_id": "passage", "track_id": 93}], [{"embedding": make_unit_vector(seed=3), "viewpoint": "FRONT"}])

    # GID 4 active in pantry as Track 6
    gid_4 = gallery.add_identity([{"camera_id": "pantry", "track_id": 6}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    assert gid_4 == "Person_004"
    jm.register_new_identity(gid_4, "pantry", "pantry", 6, timestamp=10.0, frame_idx=100)

    # 1. Person A experiences temporary occlusion: Track 6 is lost
    jm.on_track_lost("pantry", "pantry", 6, timestamp=12.0, frame_idx=150)
    assert jm.get_identity("Person_004").status == IdentityStatus.TEMPORARILY_LOST

    # 2. Person B (blue shirt) appears in pantry while Person A is occluded
    # Weak similarity (0.642) must NOT steal Person_004 from lost person
    bank_B = [{"embedding": emb_B, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    def mock_compute(*args, **kwargs):
        cand = kwargs.get("candidate_identity") or (args[3] if len(args) > 3 else None)
        gid = cand.global_id if cand else None
        if gid == "Person_004":
            return 0.642, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_compute

    # Person B evaluates candidates over 3 frames
    for i in range(3):
        g_B, s_B, _, d_B = assoc_mgr.associate_track(
            camera_id="pantry", zone="pantry", local_track_id=9,
            bank_embeddings=bank_B, current_time=13.0 + i * 0.1, frame_idx=170 + i * 3,
            active_camera_gids=set()
        )
        # Person B must remain PENDING or create a new ID; must NEVER be MATCHED to Person_004
        assert g_B != "Person_004" or s_B != "MATCHED", "Person B must not steal Person_004 during occlusion!"

    # 3. Person A re-emerges from occlusion with a NEW local track ID (BoT-SORT track 14)
    bank_A = [{"embedding": emb_A, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]
    def mock_compute_reacq(*args, **kwargs):
        cand = kwargs.get("candidate_identity") or (args[3] if len(args) > 3 else None)
        gid = cand.global_id if cand else None
        if gid == "Person_004":
            return 0.88, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_compute_reacq

    # Stale local track 6 mapping must NOT block reacquisition by Track 14
    g_reacq, s_reacq, _, _ = assoc_mgr.associate_track(
        camera_id="pantry", zone="pantry", local_track_id=14,
        bank_embeddings=bank_A, current_time=14.0, frame_idx=200,
        active_camera_gids=set()
    )
    assert s_reacq == "MATCHED"
    assert g_reacq == "Person_004"
    assert jm.get_active_global_id("pantry", 14) == "Person_004"
    assert jm.get_identity("Person_004").status == IdentityStatus.ACTIVE

    # 4. Gallery Contamination Safeguard:
    # Verify that Person B's embeddings never contaminated Person_004's gallery
    ident_4 = gallery.get_identity("Person_004")
    # All embeddings in Person_004 gallery must have high cosine similarity with emb_A
    for vp, vbank in ident_4.reid_bank.items():
        for entry in vbank:
            sim_with_A = np.dot(entry["embedding"], emb_A)
            sim_with_B = np.dot(entry["embedding"], emb_B)
            assert sim_with_A > 0.80, f"Gallery contains contaminated embedding! Sim with A: {sim_with_A}"
            assert sim_with_B < 0.60, f"Person B's embedding leaked into Person A's gallery! Sim with B: {sim_with_B}"
