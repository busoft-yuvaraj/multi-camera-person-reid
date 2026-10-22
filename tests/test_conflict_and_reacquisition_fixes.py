import pytest
import numpy as np
from typing import Dict, Any

from app.association.topology import TopologyGate
from app.association.temporal_gate import TemporalGate
from app.association.spatial_gate import SpatialGate
from app.association.candidate_filter import CandidateFilter
from app.association.association_manager import AssociationManager
from app.journey.models import GlobalIdentityState, IdentityStatus, JourneyEvent
from app.journey.journey_store import JourneyStore
from app.journey.journey_manager import JourneyManager
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

def build_test_environment(tmp_path, custom_config: Dict[str, Any] = None):
    config = {
        "threshold_match": 0.70,
        "threshold_ambiguous": 0.60,
        "threshold_cross_view": 0.55,
        "reid_ambiguous_margin": 0.04,
        "pending_confirm_frames": 3,
        "reid_new_id_confirm_frames": 15,
        "track_lost_frames": 60,
        "topology": {
            "waiting_lobby": ["pantry", "passage"],
            "pantry": ["waiting_lobby", "passage"],
            "passage": ["waiting_lobby", "pantry", "workspace"],
            "workspace": ["passage"]
        },
        "temporal_constraints": {
            "same_camera_reacquisition_max_seconds": 60.0,
            "overlapping_camera_pairs": [["waiting_lobby", "pantry"]]
        }
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
        overlapping_camera_pairs=config["temporal_constraints"]["overlapping_camera_pairs"]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gallery.pkl"), clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=config)
    topology_gate = TopologyGate(topology_config=config["topology"])
    temporal_gate = TemporalGate(temporal_config=config["temporal_constraints"])
    spatial_gate = SpatialGate()
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
        "assoc_mgr": assoc_mgr
    }

# ---------------------------------------------------------------------------
# Test 1: Simultaneous observations in configured overlapping cameras retain one Global ID
# ---------------------------------------------------------------------------
def test_1_overlapping_cameras_retain_one_global_id(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_A = make_unit_vector(seed=101)
    gid = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)

    # Person A is simultaneously visible in pantry (Track 2) at t=1.05s
    bank_embs_2 = [{"embedding": emb_A, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        bank_embeddings=bank_embs_2,
        current_time=1.05,
        frame_idx=2,
        active_camera_gids=set()
    )

    assert status == "MATCHED"
    assert assigned_gid == "Person_001"
    assert jm.get_active_global_id("waiting_lobby", 1) == "Person_001"
    assert jm.get_active_global_id("pantry", 2) == "Person_001"

    # Verify no GLOBAL_ID_CONFLICT event was generated
    ident = jm.get_identity("Person_001")
    conflict_events = [e for e in ident.journey_history if e.event == "GLOBAL_ID_CONFLICT"]
    assert len(conflict_events) == 0

# ---------------------------------------------------------------------------
# Test 2: Two distinct people cannot improperly share one Global ID
# ---------------------------------------------------------------------------
def test_2_two_distinct_people_cannot_share_global_id(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    emb_A = make_unit_vector(seed=101)
    gid = gallery.add_identity([{"camera_id": "passage", "track_id": 1}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "passage", "passage", 1, timestamp=1.0, frame_idx=1)

    # In waiting_lobby (non-overlapping with passage at t=1.05s, no transition),
    # another track tries to claim Person_001 simultaneously
    confirmed = jm.confirm_association(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=2,
        timestamp=1.05,
        frame_idx=2,
        decision_reason="SIMULTANEOUS_INCOMPATIBLE"
    )
    assert confirmed is False

    ident = jm.get_identity("Person_001")
    conflict_events = [e for e in ident.journey_history if e.event == "GLOBAL_ID_CONFLICT"]
    assert len(conflict_events) == 1
    assert jm.get_active_global_id("waiting_lobby", 2) is None

# ---------------------------------------------------------------------------
# Test 3: Stale local-track mappings do not block reacquisition
# ---------------------------------------------------------------------------
def test_3_stale_local_track_mappings_do_not_block_reacquisition(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    c_filter = env["candidate_filter"]

    emb_A = make_unit_vector(seed=101)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Track 1 disappeared (not observed in current frame).
    # Lifecycle-aware filter: active_camera_gids only contains currently observed tracks!
    # Because Track 1 is not observed, active_camera_gids is empty.
    identities = jm.get_all_identities()
    valid_gids, evals, excluded = c_filter.filter_candidates(
        identities=identities,
        current_camera="pantry",
        current_zone="pantry",
        current_time=3.0,
        exclude_gids=set()  # Track 1 is not in active_camera_gids
    )
    assert "Person_001" in valid_gids
    assert "Person_001" not in excluded

# ---------------------------------------------------------------------------
# Test 4: Lost track changing from L3 to L9 recovers original Global ID
# ---------------------------------------------------------------------------
def test_4_lost_track_changing_from_l3_to_l9_recovers_original_global_id(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_A = make_unit_vector(seed=200)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 3}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 3, timestamp=1.0, frame_idx=1)

    # Track 3 is occluded and lost
    jm.on_track_lost("pantry", "pantry", 3, timestamp=2.0, frame_idx=30)
    assert jm.get_identity("Person_001").status == IdentityStatus.TEMPORARILY_LOST

    # Track 9 appears in pantry with same embedding
    bank_embs_9 = [{"embedding": emb_A, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=9,
        bank_embeddings=bank_embs_9,
        current_time=2.5,
        frame_idx=45,
        active_camera_gids=set()
    )

    assert status == "MATCHED"
    assert assigned_gid == "Person_001"
    assert jm.get_active_global_id("pantry", 9) == "Person_001"
    assert jm.get_identity("Person_001").status == IdentityStatus.ACTIVE

# ---------------------------------------------------------------------------
# Test 5: Valid second-best candidate is considered after a genuine conflict
# ---------------------------------------------------------------------------
def test_5_valid_second_best_candidate_considered_after_conflict(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=301)
    emb_2 = make_unit_vector(seed=302)

    gid_1 = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)

    gid_2 = gallery.add_identity([{"camera_id": "passage", "track_id": 2}], [{"embedding": emb_2, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_2, "passage", "passage", 2, timestamp=0.5, frame_idx=1)
    jm.on_track_lost("passage", "passage", 2, timestamp=0.8, frame_idx=10) # Person_002 is lost and re-acquirable

    # Person X arrives in passage at t=1.6s (elapsed=0.6s from waiting_lobby, within 0.5s-15s temporal window).
    # Person X's embedding is closer to Person_001 (0.85), but also similar to Person_002 (0.75).
    # Since Person_001 is active in waiting_lobby (time_diff=0.6s < 1.0s, conflict!),
    # confirm_association rejects Person_001, and AssociationManager evaluates candidate #2 (Person_002)!
    noisy_emb = make_similar_vector(emb_2, target_sim=0.76, seed=400)
    bank_embs_X = [{"embedding": noisy_emb, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Make sure noisy_emb scores 0.85 against Person_001 and 0.75 against Person_002
    orig_compute = env["matcher"].compute_candidate_score
    def mock_compute(t_prototype, t_par_data, dom_t_view, candidate_identity, current_camera, consecutive_candidate_count):
        if candidate_identity.global_id == "Person_001":
            return 0.85, {"is_cross_view": False}
        elif candidate_identity.global_id == "Person_002":
            return 0.75, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_compute

    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="passage",
        zone="passage",
        local_track_id=5,
        bank_embeddings=bank_embs_X,
        current_time=1.6,
        frame_idx=15,
        active_camera_gids=set()
    )

    assert status == "MATCHED"
    assert assigned_gid == "Person_002"
    assert "ALTERNATIVE" in details["decision"]
    assert "Person_001" in details["conflict_rejections"]

# ---------------------------------------------------------------------------
# Test 6 & 7: Conflicts do not remain pending indefinitely or create duplicate IDs
# ---------------------------------------------------------------------------
def test_6_and_7_conflict_does_not_create_duplicate_id_after_fixed_frames(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=501)
    gid = gallery.add_identity([{"camera_id": "passage", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "passage", "passage", 1, timestamp=1.0, frame_idx=1)

    # In waiting_lobby (conflicting camera), Track 2 arrives with identical embedding to Person_001.
    # Track 1 in passage remains actively tracked simultaneously.
    bank_embs_2 = [{"embedding": emb_1, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]

    for f in range(2, 25): # 23 consecutive frames (exceeds 15-frame window)
        t = 1.05 + f * 0.04
        # Keep passage Track 1 actively updated at t - 0.7s:
        # 1. TemporalGate sees elapsed = 0.7s >= 0.5s -> Valid candidate
        # 2. JourneyManager sees time_diff = 0.7s < 1.0s -> Genuinely non-overlapping simultaneous conflict!
        jm.update_track_presence("passage", 1, timestamp=t - 0.7, frame_idx=f)

        assigned_gid, status, sim, details = assoc_mgr.associate_track(
            camera_id="waiting_lobby",
            zone="waiting_lobby",
            local_track_id=2,
            bank_embeddings=bank_embs_2,
            current_time=t,
            frame_idx=f,
            active_camera_gids=set()
        )
        assert status == "PENDING"
        assert "CONFLICT_DEFERRED" in details["decision"]

    # Verify no-match counter was NOT incremented and NO new global ID was created
    assert assoc_mgr.no_match_counters.get(("waiting_lobby", 2), 0) == 0
    assert len(gallery.get_identities()) == 1
    assert "Person_002" not in gallery.get_identities()

# ---------------------------------------------------------------------------
# Test 8: Expiring one local track does not remove other camera's active mapping
# ---------------------------------------------------------------------------
def test_8_expiring_one_local_track_retains_other_camera_active_mapping(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_A = make_unit_vector(seed=601)
    gid = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)

    # Overlapping confirmation in pantry Track 2
    bank_embs_2 = [{"embedding": emb_A, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]
    assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        bank_embeddings=bank_embs_2,
        current_time=1.05,
        frame_idx=2,
        active_camera_gids=set()
    )
    assert jm.get_active_global_id("pantry", 2) == "Person_001"

    # waiting_lobby Track 1 disappears
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 1, timestamp=3.0, frame_idx=70)

    # pantry Track 2 must still be mapped to Person_001 and status must remain ACTIVE!
    assert jm.get_active_global_id("pantry", 2) == "Person_001"
    ident = jm.get_identity("Person_001")
    assert ident.status == IdentityStatus.ACTIVE

# ---------------------------------------------------------------------------
# Test 9: Pending or rejected candidates do not contaminate confirmed gallery
# ---------------------------------------------------------------------------
def test_9_pending_or_rejected_candidates_do_not_contaminate_gallery(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=701)
    gid = gallery.add_identity([{"camera_id": "passage", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "passage", "passage", 1, timestamp=1.0, frame_idx=1)

    init_exemplar_count = len(gallery.get_identities()["Person_001"].get_all_embeddings())

    # Borderline candidate (sim ~0.62, ambiguous)
    noisy_emb = make_similar_vector(emb_1, target_sim=0.62, seed=800)
    bank_embs_cand = [{"embedding": noisy_emb, "viewpoint": "FRONT", "quality": 0.80} for _ in range(3)]

    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=99,
        bank_embeddings=bank_embs_cand,
        current_time=5.0,
        frame_idx=120,
        active_camera_gids=set()
    )
    assert status == "PENDING"

    # Gallery exemplar count must not change
    assert len(gallery.get_identities()["Person_001"].get_all_embeddings()) == init_exemplar_count
