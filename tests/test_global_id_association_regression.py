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


def build_regression_env(tmp_path, custom_config: Dict[str, Any] = None):
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
            "passage": ["waiting_lobby", "pantry"],
            "restricted_vault": []  # Isolated camera
        },
        "temporal_constraints": {
            "same_camera_reacquisition_max_seconds": 60.0,
            "overlapping_camera_pairs": [
                ["waiting_lobby", "pantry"],
                ["waiting_lobby", "passage"],
                ["pantry", "passage"]
            ],
            "pantry_to_passage": {"min_seconds": 0.5, "max_seconds": 30.0},
            "waiting_lobby_to_pantry": {"min_seconds": 0.5, "max_seconds": 30.0}
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
        "assoc_mgr": assoc_mgr,
        "topology_gate": topology_gate,
        "temporal_gate": temporal_gate,
        "spatial_gate": spatial_gate,
        "candidate_filter": candidate_filter
    }


# ===========================================================================
# 1. Local track disappearing and returning with the same global identity
# ===========================================================================
def test_1_track_disappearing_and_returning_same_global_identity(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb = make_unit_vector(seed=101)
    gid = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)

    # Track 1 disappears
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 1, timestamp=5.0, frame_idx=120)
    assert jm.get_identity(gid).status in [IdentityStatus.OCCLUDED, IdentityStatus.TEMPORARILY_LOST]

    # Reappears as Track 5 with high visual similarity
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        bank_embeddings=bank,
        current_time=8.0,
        frame_idx=200,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert assigned == gid
    assert details["decision"] == "MATCH_CONFIRMED"


# ===========================================================================
# 2. Two people crossing paths without swapping identities
# ===========================================================================
def test_2_crossing_paths_without_swapping_identities(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb_A = make_unit_vector(seed=201)
    emb_B = make_unit_vector(seed=202)

    gid_A = gallery.add_identity([{"camera_id": "pantry", "track_id": 2}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_A, "pantry", "pantry", 2, timestamp=1.0, frame_idx=10)

    gid_B = gallery.add_identity([{"camera_id": "pantry", "track_id": 3}], [{"embedding": emb_B, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_B, "pantry", "pantry", 3, timestamp=1.0, frame_idx=10)

    # During crossing, both are active simultaneously in the same camera
    bank_A = [{"embedding": emb_A, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]
    bank_B = [{"embedding": emb_B, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]

    # Track 2 matches A; active_camera_gids for Track 2 includes B
    assigned_A, status_A, _, _ = am.associate_track(
        camera_id="pantry", zone="pantry", local_track_id=2,
        bank_embeddings=bank_A, current_time=2.0, frame_idx=50,
        active_camera_gids={gid_B}
    )
    assert assigned_A == gid_A

    # Track 3 matches B; active_camera_gids for Track 3 includes A
    assigned_B, status_B, _, _ = am.associate_track(
        camera_id="pantry", zone="pantry", local_track_id=3,
        bank_embeddings=bank_B, current_time=2.0, frame_idx=50,
        active_camera_gids={gid_A}
    )
    assert assigned_B == gid_B
    assert assigned_A != assigned_B


# ===========================================================================
# 3. One person occluded and subsequently recovered
# ===========================================================================
def test_3_occluded_and_subsequently_recovered(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb = make_unit_vector(seed=301)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 3}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 3, timestamp=10.0, frame_idx=250)

    # Occluded for 10 frames, tracker drops Track 3
    jm.on_track_lost("pantry", "pantry", 3, timestamp=15.0, frame_idx=375)
    assert jm.get_identity(gid).status in [IdentityStatus.OCCLUDED, IdentityStatus.TEMPORARILY_LOST]

    # Person re-emerges as Track 9
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=9,
        bank_embeddings=bank,
        current_time=16.0,
        frame_idx=400,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert assigned == gid
    # Re-acquisition logged as TRACK_RECOVERED event in journey history
    ident = jm.get_identity(gid)
    assert any(e.event == "TRACK_RECOVERED" for e in ident.journey_history)


# ===========================================================================
# 4. Two different people having similar ReID embeddings (no steal)
# ===========================================================================
def test_4_similar_reid_embeddings_cannot_steal_identity(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=401)
    # Similar embedding (similarity ~0.76)
    emb_sim = make_similar_vector(emb_1, target_sim=0.76, seed=402)

    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Person 1 is actively in pantry; Track 8 appears in pantry with emb_sim
    bank_8 = [{"embedding": emb_sim, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Track 8 cannot steal Person 1's identity because Person 1 is occupied in the camera
    assigned, status, sim, details = am.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=8,
        bank_embeddings=bank_8,
        current_time=1.5,
        frame_idx=15,
        active_camera_gids={gid_1}
    )
    assert assigned != gid_1
    assert "EXCLUDED_OCCUPIED_IN_CAMERA" in details["rejected_candidates"].values()


# ===========================================================================
# 5. Valid cross-camera transition
# ===========================================================================
def test_5_valid_cross_camera_transition(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb = make_unit_vector(seed=501)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 6}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 6, timestamp=1.0, frame_idx=1)

    # Person leaves pantry and crosses transition to waiting_lobby
    jm.on_track_lost("pantry", "pantry", 6, timestamp=5.0, frame_idx=120)

    # Person appears in waiting_lobby at t=6.0s (1.0s transit, within valid 0.5-30s window)
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=2,
        bank_embeddings=bank,
        current_time=6.0,
        frame_idx=145,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert assigned == gid
    assert details["decision"] == "MATCH_CONFIRMED"


# ===========================================================================
# 6. Invalid camera transition rejected by topology
# ===========================================================================
def test_6_invalid_camera_transition_rejected_by_topology(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb = make_unit_vector(seed=601)
    gid = gallery.add_identity([{"camera_id": "restricted_vault", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "restricted_vault", "restricted_vault", 1, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("restricted_vault", "restricted_vault", 1, timestamp=2.0, frame_idx=50)

    # Track appears in pantry with same embedding, but restricted_vault has NO topology connection to pantry
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.90} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=5,
        bank_embeddings=bank,
        current_time=3.0,
        frame_idx=75,
        active_camera_gids=set()
    )
    # Must reject candidate via TOPOLOGY_REJECTED
    assert gid in details["rejected_candidates"]
    assert "TOPOLOGY_REJECTED" in details["rejected_candidates"][gid]
    assert assigned != gid


# ===========================================================================
# 7. Simultaneous visibility in overlapping cameras
# ===========================================================================
def test_7_simultaneous_visibility_in_overlapping_cameras(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    emb = make_unit_vector(seed=701)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 6}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 6, timestamp=10.0, frame_idx=240)

    # Person stands in doorway connecting pantry and waiting_lobby (concurrent/overlapping visibility)
    bank = [{"embedding": emb, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=8,
        bank_embeddings=bank,
        current_time=10.05,
        frame_idx=241,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert assigned == gid
    # Both tracks mapped concurrently in overlapping cameras
    assert jm.get_active_global_id("pantry", 6) == gid
    assert jm.get_active_global_id("waiting_lobby", 8) == gid


# ===========================================================================
# 8. Ambiguous matches remain pending instead of being forcibly assigned
# ===========================================================================
def test_8_ambiguous_matches_remain_pending(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery, am = env["journey_mgr"], env["gallery"], env["assoc_mgr"]

    base = make_unit_vector(seed=800)
    # Two candidates in gallery with almost identical embeddings to new track
    emb_A = make_similar_vector(base, target_sim=0.74, seed=801)
    emb_B = make_similar_vector(base, target_sim=0.73, seed=802)  # Margin only 0.01 (< ambig_margin 0.04)

    gid_A = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb_A, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_A, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 1, timestamp=2.0, frame_idx=50)

    gid_B = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 2}], [{"embedding": emb_B, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_B, "waiting_lobby", "waiting_lobby", 2, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 2, timestamp=2.0, frame_idx=50)

    # New track with base embedding
    bank = [{"embedding": base, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    assigned, status, sim, details = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=10,
        bank_embeddings=bank,
        current_time=3.0,
        frame_idx=75,
        active_camera_gids=set()
    )
    # Must remain PENDING with AMBIGUOUS_MATCH diagnostic, not forcibly confirmed
    assert status == "PENDING"
    assert "AMBIGUOUS_MATCH" in details["decision"]


# ===========================================================================
# 9. Gallery contamination prevention
# ===========================================================================
def test_9_gallery_contamination_prevention(tmp_path):
    env = build_regression_env(tmp_path)
    gallery = env["gallery"]

    emb_clean = make_unit_vector(seed=901)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_clean, "viewpoint": "FRONT", "quality": 0.90}])
    ident = gallery.get_identity(gid)
    initial_count = len(ident.get_all_embeddings())

    # Add embeddings for Track 2
    emb_good = make_unit_vector(seed=903)
    good_batch = [{"embedding": emb_good, "quality": 0.85, "viewpoint": "FRONT", "camera_id": "pantry", "track_id": 2}]
    gallery.update_identity(gid, [{"camera_id": "pantry", "track_id": 2}], good_batch)
    assert len(ident.get_all_embeddings()) == initial_count + 1

    # Rollback track 2: removes track 2 embeddings cleanly without contaminating Person 1's gallery
    gallery.rollback_track(gid, "pantry", 2)
    assert len(ident.get_all_embeddings()) == initial_count


# ===========================================================================
# 10. Global identity persistence after local track termination
# ===========================================================================
def test_10_global_identity_persistence_after_local_track_termination(tmp_path):
    env = build_regression_env(tmp_path)
    jm, gallery = env["journey_mgr"], env["gallery"]

    emb = make_unit_vector(seed=1001)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=10)

    # Local tracker terminates track 1
    jm.on_track_lost("pantry", "pantry", 1, timestamp=5.0, frame_idx=120)

    # Global identity MUST persist in journey_mgr and gallery
    assert gid in jm.get_all_identities()
    assert gid in gallery.get_identities()
    state = jm.get_identity(gid)
    assert state.status in [IdentityStatus.OCCLUDED, IdentityStatus.TEMPORARILY_LOST]
    # Local track mapping is safely unmapped without destroying global identity
    assert jm.get_active_global_id("pantry", 1) is None
