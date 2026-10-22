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

def build_test_environment(tmp_path, custom_config: Dict[str, Any] = None):
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
            "waiting_lobby": ["pantry", "passage"],
            "pantry": ["waiting_lobby", "passage"],
            "passage": ["waiting_lobby", "pantry", "workspace"],
            "workspace": ["passage"]
        },
        "temporal_constraints": {
            "same_camera_reacquisition_max_seconds": 60.0,
            "overlapping_camera_pairs": [],
            "pantry_to_passage": {"min_seconds": 0.5, "max_seconds": 45.0},
            "passage_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0},
            "pantry_to_waiting_lobby": {"min_seconds": 0.5, "max_seconds": 45.0},
            "waiting_lobby_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0}
        },
        "transition_rules": [
            {
                "from_camera": "pantry",
                "transition": "pantry_yellow",
                "candidate_zones": ["passage"]
            },
            {
                "from_camera": "passage",
                "transition": "passage_green",
                "candidate_zones": ["pantry"]
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
        overlapping_camera_pairs=config["temporal_constraints"]["overlapping_camera_pairs"]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gallery.pkl"), clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=config)
    topology_gate = TopologyGate(topology_config=config["topology"])
    temporal_gate = TemporalGate(temporal_config=config["temporal_constraints"])
    spatial_gate = SpatialGate(transition_rules=config.get("transition_rules", []))
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
# Test 1 & 2: Person 1 and Person 4 have separate confirmed GIDs.
# Momentary occlusion must not allow local track swap.
# ---------------------------------------------------------------------------
def test_1_separate_confirmed_gids_preserved_during_occlusion(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    emb_1 = make_unit_vector(seed=1)
    emb_4 = make_unit_vector(seed=4)

    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    gid_4 = gallery.add_identity([{"camera_id": "pantry", "track_id": 4}], [{"embedding": emb_4, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_4, "pantry", "pantry", 4, timestamp=1.0, frame_idx=1)

    # Both tracks are active in the same room
    assert jm.get_active_global_id("pantry", 1) == "Person_001"
    assert jm.get_active_global_id("pantry", 4) == "Person_002" # second registered identity

    # Track 4 is occluded for 1 frame (timestamp 1.05s).
    # A new track (Track 9) cannot steal GID_4 while Track 4 was seen 0.05s ago
    confirmed = jm.confirm_association(
        global_id="Person_002",
        camera_id="pantry",
        zone="pantry",
        local_track_id=9,
        timestamp=1.05,
        frame_idx=2
    )
    assert confirmed is False, "Same-camera active track must not be replaced during momentary occlusion"

# ---------------------------------------------------------------------------
# Test 3 & 4: Ambiguous embeddings must not swap GID 1 and GID 4.
# Original identity can be reacquired after occlusion.
# ---------------------------------------------------------------------------
def test_3_and_4_ambiguous_embeddings_do_not_swap_and_reacquisition_works(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=10)
    emb_4 = make_unit_vector(seed=40)

    gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_001", "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    gallery.add_identity([{"camera_id": "pantry", "track_id": 4}], [{"embedding": emb_4, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_004", "pantry", "pantry", 4, timestamp=1.0, frame_idx=1)

    # Track 1 disappears (occluded)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=5.0, frame_idx=150)

    # Track 4 is still active in pantry.
    # Track 4 evaluates embeddings that are ambiguous with Person_001
    ambig_emb = make_similar_vector(emb_4, target_sim=0.72, seed=999)
    bank_embs = [{"embedding": ambig_emb, "viewpoint": "FRONT", "quality": 0.8} for _ in range(3)]

    # Person 4 is active in pantry, so Person 4 is occupied
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=4,
        bank_embeddings=bank_embs,
        current_time=5.2,
        frame_idx=156,
        active_camera_gids={"Person_004"}
    )
    # Track 4 must NOT be swapped to Person_001
    assert assigned_gid != "Person_001"

    # Now Person 1 reappears with new track ID 12
    bank_embs_1 = [{"embedding": emb_1, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    assigned_gid_12, status_12, sim_12, details_12 = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=12,
        bank_embeddings=bank_embs_1,
        current_time=5.5,
        frame_idx=165,
        active_camera_gids={"Person_004"}
    )
    # Person 1 is reacquired safely
    assert assigned_gid_12 == "Person_001"
    assert status_12 == "MATCHED"

# ---------------------------------------------------------------------------
# Test 5 & 6: Person 5 crosses from Pantry into Passage.
# Person 5 cannot acquire GID 1 without supporting identity evidence.
# ---------------------------------------------------------------------------
def test_5_and_6_person_5_pantry_to_passage_does_not_steal_gid_1(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_1 = make_unit_vector(seed=101)
    emb_5 = make_unit_vector(seed=505)

    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 10}], [{"embedding": emb_1, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_1, "pantry", "pantry", 10, timestamp=1.0, frame_idx=1)

    gid_5 = gallery.add_identity([{"camera_id": "pantry", "track_id": 26}], [{"embedding": emb_5, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid_5, "pantry", "pantry", 26, timestamp=1.0, frame_idx=1)

    # Person 5 crosses transition line in pantry towards passage at t=10.0
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=26,
        transition_id="pantry_yellow",
        timestamp=10.0,
        frame_idx=300,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(100.0, 200.0),
        global_id=gid_5,
        target_camera="passage",
        candidate_destinations=["passage"],
        candidate_zones=["passage"]
    )
    jm.record_transition(trans_evt)

    # Person 5 walks through the doorway and exits pantry at t=12.5s (2.5s later)
    jm.update_track_presence("pantry", 26, timestamp=12.5, frame_idx=375)
    jm.on_track_lost("pantry", "pantry", 26, timestamp=12.6, frame_idx=378)

    # Track 8 appears in passage at t=13.0s (Person 5 entering passage)
    # Person 5 has similarity to Person_005 (0.75) and only 0.65 to Person_001
    bank_embs_p5 = [{"embedding": emb_5, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Person 1 is STILL in pantry (active on track 10)
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="passage",
        zone="passage",
        local_track_id=8,
        bank_embeddings=bank_embs_p5,
        current_time=13.0,
        frame_idx=390,
        active_camera_gids=set()
    )

    # Must NOT assign Person_001! Must assign Person_005 (gid_5)
    assert assigned_gid == gid_5
    assert assigned_gid != gid_1
    assert status == "MATCHED"
    assert status == "MATCHED"

# ---------------------------------------------------------------------------
# Test 7: A valid transition is not incorrectly rejected as simultaneous occupancy
# ---------------------------------------------------------------------------
def test_7_valid_transition_not_rejected_as_simultaneous_conflict(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    emb = make_unit_vector(seed=777)
    gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_001", "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Record transition from pantry to passage
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=1,
        transition_id="pantry_yellow",
        timestamp=2.0,
        frame_idx=60,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(100.0, 200.0),
        global_id="Person_001",
        target_camera="passage",
        candidate_destinations=["passage"]
    )
    jm.record_transition(trans_evt)

    # Target camera handoff within 15.0s
    confirmed = jm.confirm_association(
        global_id="Person_001",
        camera_id="passage",
        zone="passage",
        local_track_id=5,
        timestamp=2.5,
        frame_idx=75,
        decision_details={"has_matching_transition": True}
    )
    assert confirmed is True, "Valid transition handoff must be confirmed"

# ---------------------------------------------------------------------------
# Test 8: A conflicting top candidate cannot cause premature confirmation of an unsafe runner-up
# ---------------------------------------------------------------------------
def test_8_conflicting_top_candidate_prevents_unsafe_runner_up_confirmation(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb_top = make_unit_vector(seed=801)
    emb_sub = make_unit_vector(seed=802)

    gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_top, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_001", "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    gallery.add_identity([{"camera_id": "pantry", "track_id": 2}], [{"embedding": emb_sub, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_002", "pantry", "pantry", 2, timestamp=1.0, frame_idx=1)
    jm.on_track_lost("pantry", "pantry", 2, timestamp=1.5, frame_idx=45)

    # New track 99 enters pantry. It matches Person_001 with 0.85 (top candidate).
    # But Person_001 is ALREADY OCCUPIED in pantry on track 1!
    # Track 99 only scores 0.62 on Person_002 (weak sub-threshold).
    orig_compute = env["matcher"].compute_candidate_score
    def mock_compute(t_prototype, t_par_data, dom_t_view, candidate_identity, current_camera, consecutive_candidate_count):
        if candidate_identity.global_id == "Person_001":
            return 0.85, {"is_cross_view": False}
        elif candidate_identity.global_id == "Person_002":
            return 0.62, {"is_cross_view": False}
        return 0.40, {}
    env["matcher"].compute_candidate_score = mock_compute

    bank_embs = [{"embedding": emb_top, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]
    assigned_gid, status, sim, details = assoc_mgr.associate_track(
        camera_id="pantry",
        zone="pantry",
        local_track_id=99,
        bank_embeddings=bank_embs,
        current_time=2.0,
        frame_idx=60,
        active_camera_gids={"Person_001"}
    )

    # Track 99 must NOT be confirmed as Person_002 merely because Person_001 was occupied!
    assert status != "MATCHED" or assigned_gid != "Person_002"

# ---------------------------------------------------------------------------
# Test 9: Stale transition evidence cannot associate a person with an unrelated identity
# ---------------------------------------------------------------------------
def test_9_stale_transition_evidence_rejected(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]
    assoc_mgr = env["assoc_mgr"]

    emb = make_unit_vector(seed=901)
    gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity("Person_001", "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # Stale transition crossed 40 seconds ago
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=1,
        transition_id="pantry_yellow",
        timestamp=1.0,
        frame_idx=30,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(100.0, 200.0),
        global_id="Person_001",
        target_camera="passage",
        candidate_destinations=["passage"]
    )
    jm.record_transition(trans_evt)
    jm.on_track_lost("pantry", "pantry", 1, timestamp=2.0, frame_idx=60)

    # Track in passage appears at t=45.0s (time_since_trans = 44s > 15s)
    evals = env["candidate_filter"].filter_candidates(
        identities=jm.get_all_identities(),
        current_camera="passage",
        current_zone="passage",
        current_time=45.0,
        exclude_gids=set()
    )
    # Stale transition must not grant transition handoff
    cand_eval = evals[1].get("Person_001", {})
    assert cand_eval.get("time_since_transition", 0.0) > 15.0 or "STALE" in str(cand_eval.get("spatial_reason"))

# ---------------------------------------------------------------------------
# Test 10: Incorrect associations cannot contaminate permanent galleries
# ---------------------------------------------------------------------------
def test_10_incorrect_associations_do_not_contaminate_gallery(tmp_path):
    env = build_test_environment(tmp_path)
    gallery = env["gallery"]

    emb_clean = make_unit_vector(seed=1001)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb_clean, "viewpoint": "FRONT"}])

    ident = gallery.get_identity(gid)
    initial_count = len(ident.get_all_embeddings())

    # Mistaken embedding added for a wrong track
    emb_wrong = make_unit_vector(seed=9999)
    gallery.update_identity(gid, [{"camera_id": "pantry", "track_id": 99}], [{"embedding": emb_wrong, "viewpoint": "FRONT", "track_id": 99, "camera_id": "pantry"}])
    assert len(ident.get_all_embeddings()) > initial_count

    # Gallery rollback removes the contaminated track embeddings
    removed = gallery.rollback_track(gid, "pantry", 99)
    assert removed > 0
    assert len(ident.get_all_embeddings()) == initial_count

# ---------------------------------------------------------------------------
# Test 11: Two distinct people in non-overlapping ROIs cannot simultaneously own the same active GID
# ---------------------------------------------------------------------------
def test_11_simultaneous_ownership_in_non_overlapping_rois_rejected(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    emb = make_unit_vector(seed=1101)
    gid = gallery.add_identity([{"camera_id": "pantry", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "pantry", "pantry", 1, timestamp=1.0, frame_idx=1)

    # In waiting_lobby (non-overlapping), another track attempts simultaneous ownership without transition
    confirmed = jm.confirm_association(
        global_id=gid,
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=2,
        timestamp=1.2,
        frame_idx=36
    )
    assert confirmed is False, "Simultaneous ownership in non-overlapping cameras must be strictly rejected"

# ---------------------------------------------------------------------------
# Test 12: Valid cross-camera transitions and camera revisits continue to work
# ---------------------------------------------------------------------------
def test_12_valid_cross_camera_transitions_and_revisits_work(tmp_path):
    env = build_test_environment(tmp_path)
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    emb = make_unit_vector(seed=1201)
    gid = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": emb, "viewpoint": "FRONT"}])
    jm.register_new_identity(gid, "waiting_lobby", "waiting_lobby", 1, timestamp=1.0, frame_idx=1)

    # Legitimate move to pantry
    trans_1 = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        transition_id="lobby_pantry",
        timestamp=2.0,
        frame_idx=60,
        direction="EXIT",
        foot_point=(10.0, 20.0),
        global_id=gid,
        target_camera="pantry",
        candidate_destinations=["pantry"]
    )
    jm.record_transition(trans_1)
    jm.on_track_lost("waiting_lobby", "waiting_lobby", 1, timestamp=2.2, frame_idx=66)

    # Confirmed in pantry
    conf_pantry = jm.confirm_association(
        global_id=gid,
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        timestamp=3.0,
        frame_idx=90,
        decision_details={"has_matching_transition": True}
    )
    assert conf_pantry is True

    # Revisit: moves back to waiting_lobby
    trans_2 = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        transition_id="pantry_yellow",
        timestamp=5.0,
        frame_idx=150,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(10.0, 20.0),
        global_id=gid,
        target_camera="waiting_lobby",
        candidate_destinations=["waiting_lobby"]
    )
    jm.record_transition(trans_2)
    jm.on_track_lost("pantry", "pantry", 2, timestamp=5.2, frame_idx=156)

    # Confirmed back in waiting_lobby (revisit)
    conf_revisit = jm.confirm_association(
        global_id=gid,
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=3,
        timestamp=6.0,
        frame_idx=180,
        decision_details={"has_matching_transition": True}
    )
    assert conf_revisit is True, "Camera revisit must be successfully confirmed"

def test_13_waiting_lobby_track_cannot_steal_person_4_from_pantry_pink_shirt(tmp_path):
    """
    Failure scenario:
    - Person 4 (pink shirt) enters Pantry from Passage (valid transition, score 0.77).
    - An unrelated guy in checkered shirt appears in Waiting Lobby (weak score 0.65 to Person 4,
      no transition event, runner-up margin negative).
    - Waiting Lobby must NOT steal Person 4.
    - Pink shirt in Pantry MUST be confirmed as Person 4.
    """
    env = build_test_environment(tmp_path)
    am = env["assoc_mgr"]
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    vec_p4 = make_unit_vector(seed=404)
    gid_4 = gallery.add_identity([{"camera_id": "passage", "track_id": 1}], [{"embedding": vec_p4, "viewpoint": "FRONT", "quality": 0.85}])
    jm.register_new_identity(gid_4, "passage", "passage", local_track_id=1, timestamp=25.0, frame_idx=600)

    # Person 4 crosses transition from passage toward pantry
    trans = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="passage",
        zone="passage",
        local_track_id=1,
        transition_id="passage_green",
        timestamp=33.0,
        frame_idx=800,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(10.0, 20.0),
        global_id=gid_4,
        target_camera="pantry",
        candidate_destinations=["pantry", "waiting_lobby"],
        candidate_zones=["pantry", "waiting_lobby"]
    )
    jm.record_transition(trans)
    jm.on_track_lost("passage", "passage", 1, timestamp=36.0, frame_idx=880)

    # Setup other people already occupied in pantry (Person 1, Person 2, Person 3)
    vec_p1 = make_similar_vector(vec_p4, target_sim=0.78, seed=101)
    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 9}], [{"embedding": vec_p1, "viewpoint": "FRONT", "quality": 0.85}])
    jm.register_new_identity(gid_1, "pantry", "pantry", local_track_id=9, timestamp=10.0, frame_idx=200)

    # Pink shirt guy (Track 6) in pantry: score 0.77 to Person 4 across 3 frames
    vec_pink = make_similar_vector(vec_p4, target_sim=0.77, seed=606)
    for f in range(3):
        gid_pantry, status_pantry, sim_pantry, _ = am.associate_track(
            camera_id="pantry",
            zone="pantry",
            local_track_id=6,
            bank_embeddings=[{"embedding": vec_pink, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)],
            current_time=36.5 + f * 0.1,
            frame_idx=890 + f,
            active_camera_gids={gid_1}
        )
    # Pink shirt in pantry must be confirmed as Person 4
    assert gid_pantry == gid_4
    assert status_pantry == "MATCHED"

    # Now unrelated guy (Track 1) in waiting lobby: weak score 0.65 to Person 4, no transition to waiting lobby
    vec_checkered = make_similar_vector(vec_p4, target_sim=0.65, seed=707)
    gid_lobby, status_lobby, sim_lobby, _ = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        bank_embeddings=[{"embedding": vec_checkered, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)],
        current_time=36.6,
        frame_idx=895,
        active_camera_gids=set()
    )
    # Waiting lobby guy must NOT steal Person 4!
    assert gid_lobby != gid_4, "Waiting lobby track must NOT steal Person 4 from pantry"
    assert status_lobby in ("PENDING", "UNASSIGNED")

def test_14_waiting_lobby_track_cannot_steal_person_1_from_pantry_blue_shirt(tmp_path):
    """
    Failure scenario:
    - Person 1 (blue shirt guy) is active in Pantry (sitting at sink).
    - An unrelated guy in checkered shirt (Track 1) appears in Waiting Lobby.
    - Checkered shirt track has weak score (0.58 < 0.70) to Person 1,
      negative margin relative to gallery runner up (-0.09), and no transition event.
    - Even after 5 consecutive frames (exceeding pending_confirm_frames = 3),
      Waiting Lobby track MUST NOT steal Person 1 or be promoted to Person 1!
    - Person 1 remains confirmed and active in Pantry.
    """
    env = build_test_environment(tmp_path)
    am = env["assoc_mgr"]
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    vec_p1 = make_unit_vector(seed=101)
    gid_1 = gallery.add_identity([{"camera_id": "pantry", "track_id": 9}], [{"embedding": vec_p1, "viewpoint": "FRONT", "quality": 0.85}])
    jm.register_new_identity(gid_1, "pantry", "pantry", local_track_id=9, timestamp=10.0, frame_idx=200)

    vec_p2 = make_unit_vector(seed=202)
    gid_2 = gallery.add_identity([{"camera_id": "pantry", "track_id": 2}], [{"embedding": vec_p2, "viewpoint": "FRONT", "quality": 0.85}])
    jm.register_new_identity(gid_2, "pantry", "pantry", local_track_id=2, timestamp=10.0, frame_idx=200)

    # Blue shirt guy in pantry is active
    assert jm.active_tracks.get(("pantry", 9)) == gid_1

    # Checkered shirt track in waiting lobby produces vector with 0.58 sim to Person 1,
    # but 0.67 sim to Person 2 (gallery margin = -0.09).
    # Mock compute_candidate_score to replicate exact camera scores at f900-f906
    orig_compute = env["matcher"].compute_candidate_score
    def mock_scores(t_prototype, t_par_data, dom_t_view, candidate_identity, current_camera, consecutive_candidate_count):
        if candidate_identity.global_id == gid_1:
            return 0.581, {"is_cross_view": False}
        elif candidate_identity.global_id == gid_2:
            return 0.674, {"is_cross_view": False}
        return 0.50, {}
    env["matcher"].compute_candidate_score = mock_scores

    vec_checkered = make_similar_vector(vec_p1, target_sim=0.581, seed=808)
    for f in range(5):
        assigned_gid, status, sim, details = am.associate_track(
            camera_id="waiting_lobby",
            zone="waiting_lobby",
            local_track_id=1,
            bank_embeddings=[{"embedding": vec_checkered, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)],
            current_time=36.0 + f * 0.1,
            frame_idx=900 + f,
            active_camera_gids=set()
        )
        # MUST NOT be confirmed as Person 1
        assert assigned_gid != gid_1, f"Frame {900+f}: Waiting lobby checkered guy must NOT steal Person 1!"
        assert status != "MATCHED"

    # Verify blue shirt guy in pantry still owns Person 1
    assert jm.active_tracks.get(("pantry", 9)) == gid_1, "Person 1 must still belong to blue shirt guy in Pantry"

def test_15_waiting_lobby_black_shirt_creates_new_id_and_pink_shirt_recovers_after_occlusion(tmp_path):
    """
    Exact real-world CCTV scenario:
    1. Person_004 (pink shirt) is tracked in Pantry as Track 6 from t=10.0 to t=41.33.
    2. A new person (checkered black shirt) enters Waiting Lobby as Track 1 at t=36.15.
    3. Track 1 in Waiting Lobby is a new person and receives a new Global ID (e.g. Person_005)
       after sustained absence of valid candidate match (reid_new_id_confirm_frames).
    4. At t=41.33, Person_004 occludes in Pantry (BoT-SORT drops Track 6).
    5. Track 1 in Waiting Lobby coexisted with Person_004 while Person_004 was in Pantry,
       so Track 1 cannot physically be Person_004 (spacetime simultaneous conflict).
    6. At t=45.02, Person_004 reappears in Pantry as Track 13.
    7. Track 13 in Pantry successfully reacquires Person_004!
    """
    env = build_test_environment(tmp_path)
    am = env["assoc_mgr"]
    jm = env["journey_mgr"]
    gallery = env["gallery"]

    # Setup Person_004 in Pantry (pink shirt guy)
    vec_p4 = make_unit_vector(seed=404)
    gid_4 = gallery.add_identity([{"camera_id": "pantry", "track_id": 6}], [{"embedding": vec_p4, "viewpoint": "FRONT", "quality": 0.88}])
    jm.register_new_identity(gid_4, "pantry", "pantry", local_track_id=6, timestamp=10.0, frame_idx=250)

    # 1. New entrant (black checkered shirt, Track 1) appears in Waiting Lobby at t=36.15
    vec_black = make_unit_vector(seed=909)  # Distinct new appearance
    bank_black = [{"embedding": vec_black, "viewpoint": "FRONT", "quality": 0.85} for _ in range(3)]

    # Associate Track 1 over 15 frames in Waiting Lobby
    new_id_threshold = int(am.config.get("reid_new_id_confirm_frames", 15))
    assigned_gid_lobby = None
    status_lobby = None
    for f in range(new_id_threshold):
        t_val = 36.15 + f * 0.1
        f_val = 900 + f
        # Person 4 remains active in pantry during this time
        jm.update_track_presence("pantry", 6, timestamp=t_val, frame_idx=f_val)

        assigned_gid_lobby, status_lobby, _, details = am.associate_track(
            camera_id="waiting_lobby",
            zone="waiting_lobby",
            local_track_id=1,
            bank_embeddings=bank_black,
            current_time=t_val,
            frame_idx=f_val,
            active_camera_gids=set()
        )

    # Track 1 must receive a new Global ID and MUST NOT be Person_004
    assert status_lobby == "MATCHED", f"Track 1 should be MATCHED with new ID, got {status_lobby}"
    assert assigned_gid_lobby is not None
    assert assigned_gid_lobby != gid_4, "Track 1 must NOT steal Person_004!"

    # 2. At t=41.33, Person_004 occludes in Pantry (BoT-SORT drops Track 6)
    jm.on_track_lost("pantry", "pantry", 6, timestamp=41.33, frame_idx=1029)
    assert jm.get_identity(gid_4).status == IdentityStatus.TEMPORARILY_LOST

    # 3. Person_004 reappears in Pantry as Track 13 at t=45.02 (reacquisition within same camera)
    bank_p4_reacq = [{"embedding": vec_p4, "viewpoint": "FRONT", "quality": 0.88} for _ in range(3)]
    for f in range(3):
        gid_reacq, status_reacq, _, _ = am.associate_track(
            camera_id="pantry",
            zone="pantry",
            local_track_id=13,
            bank_embeddings=bank_p4_reacq,
            current_time=45.02 + f * 0.1,
            frame_idx=1104 + f,
            active_camera_gids=set()
        )

    # Track 13 must successfully reacquire Person_004
    assert gid_reacq == gid_4, f"Track 13 must reacquire original Person_004, got {gid_reacq}"
    assert status_reacq == "MATCHED"
    assert jm.get_active_global_id("pantry", 13) == gid_4
    assert jm.get_identity(gid_4).status == IdentityStatus.ACTIVE



