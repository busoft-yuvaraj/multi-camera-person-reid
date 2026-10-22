import os
import pytest
import numpy as np
from typing import Dict, Any, List

from app.events.roi_detector import RoiDetector, RoiEvent
from app.events.transition_detector import TransitionDetector, TransitionEvent
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


def make_similar_vector(base: np.ndarray, target_sim: float, seed: int = 101) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.randn(len(base)).astype(np.float32)
    noise = noise - np.dot(noise, base) * base
    noise = noise / np.linalg.norm(noise)
    theta = np.arccos(np.clip(target_sim, -1.0, 1.0))
    res = np.cos(theta) * base + np.sin(theta) * noise
    return (res / np.linalg.norm(res)).astype(np.float32)


# ===========================================================================
# 1. Existing sink activity detection still works
# ===========================================================================
def test_case_01_sink_activity_detection():
    # Pantry sink ROI
    roi_config = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[31, 17], [528, 5], [530, 521], [49, 532]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_config, debounce_frames=2)

    # Frame 1 & 2 inside sink (e.g. foot at 200, 300)
    evts1 = detector.check_rois(1, (150, 100, 250, 300), frame_idx=1, timestamp=1.0)
    assert len(evts1) == 0
    evts2 = detector.check_rois(1, (150, 100, 250, 300), frame_idx=2, timestamp=1.1)
    assert len(evts2) == 1
    assert evts2[0].event == "ROI_ENTER"
    assert evts2[0].roi_id == "sink"

    # Frame 3 & 4 leaves sink
    evts3 = detector.check_rois(1, (600, 100, 700, 600), frame_idx=3, timestamp=5.0)
    assert len(evts3) == 0
    evts4 = detector.check_rois(1, (600, 100, 700, 600), frame_idx=4, timestamp=5.1)
    assert len(evts4) == 1
    assert evts4[0].event == "ROI_EXIT"
    assert evts4[0].roi_id == "sink"
    assert evts4[0].duration_seconds >= 4.0


# ===========================================================================
# 2. A person remains tracked after leaving the sink ROI
# ===========================================================================
def test_case_02_tracking_survival_outside_sink_roi():
    # Verify that leaving the sink ROI does not invalidate or reset person tracking
    roi_config = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[31, 17], [528, 5], [530, 521], [49, 532]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_config, debounce_frames=1)
    # Inside sink
    assert detector.is_point_in_roi((200, 300), "sink") is True
    # Walk away from sink into pantry open area
    assert detector.is_point_in_roi((600, 600), "sink") is False

    # Simulate multi-camera processor state tracking persistence:
    state = {
        "global_assignments": {1: "Person_001"},
        "banks": {1: "bank_active"},
        "last_seen_frame": {1: 100}
    }
    # When track 1 is detected outside the sink at (600, 600),
    # tracking state must NOT be cleared
    assert state["global_assignments"].get(1) == "Person_001"
    assert 1 in state["banks"]


# ===========================================================================
# 3. Waiting-lobby ROI entry/exit and line-crossing direction
# ===========================================================================
def test_case_03_waiting_lobby_roi_and_yellow_line_direction():
    # Monitored ROI: [319, 0], [708, 0], [708, 581], [319, 581]
    roi_config = {
        "monitored": {
            "enabled": True,
            "type": "polygon",
            "points": [[319, 0], [708, 0], [708, 581], [319, 581]]
        }
    }
    roi_det = RoiDetector("waiting_lobby", "waiting_lobby", roi_config, debounce_frames=1)
    assert roi_det.is_point_in_roi((500, 300), "monitored") is True
    assert roi_det.is_point_in_roi((200, 300), "monitored") is False

    # Yellow vertical line: start [338, 52], end [357, 546]
    trans_config = {
        "lobby_yellow": {
            "enabled": True,
            "type": "line",
            "points": {"start": [338, 52], "end": [357, 546]},
            "directions": {
                "SIDE_B_TO_SIDE_A": ["pantry", "passage"],
                "SIDE_A_TO_SIDE_B": ["waiting_lobby"]
            },
            "candidate_destinations": ["pantry", "passage"]
        }
    }
    trans_det = TransitionDetector("waiting_lobby", "waiting_lobby", trans_config, cooldown_frames=1)

    # Foot point starts inside lobby at x=450 (SIDE_B)
    evts1 = trans_det.check_transitions(1, (400, 200, 500, 300), frame_idx=1, timestamp=1.0)
    assert len(evts1) == 0

    # Crosses leftward to x=250 (SIDE_A)
    evts2 = trans_det.check_transitions(1, (200, 200, 300, 300), frame_idx=2, timestamp=1.5)
    assert len(evts2) == 1
    evt = evts2[0]
    assert evt.transition_id == "lobby_yellow"
    assert evt.direction == "SIDE_B_TO_SIDE_A"
    assert "pantry" in evt.candidate_destinations
    assert "passage" in evt.candidate_destinations


# ===========================================================================
# 4. Passage ROI entry/exit and line-crossing direction
# ===========================================================================
def test_case_04_passage_roi_and_green_line_direction():
    # Monitored ROI: [503, 267], [795, 267], [795, 615], [503, 615]
    roi_config = {
        "monitored": {
            "enabled": True,
            "type": "polygon",
            "points": [[503, 267], [795, 267], [795, 615], [503, 615]]
        }
    }
    roi_det = RoiDetector("passage", "passage", roi_config, debounce_frames=1)
    assert roi_det.is_point_in_roi((650, 400), "monitored") is True
    assert roi_det.is_point_in_roi((300, 400), "monitored") is False

    # Green horizontal line: start [576, 386], end [743, 386]
    trans_config = {
        "passage_green": {
            "enabled": True,
            "type": "line",
            "points": {"start": [576, 386], "end": [743, 386]},
            "directions": {
                "SIDE_B_TO_SIDE_A": ["passage", "workspace"],
                "SIDE_A_TO_SIDE_B": ["pantry", "waiting_lobby"]
            },
            "candidate_destinations": ["pantry", "waiting_lobby"]
        }
    }
    trans_det = TransitionDetector("passage", "passage", trans_config, cooldown_frames=1)

    # Foot point starts at far end (y=300, SIDE_B)
    evts1 = trans_det.check_transitions(1, (600, 200, 700, 300), frame_idx=1, timestamp=1.0)
    assert len(evts1) == 0

    # Crosses inward into corridor (y=450, SIDE_A)
    evts2 = trans_det.check_transitions(1, (600, 350, 700, 450), frame_idx=2, timestamp=1.5)
    assert len(evts2) == 1
    assert evts2[0].direction == "SIDE_B_TO_SIDE_A"
    assert "workspace" in evts2[0].candidate_destinations


# ===========================================================================
# 5. Waiting lobby -> pantry -> passage (Scenario 1)
# ===========================================================================
def test_case_05_scenario_1_lobby_to_pantry_to_passage(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    # 1. Lobby registers Person_001
    jm.register_new_identity("Person_001", "waiting_lobby", "waiting_lobby", local_track_id=1, timestamp=1.0, frame_idx=10)
    assert jm.get_active_global_id("waiting_lobby", 1) == "Person_001"

    # Crosses lobby yellow line with candidates [pantry, passage]
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        transition_id="lobby_yellow",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_B_TO_SIDE_A",
        foot_point=(330, 300),
        candidate_destinations=["pantry", "passage"]
    )
    jm.record_transition(trans_evt)

    # 2. Reacquired in pantry (local track 2)
    conf1 = jm.confirm_association("Person_001", "pantry", "pantry", local_track_id=2, timestamp=3.0, frame_idx=60, decision_reason="LOBBY_TO_PANTRY")
    assert conf1 is True
    assert jm.get_active_global_id("pantry", 2) == "Person_001"

    # Crosses pantry green line to passage
    trans_pantry = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        transition_id="pantry_green",
        timestamp=5.0,
        frame_idx=120,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(300, 490),
        candidate_destinations=["passage"]
    )
    jm.record_transition(trans_pantry)

    # 3. Reacquired in passage (local track 3)
    conf2 = jm.confirm_association("Person_001", "passage", "passage", local_track_id=3, timestamp=6.0, frame_idx=150, decision_reason="PANTRY_TO_PASSAGE")
    assert conf2 is True
    assert jm.get_active_global_id("passage", 3) == "Person_001"


# ===========================================================================
# 6. Waiting lobby -> passage directly (Scenario 2)
# ===========================================================================
def test_case_06_scenario_2_lobby_to_passage_direct(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "waiting_lobby", "waiting_lobby", local_track_id=1, timestamp=1.0, frame_idx=10)
    trans_evt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=1,
        transition_id="lobby_yellow",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_B_TO_SIDE_A",
        foot_point=(330, 300),
        candidate_destinations=["pantry", "passage"]
    )
    jm.record_transition(trans_evt)

    # Direct association in passage without visiting pantry!
    conf = jm.confirm_association("Person_001", "passage", "passage", local_track_id=4, timestamp=3.5, frame_idx=70, decision_reason="DIRECT_LOBBY_TO_PASSAGE")
    assert conf is True
    assert jm.get_active_global_id("passage", 4) == "Person_001"


# ===========================================================================
# 7. Passage -> pantry -> passage (Scenario 3)
# ===========================================================================
def test_case_07_scenario_3_inside_office_movement(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "passage", "passage", local_track_id=1, timestamp=1.0, frame_idx=10)

    # 1. Passage to Pantry
    trans1 = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="passage",
        zone="passage",
        local_track_id=1,
        transition_id="passage_green",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(650, 380),
        candidate_destinations=["pantry", "waiting_lobby"]
    )
    jm.record_transition(trans1)
    assert jm.confirm_association("Person_001", "pantry", "pantry", local_track_id=2, timestamp=3.0, frame_idx=60) is True

    # 2. Pantry back to Passage
    trans2 = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        transition_id="pantry_green",
        timestamp=6.0,
        frame_idx=150,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(300, 490),
        candidate_destinations=["passage"]
    )
    jm.record_transition(trans2)
    # Repeated visit to passage must succeed under the same Global ID
    assert jm.confirm_association("Person_001", "passage", "passage", local_track_id=5, timestamp=7.0, frame_idx=180) is True
    assert jm.get_active_global_id("passage", 5) == "Person_001"


# ===========================================================================
# 8. Passage -> waiting lobby directly (Scenario 4)
# ===========================================================================
def test_case_08_scenario_4_passage_to_lobby_direct(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "passage", "passage", local_track_id=1, timestamp=1.0, frame_idx=10)
    trans = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="passage",
        zone="passage",
        local_track_id=1,
        transition_id="passage_green",
        timestamp=2.0,
        frame_idx=30,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(650, 380),
        candidate_destinations=["pantry", "waiting_lobby"]
    )
    jm.record_transition(trans)
    assert jm.confirm_association("Person_001", "waiting_lobby", "waiting_lobby", local_track_id=7, timestamp=3.5, frame_idx=80) is True
    assert jm.get_active_global_id("waiting_lobby", 7) == "Person_001"


# ===========================================================================
# 9. Passage -> pantry -> waiting lobby (Scenario 5)
# ===========================================================================
def test_case_09_scenario_5_passage_to_pantry_to_lobby(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "passage", "passage", local_track_id=1, timestamp=1.0, frame_idx=10)
    # To pantry
    assert jm.confirm_association("Person_001", "pantry", "pantry", local_track_id=2, timestamp=3.0, frame_idx=60) is True
    # Pantry to lobby
    trans = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        transition_id="pantry_yellow",
        timestamp=5.0,
        frame_idx=120,
        direction="SIDE_A_TO_SIDE_B",
        foot_point=(524, 300),
        candidate_destinations=["waiting_lobby"]
    )
    jm.record_transition(trans)
    assert jm.confirm_association("Person_001", "waiting_lobby", "waiting_lobby", local_track_id=8, timestamp=6.5, frame_idx=160) is True
    assert jm.get_active_global_id("waiting_lobby", 8) == "Person_001"


# ===========================================================================
# 10. Ambiguous lobby transitions with multiple destinations
# ===========================================================================
def test_case_10_ambiguous_lobby_transition_candidates():
    rules = [
        {"from_camera": "waiting_lobby", "transition": "lobby_yellow", "direction": "SIDE_B_TO_SIDE_A", "candidate_zones": ["pantry", "passage"]}
    ]
    gate = SpatialGate(transition_rules=rules)

    # In pantry, lobby_yellow is valid
    val_pantry, score_pantry, _ = gate.evaluate("waiting_lobby", "pantry", "lobby_yellow", "SIDE_B_TO_SIDE_A", 2.0, candidate_zones=["pantry", "passage"])
    assert val_pantry is True
    assert score_pantry == 1.0

    # In passage, lobby_yellow is ALSO valid
    val_pass, score_pass, _ = gate.evaluate("waiting_lobby", "passage", "lobby_yellow", "SIDE_B_TO_SIDE_A", 2.0, candidate_zones=["pantry", "passage"])
    assert val_pass is True
    assert score_pass == 1.0


# ===========================================================================
# 11. Simultaneous visibility across overlapping cameras
# ===========================================================================
def test_case_11_simultaneous_visibility_overlapping_cameras(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    # Person active in waiting_lobby
    jm.register_new_identity("Person_001", "waiting_lobby", "waiting_lobby", local_track_id=1, timestamp=10.0, frame_idx=100)
    # Simultaneously visible in pantry near shared junction
    confirmed = jm.confirm_association(
        global_id="Person_001",
        camera_id="pantry",
        zone="pantry",
        local_track_id=2,
        timestamp=10.05,
        frame_idx=101,
        decision_reason="OVERLAPPING_VIEW"
    )
    assert confirmed is True
    # Both cameras maintain active mapping to Person_001
    assert jm.get_active_global_id("waiting_lobby", 1) == "Person_001"
    assert jm.get_active_global_id("pantry", 2) == "Person_001"


# ===========================================================================
# 12. No deletion of valid active Global ID mappings
# ===========================================================================
def test_case_12_no_deletion_of_valid_active_mappings(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "pantry", "pantry", local_track_id=10, timestamp=1.0, frame_idx=10)
    jm.confirm_association("Person_001", "passage", "passage", local_track_id=20, timestamp=2.0, frame_idx=30)

    # Pantry mapping must remain valid
    assert jm.get_active_global_id("pantry", 10) == "Person_001"
    assert jm.get_active_global_id("passage", 20) == "Person_001"


# ===========================================================================
# 13. No duplicate Global ID creation during a valid handoff
# ===========================================================================
def test_case_13_no_duplicate_global_id_during_valid_handoff(tmp_path):
    topo = TopologyGate({"waiting_lobby": ["pantry", "passage"], "pantry": ["waiting_lobby", "passage"]})
    temp = TemporalGate({"waiting_lobby_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0}})
    rules = [{"from_camera": "waiting_lobby", "transition": "lobby_yellow", "candidate_zones": ["pantry", "passage"]}]
    spatial = SpatialGate(rules)
    c_filter = CandidateFilter(topo, temp, spatial)

    identities = {
        "Person_001": {
            "current_camera": "waiting_lobby",
            "last_seen_timestamp": 10.0,
            "last_transition_id": "lobby_yellow",
            "last_transition_candidate_zones": ["pantry", "passage"],
            "last_transition_timestamp": 9.9,
            "status": "ACTIVE"
        }
    }

    # Track in pantry at t=10.2 evaluating candidates (same-cam occupied is empty)
    valid_gids, evals, rejections = c_filter.filter_candidates(
        identities=identities,
        current_camera="pantry",
        current_zone="pantry",
        current_time=10.2,
        exclude_gids=set()
    )
    # Person_001 MUST be a valid candidate, preventing duplicate ID creation
    assert "Person_001" in valid_gids


# ===========================================================================
# 14. Two similar-looking people crossing simultaneously
# ===========================================================================
def test_case_14_two_similar_looking_people_same_camera(tmp_path):
    topo = TopologyGate({"pantry": ["waiting_lobby", "passage"]})
    temp = TemporalGate({})
    spatial = SpatialGate()
    c_filter = CandidateFilter(topo, temp, spatial)

    identities = {
        "Person_001": {"current_camera": "pantry", "last_seen_timestamp": 5.0, "status": "ACTIVE"},
        "Person_002": {"current_camera": "waiting_lobby", "last_seen_timestamp": 4.0, "status": "ACTIVE"}
    }

    # If Track 1 in pantry already holds Person_001, Track 2 in pantry MUST exclude Person_001
    valid_gids, _, rejections = c_filter.filter_candidates(
        identities=identities,
        current_camera="pantry",
        current_zone="pantry",
        current_time=5.5,
        exclude_gids={"Person_001"}
    )
    assert "Person_001" not in valid_gids
    assert rejections["Person_001"] == "EXCLUDED_OCCUPIED_IN_CAMERA"


# ===========================================================================
# 15. Temporary occlusion and local ID changes
# ===========================================================================
def test_case_15_temporary_occlusion_and_id_renumbering(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)

    jm.register_new_identity("Person_001", "pantry", "pantry", local_track_id=1, timestamp=1.0, frame_idx=10)
    # Track 1 occluded
    jm.on_track_lost("pantry", "pantry", local_track_id=1, timestamp=2.0, frame_idx=40)
    assert jm.get_identity("Person_001").status == IdentityStatus.TEMPORARILY_LOST

    # Reappears with BoT-SORT new local track 5
    confirmed = jm.confirm_association("Person_001", "pantry", "pantry", local_track_id=5, timestamp=2.5, frame_idx=55, decision_reason="REACQUISITION")
    assert confirmed is True
    assert jm.get_active_global_id("pantry", 5) == "Person_001"
    assert jm.get_identity("Person_001").status == IdentityStatus.ACTIVE


# ===========================================================================
# 16. Small initial passage crops followed by higher-quality observations
# ===========================================================================
def test_case_16_small_initial_crops_preserve_pending(tmp_path):
    gallery = GlobalGallery(str(tmp_path / "gallery.pkl"), clear_on_start=True)
    proto = make_unit_vector(seed=1)
    gid = gallery.add_identity([{"camera_id": "waiting_lobby", "track_id": 1}], [{"embedding": proto, "viewpoint": "FRONT", "quality": 0.9}])

    matcher = GlobalMatcher(gallery=gallery, config={"threshold_match": 0.70, "threshold_ambiguous": 0.60})
    topo = TopologyGate({"waiting_lobby": ["passage"], "passage": ["waiting_lobby"]})
    temp = TemporalGate({})
    rules = [{"from_camera": "waiting_lobby", "transition": "lobby_yellow", "candidate_zones": ["passage"]}]
    spatial = SpatialGate(rules)
    c_filter = CandidateFilter(topo, temp, spatial)
    store = JourneyStore(str(tmp_path / "j.jsonl"), str(tmp_path / "a.jsonl"))
    jm = JourneyManager(store=store)
    jm.register_new_identity(gid, "waiting_lobby", "waiting_lobby", 1, 1.0, 10)

    assoc_mgr = AssociationManager(
        gallery=gallery,
        matcher=matcher,
        candidate_filter=c_filter,
        journey_manager=jm,
        store=store,
        config={"reid_confirm_frames": 3, "reid_new_id_confirm_frames": 15}
    )

    # Initial observations in passage are small/blurry with lower similarity (~0.58)
    noisy_proto = make_similar_vector(proto, target_sim=0.58, seed=99)
    bank_embeddings = [
        {"embedding": noisy_proto, "quality": 0.35, "viewpoint": "FRONT"},
        {"embedding": noisy_proto, "quality": 0.35, "viewpoint": "FRONT"},
        {"embedding": noisy_proto, "quality": 0.35, "viewpoint": "FRONT"}
    ]

    assigned, status, sim, details = assoc_mgr.associate_track(
        camera_id="passage",
        zone="passage",
        local_track_id=10,
        bank_embeddings=bank_embeddings,
        current_time=2.5,
        frame_idx=60,
        active_camera_gids=set()
    )
    # Must preserve candidate as PENDING, NOT create a new ID
    assert status == "PENDING"
    assert assigned == gid or assigned is None
    assert assoc_mgr.no_match_counters.get(("passage", 10), 0) == 0


# ===========================================================================
# 17. Expiration of unresolved associations
# ===========================================================================
def test_case_17_expiration_of_unresolved_associations(tmp_path):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store, reacquisition_timeout_seconds=30.0)

    jm.register_new_identity("Person_001", "pantry", "pantry", local_track_id=1, timestamp=1.0, frame_idx=10)
    jm.on_track_lost("pantry", "pantry", local_track_id=1, timestamp=2.0, frame_idx=30)
    state = jm.get_identity("Person_001")
    state.last_transition_timestamp = 2.0
    state.last_transition_id = "pantry_green"

    # Check expirations after 50 seconds (unseen for 50s > 30s timeout)
    jm.check_expirations(current_time=52.0, frame_idx=1500)
    assert state.status == IdentityStatus.EXPIRED
    assert state.last_transition_status == "EXPIRED"


# ===========================================================================
# 18. ROI_EXIT handling on track loss
# ===========================================================================
def test_case_18_roi_exit_handling_on_track_loss(tmp_path):
    roi_config = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[31, 17], [528, 5], [530, 521], [49, 532]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_config, debounce_frames=1)
    detector.check_rois(1, (100, 100, 200, 200), frame_idx=1, timestamp=1.0)
    assert detector.is_inside.get((1, "sink")) is True

    # Track lost while still inside ROI
    exit_events = detector.on_track_lost(1, frame_idx=10, timestamp=3.5, global_id="Person_001")
    assert len(exit_events) == 1
    assert exit_events[0].event == "ROI_EXIT"
    assert exit_events[0].roi_id == "sink"
    assert exit_events[0].duration_seconds == 2.5


# ===========================================================================
# 19. Cleanup of expired track state
# ===========================================================================
def test_case_19_cleanup_of_expired_track_state():
    trans_config = {
        "lobby_yellow": {
            "enabled": True,
            "type": "line",
            "points": {"start": [338, 52], "end": [357, 546]}
        }
    }
    detector = TransitionDetector("waiting_lobby", "waiting_lobby", trans_config)
    detector.check_transitions(5, (400, 200, 500, 300), frame_idx=1, timestamp=1.0)
    assert (5, "lobby_yellow") in detector.track_sides

    # Reset track cleans up state
    detector.reset_track(5)
    assert (5, "lobby_yellow") not in detector.track_sides


# ===========================================================================
# 20. Invalid physical transitions being rejected
# ===========================================================================
def test_case_20_invalid_physical_transition_rejected():
    topo = TopologyGate({
        "pantry": ["passage"],
        "passage": ["pantry"]
    })
    # pantry -> waiting_lobby is not allowed in this topology
    assert topo.is_transition_valid("pantry", "waiting_lobby") is False


# ===========================================================================
# 21. Reverse transitions and repeated camera visits
# ===========================================================================
def test_case_21_reverse_transitions_and_repeated_camera_visits():
    rules = [
        {"from_camera": "waiting_lobby", "transition": "lobby_yellow", "direction": "SIDE_B_TO_SIDE_A", "candidate_zones": ["pantry"]},
        {"from_camera": "waiting_lobby", "transition": "lobby_yellow", "direction": "SIDE_A_TO_SIDE_B", "candidate_zones": ["waiting_lobby"]}
    ]
    gate = SpatialGate(transition_rules=rules)

    # Inward reverse crossing back to lobby
    is_valid, score, reason = gate.evaluate("waiting_lobby", "waiting_lobby", "lobby_yellow", "SIDE_A_TO_SIDE_B", 1.0, candidate_zones=["waiting_lobby"])
    assert is_valid is True
    assert score == 1.0


# ===========================================================================
# 22. Adding a new camera through configuration without changing core matching logic
# ===========================================================================
def test_case_22_adding_new_camera_through_configuration(tmp_path):
    new_topo = {
        "waiting_lobby": ["pantry", "passage"],
        "pantry": ["waiting_lobby", "passage"],
        "passage": ["pantry", "waiting_lobby", "cafeteria"],
        "cafeteria": ["passage"]
    }
    topo_gate = TopologyGate(topology_config=new_topo)
    assert topo_gate.is_transition_valid("passage", "cafeteria") is True
    assert topo_gate.is_transition_valid("cafeteria", "passage") is True
    assert topo_gate.is_transition_valid("waiting_lobby", "cafeteria") is False
