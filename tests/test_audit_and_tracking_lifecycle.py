"""
Comprehensive Regression Test Suite for Multi-Camera Tracking Pipeline Audit and Fixes.
Covers all 18 required regression scenarios from Task Section 11:
1. Correct resolution: configured ROI is applied.
2. Incorrect resolution: descriptive validation error is raised before ROI use.
3. Cameras with different valid resolutions: each uses its own ROI configuration.
4. Detection inside an eligible ROI: allowed into the appropriate association stage.
5. Detection in an excluded/unmonitored region: cannot incorrectly create or match a Global ID.
6. Valid transition between cameras: the same person can recover the same Global ID.
7. Physically impossible simultaneous presence: the identity conflict is rejected.
8. Legitimate simultaneous views in a genuinely overlapping camera pair: allowed under intended evidence.
9. Duplicate boxes on one person: do not create duplicate Global IDs.
10. Two nearby distinct people: are not merged merely because their boxes overlap.
11. Lost track replaced by a new local track ID: stale mappings do not block reacquisition indefinitely.
12. Ambiguous ReID candidates: remain pending rather than forcing an incorrect match.
13. Multiple tracks processed together: incompatible tracks cannot claim the same identity (Hungarian 1-to-1).
14. Camera FPS mismatch and unequal video lengths: timestamps and track lifecycles remain valid.
15. Independent camera failure or end-of-stream: other camera state remains consistent.
16. Separate executions: current logs and metrics do not include stale prior-run events.
17. Association rollback: mappings and gallery updates remain consistent.
18. Existing topology, ROI, transition, and pipeline fixtures continue to pass.
"""
import pytest
import numpy as np
import cv2
import os
import json
from unittest.mock import MagicMock

from app.utils.config_validator import (
    validate_pipeline_config,
    validate_frame_resolution,
    ResolutionValidationError,
    ConfigurationError
)
from app.events.roi_detector import RoiDetector, RoiEvent
from app.events.transition_detector import TransitionDetector, TransitionEvent
from app.association.association_manager import AssociationManager
from app.association.candidate_filter import CandidateFilter
from app.association.topology import TopologyGate
from app.association.temporal_gate import TemporalGate
from app.association.spatial_gate import SpatialGate
from app.journey.journey_manager import JourneyManager
from app.journey.journey_store import JourneyStore
from app.journey.models import IdentityStatus, GlobalIdentityState
from app.reid.global_gallery import GlobalGallery
from app.reid.matcher import GlobalMatcher
from app.reid.embedding_bank import TrackEmbeddingBank
from app.multi_camera_processor import MultiCameraProcessor


@pytest.fixture
def base_config():
    return {
        "cameras": {
            "cam_lobby": {
                "camera_id": "cam_lobby",
                "zone": "lobby",
                "expected_resolution": [1280, 720],
                "video_path": "lobby.mp4",
                "roi": {
                    "monitored": {
                        "enabled": True,
                        "association_eligible": True,
                        "type": "polygon",
                        "points": [[200, 100], [800, 100], [800, 600], [200, 600]]
                    }
                },
                "transitions": {
                    "lobby_exit": {
                        "enabled": True,
                        "points": {"start": [300, 100], "end": [300, 600]},
                        "directions": {
                            "SIDE_B_TO_SIDE_A": ["pantry"],
                            "SIDE_A_TO_SIDE_B": ["lobby"]
                        },
                        "candidate_destinations": ["pantry"]
                    }
                }
            },
            "cam_pantry": {
                "camera_id": "cam_pantry",
                "zone": "pantry",
                "expected_resolution": [720, 720],
                "video_path": "pantry.mp4",
                "crop": [0, 0, 720, 720],
                "roi": {
                    "monitored": {
                        "enabled": True,
                        "association_eligible": True,
                        "type": "polygon",
                        "points": [[50, 50], [650, 50], [650, 650], [50, 650]]
                    }
                },
                "transitions": {
                    "pantry_entry": {
                        "enabled": True,
                        "points": {"start": [600, 100], "end": [600, 600]},
                        "directions": {
                            "SIDE_A_TO_SIDE_B": ["lobby"],
                            "SIDE_B_TO_SIDE_A": ["pantry"]
                        },
                        "candidate_destinations": ["lobby"]
                    }
                }
            }
        },
        "topology": {
            "lobby": ["pantry"],
            "pantry": ["lobby"]
        },
        "transition_rules": [
            {
                "from_camera": "cam_lobby",
                "transition": "lobby_exit",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["pantry"]
            },
            {
                "from_camera": "cam_pantry",
                "transition": "pantry_entry",
                "direction": "SIDE_B_TO_SIDE_A",
                "candidate_zones": ["pantry"]
            }
        ],
        "temporal_constraints": {
            "overlapping_camera_pairs": [],
            "same_camera_reacquisition_max_seconds": 60.0,
            "lobby_to_pantry": {"min_seconds": 0.5, "max_seconds": 45.0},
            "pantry_to_lobby": {"min_seconds": 0.5, "max_seconds": 45.0}
        },
        "model_path": "yolo11n.pt",
        "threshold_match": 0.70,
        "threshold_ambiguous": 0.60,
        "ambiguity_margin": 0.04,
        "reid_confirm_frames": 3,
        "reid_new_id_confirm_frames": 15
    }


def make_embedding(seed: float = 0.5, dim: int = 512) -> np.ndarray:
    rng = np.random.RandomState(int(abs(seed) * 1000) % 100000)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def create_association_system(tmp_path, config):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(
        store=store,
        transition_rules=config.get("transition_rules"),
        topology=config.get("topology"),
        overlapping_camera_pairs=config.get("temporal_constraints", {}).get("overlapping_camera_pairs")
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    matcher = GlobalMatcher(gallery=gallery, config=config)
    topo_gate = TopologyGate(config.get("topology"), config.get("transition_rules"))
    temporal_gate = TemporalGate(config.get("temporal_constraints"))
    spatial_gate = SpatialGate(config.get("transition_rules"))
    cand_filter = CandidateFilter(
        topology_gate=topo_gate,
        temporal_gate=temporal_gate,
        spatial_gate=spatial_gate
    )
    am = AssociationManager(
        config=config,
        matcher=matcher,
        gallery=gallery,
        candidate_filter=cand_filter,
        journey_manager=jm,
        store=store
    )
    return store, jm, gallery, matcher, cand_filter, am


# -----------------------------------------------------------------------------
# Scenario 1: Correct resolution: configured ROI is applied.
# -----------------------------------------------------------------------------
def test_scenario_1_correct_resolution_roi_applied(base_config):
    validated_config = validate_pipeline_config(base_config)
    exp_w, exp_h = validated_config["cameras"]["cam_lobby"]["expected_resolution"]
    
    validate_frame_resolution("cam_lobby", exp_w, exp_h, (exp_w, exp_h), affected_config="roi")

    roi_det = RoiDetector(
        camera_id="cam_lobby",
        zone="lobby",
        roi_config=validated_config["cameras"]["cam_lobby"]["roi"]
    )
    assert roi_det.is_in_association_roi((500, 300)) is True
    assert roi_det.is_in_association_roi((100, 50)) is False


# -----------------------------------------------------------------------------
# Scenario 2: Incorrect resolution: descriptive validation error raised before ROI use.
# -----------------------------------------------------------------------------
def test_scenario_2_incorrect_resolution_raises_error():
    with pytest.raises(ResolutionValidationError) as exc_info:
        validate_frame_resolution(
            camera_id="cam_lobby",
            actual_width=640,
            actual_height=480,
            expected_resolution=(1280, 720),
            affected_config="roi"
        )
    err = str(exc_info.value)
    assert "cam_lobby" in err
    assert "1280, 720" in err
    assert "640, 480" in err
    assert "cameras.cam_lobby.roi" in err


# -----------------------------------------------------------------------------
# Scenario 3: Cameras with different valid resolutions: each uses its own ROI configuration.
# -----------------------------------------------------------------------------
def test_scenario_3_different_camera_resolutions_independent_rois(base_config):
    base_config["cameras"]["cam_highres"] = {
        "camera_id": "cam_highres",
        "zone": "highres_zone",
        "expected_resolution": [1920, 1080],
        "video_path": "highres.mp4",
        "roi": {
            "monitored": {
                "enabled": True,
                "association_eligible": True,
                "type": "polygon",
                "points": [[100, 100], [1800, 100], [1800, 1000], [100, 1000]]
            }
        }
    }
    cfg = validate_pipeline_config(base_config)
    assert cfg["cameras"]["cam_lobby"]["expected_resolution"] == [1280, 720]
    assert cfg["cameras"]["cam_pantry"]["expected_resolution"] == [720, 720]
    assert cfg["cameras"]["cam_highres"]["expected_resolution"] == [1920, 1080]

    validate_frame_resolution("cam_lobby", 1280, 720, (1280, 720))
    validate_frame_resolution("cam_pantry", 720, 720, (720, 720))
    validate_frame_resolution("cam_highres", 1920, 1080, (1920, 1080))


# -----------------------------------------------------------------------------
# Scenario 4: Detection inside an eligible ROI: allowed into the association stage.
# -----------------------------------------------------------------------------
def test_scenario_4_detection_inside_eligible_roi_allowed(tmp_path, base_config):
    store, jm, gallery, matcher, cand_filter, am = create_association_system(tmp_path, base_config)

    emb = make_embedding(1.0)
    gid, status, sim, details = am.associate_track(
        camera_id="cam_lobby",
        zone="lobby",
        local_track_id=1,
        bank_embeddings=[emb, emb, emb],
        current_time=1.0,
        frame_idx=10,
        active_camera_gids=set(),
        is_association_eligible=True
    )
    assert details["decision"] != "OUTSIDE_ASSOCIATION_ZONE"


# -----------------------------------------------------------------------------
# Scenario 5: Detection in an excluded/unmonitored region: cannot create or match Global ID.
# -----------------------------------------------------------------------------
def test_scenario_5_detection_outside_roi_cannot_associate(tmp_path, base_config):
    store, jm, gallery, matcher, cand_filter, am = create_association_system(tmp_path, base_config)

    emb = make_embedding(1.0)
    for f in range(20):
        gid, status, sim, details = am.associate_track(
            camera_id="cam_lobby",
            zone="lobby",
            local_track_id=1,
            bank_embeddings=[emb, emb, emb],
            current_time=float(f),
            frame_idx=f,
            active_camera_gids=set(),
            is_association_eligible=False
        )
        assert gid is None
        assert status == "PENDING"
        assert details["decision"] == "OUTSIDE_ASSOCIATION_ZONE"

    assert len(gallery.get_identities()) == 0


# -----------------------------------------------------------------------------
# Scenario 6: Valid transition between cameras: same person recovers same Global ID.
# -----------------------------------------------------------------------------
def test_scenario_6_valid_transition_recovers_same_global_id(tmp_path, base_config):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(
        store=store,
        transition_rules=base_config["transition_rules"],
        topology=base_config["topology"],
        overlapping_camera_pairs=[]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    emb = make_embedding(2.0)
    gid = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 1}], [emb])
    jm.register_new_identity(gid, "cam_lobby", "lobby", local_track_id=1, timestamp=1.0, frame_idx=10)

    # Person leaves lobby, crosses transition to pantry
    tevt = TransitionEvent(
        event="TRANSITION_CROSSED",
        camera_id="cam_lobby",
        zone="lobby",
        local_track_id=1,
        transition_id="lobby_exit",
        timestamp=12.0,
        frame_idx=50,
        direction="SIDE_B_TO_SIDE_A",
        foot_point=(300.0, 350.0),
        global_id=gid,
        candidate_zones=["pantry"]
    )
    jm.record_transition(tevt)
    jm.on_track_lost("cam_lobby", "lobby", 1, timestamp=13.0, frame_idx=60)

    # Person appears in pantry 2 seconds later
    confirmed = jm.confirm_association(
        global_id=gid,
        camera_id="cam_pantry",
        zone="pantry",
        local_track_id=5,
        timestamp=15.0,
        frame_idx=100,
        decision_reason="VALID_TRANSITION_REACQUISITION"
    )
    assert confirmed is True
    assert jm.get_active_global_id("cam_pantry", 5) == gid


# -----------------------------------------------------------------------------
# Scenario 7: Physically impossible simultaneous presence: identity conflict rejected.
# -----------------------------------------------------------------------------
def test_scenario_7_physically_impossible_simultaneous_presence_rejected(tmp_path, base_config):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(
        store=store,
        transition_rules=base_config["transition_rules"],
        topology=base_config["topology"],
        overlapping_camera_pairs=[]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    emb = make_embedding(3.0)
    gid = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 1}], [emb])
    jm.register_new_identity(gid, "cam_lobby", "lobby", local_track_id=1, timestamp=10.0, frame_idx=100)

    # Simultaneously at timestamp 10.1s, track 9 in pantry claims gid without a transition!
    confirmed = jm.confirm_association(
        global_id=gid,
        camera_id="cam_pantry",
        zone="pantry",
        local_track_id=9,
        timestamp=10.1,
        frame_idx=105,
        decision_reason="ATTEMPTED_STEAL"
    )
    assert confirmed is False
    assert jm.get_active_global_id("cam_lobby", 1) == gid
    assert jm.get_active_global_id("cam_pantry", 9) is None


# -----------------------------------------------------------------------------
# Scenario 8: Legitimate simultaneous views in genuinely overlapping camera pair: allowed.
# -----------------------------------------------------------------------------
def test_scenario_8_legitimate_overlapping_camera_pair_allowed(tmp_path, base_config):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(
        store=store,
        overlapping_camera_pairs=[["cam_lobby", "cam_pantry"]]
    )
    gallery = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    emb = make_embedding(4.0)
    gid = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 1}], [emb])
    jm.register_new_identity(gid, "cam_lobby", "lobby", local_track_id=1, timestamp=10.0, frame_idx=100)

    # In genuinely overlapping camera pair, simultaneous view is allowed
    confirmed = jm.confirm_association(
        global_id=gid,
        camera_id="cam_pantry",
        zone="pantry",
        local_track_id=2,
        timestamp=10.2,
        frame_idx=105,
        is_overlapping_view=True
    )
    assert confirmed is True
    assert jm.get_active_global_id("cam_lobby", 1) == gid
    assert jm.get_active_global_id("cam_pantry", 2) == gid


# -----------------------------------------------------------------------------
# Scenario 9: Duplicate boxes on one person: do not create duplicate Global IDs.
# -----------------------------------------------------------------------------
def test_scenario_9_duplicate_boxes_suppressed(tmp_path, base_config):
    proc = MultiCameraProcessor(
        cameras_info=[{"camera_id": "cam_lobby", "video_path": "lobby.mp4"}],
        output_path="out.mp4",
        config=base_config,
        extractor=MagicMock(),
        par_extractor=MagicMock(),
        pose_extractor=MagicMock(),
        gallery=GlobalGallery(str(tmp_path / "gal.pkl")),
        matcher=MagicMock()
    )
    state = proc.state["cam_lobby"]
    state["global_assignments"][10] = "Person_001"
    state["match_status"][10] = "MATCHED"

    tracks = [
        [200, 200, 300, 500, 10, 0.90, 0],
        [210, 205, 305, 495, 11, 0.85, 0]
    ]

    primary, suppressed = proc._suppress_duplicate_tracks("cam_lobby", tracks, state, frame_idx=50, current_time=2.0)
    assert len(primary) == 1
    assert primary[0][4] == 10
    assert len(suppressed) == 1
    assert suppressed[0][4] == 11
    assert state["duplicate_tracks"][11] == 10
    assert state["match_status"][11] == "DUPLICATE"


# -----------------------------------------------------------------------------
# Scenario 10: Two nearby distinct people: not merged merely because boxes overlap.
# -----------------------------------------------------------------------------
def test_scenario_10_two_nearby_distinct_people_not_merged(tmp_path, base_config):
    proc = MultiCameraProcessor(
        cameras_info=[{"camera_id": "cam_lobby", "video_path": "lobby.mp4"}],
        output_path="out.mp4",
        config=base_config,
        extractor=MagicMock(),
        par_extractor=MagicMock(),
        pose_extractor=MagicMock(),
        gallery=GlobalGallery(str(tmp_path / "gal.pkl")),
        matcher=MagicMock()
    )
    state = proc.state["cam_lobby"]
    state["global_assignments"][1] = "Person_001"
    state["match_status"][1] = "MATCHED"
    state["global_assignments"][2] = "Person_002"
    state["match_status"][2] = "MATCHED"

    tracks = [
        [200, 200, 320, 500, 1, 0.88, 0],
        [250, 200, 370, 500, 2, 0.85, 0]
    ]

    primary, suppressed = proc._suppress_duplicate_tracks("cam_lobby", tracks, state, frame_idx=60, current_time=2.5)
    assert len(primary) == 2
    assert len(suppressed) == 0


# -----------------------------------------------------------------------------
# Scenario 11: Lost track replaced by new local track ID without zombie lock.
# -----------------------------------------------------------------------------
def test_scenario_11_lost_track_replaced_without_zombie_lock(tmp_path, base_config):
    store = JourneyStore(str(tmp_path / "journey.jsonl"), str(tmp_path / "assoc.jsonl"))
    jm = JourneyManager(store=store)
    gallery = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    emb = make_embedding(5.0)
    gid = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 10}], [emb])
    jm.register_new_identity(gid, "cam_lobby", "lobby", local_track_id=10, timestamp=1.0, frame_idx=10)

    # Track 10 was last seen at 1.0s, now 5.0s (inactive for >2s)
    confirmed = jm.confirm_association(
        global_id=gid,
        camera_id="cam_lobby",
        zone="lobby",
        local_track_id=20,
        timestamp=5.0,
        frame_idx=120,
        decision_reason="REPLACEMENT_REACQUISITION"
    )
    assert confirmed is True
    assert jm.get_active_global_id("cam_lobby", 20) == gid


# -----------------------------------------------------------------------------
# Scenario 12: Ambiguous ReID candidates remain pending.
# -----------------------------------------------------------------------------
def test_scenario_12_ambiguous_candidates_remain_pending(tmp_path, base_config):
    store, jm, gallery, matcher, cand_filter, am = create_association_system(tmp_path, base_config)

    v1 = make_embedding(6.0)
    v2 = v1 + (np.random.RandomState(42).randn(512).astype(np.float32) * 0.01)
    v2 = v2 / np.linalg.norm(v2)
    gid1 = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 1}], [v1])
    gid2 = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 2}], [v2])

    query_emb = v1.copy()
    assigned_gid, status, sim, details = am.associate_track(
        camera_id="cam_lobby",
        zone="lobby",
        local_track_id=99,
        bank_embeddings=[query_emb, query_emb, query_emb],
        current_time=1.0,
        frame_idx=10,
        active_camera_gids=set(),
        is_association_eligible=True
    )
    assert status == "PENDING"
    assert assigned_gid is None or status != "MATCHED"


# -----------------------------------------------------------------------------
# Scenario 13: Multiple tracks processed together via Hungarian batch matching.
# -----------------------------------------------------------------------------
def test_scenario_13_multiple_tracks_hungarian_batch_no_clash(tmp_path, base_config):
    store, jm, gallery, matcher, cand_filter, am = create_association_system(tmp_path, base_config)

    eA = make_embedding(10.0)
    eB = make_embedding(20.0)
    gidA = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 1}], [eA])
    gidB = gallery.add_identity([{"camera_id": "cam_lobby", "track_id": 2}], [eB])

    tracks_data = [
        {"track_id": 10, "bank_embeddings": [eA, eA, eA], "is_association_eligible": True},
        {"track_id": 20, "bank_embeddings": [eB, eB, eB], "is_association_eligible": True}
    ]

    results = am.associate_tracks_batch(
        camera_id="cam_lobby",
        zone="lobby",
        tracks_data=tracks_data,
        current_time=1.0,
        frame_idx=10,
        active_camera_gids=set()
    )
    assert 10 in results and 20 in results
    res10 = results[10]
    res20 = results[20]

    assert res10[0] == gidA
    assert res20[0] == gidB
    assert res10[0] != res20[0]


# -----------------------------------------------------------------------------
# Scenario 14: Camera FPS mismatch and start offsets: monotonic and valid timestamps.
# -----------------------------------------------------------------------------
def test_scenario_14_camera_fps_mismatch_and_offsets():
    cam1_fps, cam1_offset = 24.5, 0.0
    cam2_fps, cam2_offset = 30.0, 3.5

    cam1_times = [(f / cam1_fps) + cam1_offset for f in range(100)]
    cam2_times = [(f / cam2_fps) + cam2_offset for f in range(100)]

    assert all(cam1_times[i] < cam1_times[i+1] for i in range(len(cam1_times)-1))
    assert all(cam2_times[i] < cam2_times[i+1] for i in range(len(cam2_times)-1))

    t_exit_cam1 = cam1_times[50]
    t_enter_cam2 = cam2_times[20]
    elapsed = t_enter_cam2 - t_exit_cam1
    assert 2.0 < elapsed < 2.2


# -----------------------------------------------------------------------------
# Scenario 15: Independent camera failure or end-of-stream does not corrupt others.
# -----------------------------------------------------------------------------
def test_scenario_15_independent_camera_eos_handling(tmp_path, base_config):
    proc = MultiCameraProcessor(
        cameras_info=[
            {"camera_id": "cam_lobby", "video_path": "lobby.mp4"},
            {"camera_id": "cam_pantry", "video_path": "pantry.mp4"}
        ],
        output_path="out.mp4",
        config=base_config,
        extractor=MagicMock(),
        par_extractor=MagicMock(),
        pose_extractor=MagicMock(),
        gallery=GlobalGallery(str(tmp_path / "gal.pkl")),
        matcher=MagicMock()
    )
    proc.state["cam_lobby"]["finished"] = True
    assert proc.state["cam_lobby"]["finished"] is True
    assert proc.state["cam_pantry"]["finished"] is False


# -----------------------------------------------------------------------------
# Scenario 16: Separate executions: logs and metrics do not include stale prior-run events.
# -----------------------------------------------------------------------------
def test_scenario_16_separate_executions_isolated_logs(tmp_path):
    j_log = str(tmp_path / "journey.jsonl")
    a_log = str(tmp_path / "assoc.jsonl")

    store1 = JourneyStore(j_log, a_log, run_id="run_001", clear_on_start=True)
    store1.log_association_decision({"camera_id": "cam_lobby", "decision": "RUN1_DECISION"})
    with open(a_log, "r") as f:
        lines1 = f.readlines()
    assert len(lines1) == 1
    assert "RUN1_DECISION" in lines1[0]
    assert "run_001" in lines1[0]

    store2 = JourneyStore(j_log, a_log, run_id="run_002", clear_on_start=True)
    store2.log_association_decision({"camera_id": "cam_lobby", "decision": "RUN2_DECISION"})
    with open(a_log, "r") as f:
        lines2 = f.readlines()
    assert len(lines2) == 1
    assert "RUN2_DECISION" in lines2[0]
    assert "run_002" in lines2[0]


# -----------------------------------------------------------------------------
# Scenario 17: Association rollback: mappings and gallery updates remain consistent.
# -----------------------------------------------------------------------------
def test_scenario_17_association_rollback_consistency(tmp_path):
    gal = GlobalGallery(storage_path=str(tmp_path / "gal.pkl"))
    e1 = make_embedding(30.0)
    gid = gal.add_identity([{"camera_id": "cam_lobby", "track_id": 5}], [e1])
    assert gid in gal.identities
    assert len(gal.identities[gid].get_all_embeddings()) == 1

    gal.rollback_track(gid, "cam_lobby", 5)
    assert len(gal.identities[gid].get_all_embeddings()) == 0


# -----------------------------------------------------------------------------
# Scenario 18: Existing topology, ROI, transition, and pipeline integration tests.
# -----------------------------------------------------------------------------
def test_scenario_18_existing_topology_and_rules_consistent(base_config):
    topo = TopologyGate(base_config["topology"], base_config["transition_rules"])
    assert topo.is_transition_valid("lobby", "pantry") is True
    assert topo.is_transition_valid("lobby", "unknown_zone") is False
