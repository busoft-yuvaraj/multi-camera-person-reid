import pytest
import numpy as np
from app.events.transition_detector import TransitionDetector, TransitionEvent
from app.events.roi_detector import RoiDetector, RoiEvent
from app.association.topology import TopologyGate
from app.association.temporal_gate import TemporalGate
from app.association.spatial_gate import SpatialGate
from app.association.candidate_filter import CandidateFilter
from app.journey.models import IdentityStatus, JourneyEvent, GlobalIdentityState
from app.journey.journey_store import JourneyStore
from app.journey.journey_manager import JourneyManager
from app.reid.global_gallery import GlobalGallery

# -------------------------------------------------------------
# 1. TRANSITION DETECTOR TESTS
# -------------------------------------------------------------
def test_transition_detector_crossing():
    transitions_cfg = {
        "pantry_yellow": {
            "enabled": True,
            "type": "line",
            "points": {
                "start": [500, 100],
                "end": [500, 500]
            },
            "target_camera": "waiting_lobby",
            "target_zone": "waiting_lobby"
        }
    }
    detector = TransitionDetector("pantry", "pantry", transitions_cfg, cooldown_frames=5)
    
    # Track moves from left of line (x=450) to right of line (x=550)
    # Foot point at bottom center: bbox [x1, y1, x2, y2]
    # At t=0, foot_point is (450, 300) -> left
    events = detector.check_transitions(
        local_track_id=1,
        bbox=(400, 200, 500, 400), # foot_point = (450, 400)
        frame_idx=1,
        timestamp=1.0
    )
    assert len(events) == 0

    # At t=1, foot_point is (550, 400) -> crossed line at x=500
    events = detector.check_transitions(
        local_track_id=1,
        bbox=(500, 200, 600, 400), # foot_point = (550, 400)
        frame_idx=2,
        timestamp=1.1
    )
    assert len(events) == 1
    evt = events[0]
    assert evt.transition_id == "pantry_yellow"
    assert evt.camera_id == "pantry"
    assert evt.target_camera == "waiting_lobby"

# -------------------------------------------------------------
# 2. SINK ROI DETECTOR TESTS
# -------------------------------------------------------------
def test_sink_roi_enter_and_exit():
    roi_cfg = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [
                [100, 100],
                [300, 100],
                [300, 300],
                [100, 300]
            ]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_cfg, debounce_frames=2)
    
    # Frame 1: Foot point outside (50, 50)
    evts1 = detector.check_rois(1, (30, 30, 70, 70), frame_idx=1, timestamp=1.0)
    assert len(evts1) == 0
    
    # Frame 2: Foot point inside (200, 200) -> debouncing frame 1
    evts2 = detector.check_rois(1, (180, 180, 220, 220), frame_idx=2, timestamp=1.1)
    assert len(evts2) == 0
    
    # Frame 3: Inside again -> debounced, should emit ROI_ENTER
    evts3 = detector.check_rois(1, (180, 180, 220, 220), frame_idx=3, timestamp=1.2)
    assert len(evts3) == 1
    assert evts3[0].event == "ROI_ENTER"
    assert evts3[0].roi_id == "sink"
    
    # Frame 4: Leaves ROI to outside (50, 50) -> debouncing exit frame 1
    evts4 = detector.check_rois(1, (30, 30, 70, 70), frame_idx=4, timestamp=2.0)
    assert len(evts4) == 0
    
    # Frame 5: Outside again -> debounced exit, should emit ROI_EXIT with duration
    evts5 = detector.check_rois(1, (30, 30, 70, 70), frame_idx=5, timestamp=2.1)
    assert len(evts5) == 1
    assert evts5[0].event == "ROI_EXIT"
    assert evts5[0].roi_id == "sink"
    assert evts5[0].duration_seconds is not None
    assert evts5[0].duration_seconds >= 0.8

# -------------------------------------------------------------
# 3. TOPOLOGY & TEMPORAL GATING TESTS
# -------------------------------------------------------------
def test_topology_gate():
    topology = {
        "pantry": {"next_zones": ["waiting_lobby", "passage"]},
        "waiting_lobby": {"next_zones": ["pantry"]},
        "passage": {"next_zones": ["pantry"]}
    }
    gate = TopologyGate(topology_config=topology)
    
    # Valid transitions
    assert gate.is_transition_valid("pantry", "waiting_lobby") is True
    assert gate.is_transition_valid("pantry", "passage") is True
    assert gate.is_transition_valid("waiting_lobby", "pantry") is True
    
    # Invalid transition
    assert gate.is_transition_valid("waiting_lobby", "passage") is False
    # Same zone always valid
    assert gate.is_transition_valid("pantry", "pantry") is True

def test_temporal_gate():
    cfg = {
        "pantry_to_passage": {
            "min_seconds": 1.0,
            "max_seconds": 10.0
        },
        "same_camera_reacquisition_max_seconds": 5.0
    }
    gate = TemporalGate(temporal_config=cfg)
    
    # Within valid window
    is_valid, score, reason = gate.evaluate("pantry", "passage", elapsed_seconds=3.5)
    assert is_valid is True
    assert score > 0.5
    
    # Too fast (< min_seconds)
    is_valid, score, reason = gate.evaluate("pantry", "passage", elapsed_seconds=0.3)
    assert is_valid is False
    assert "TOO_FAST" in reason
    
    # Too slow (> max_seconds)
    is_valid, score, reason = gate.evaluate("pantry", "passage", elapsed_seconds=15.0)
    assert is_valid is False
    assert "TOO_SLOW" in reason
    
    # Same camera valid
    is_valid, score, reason = gate.evaluate("pantry", "pantry", elapsed_seconds=2.0)
    assert is_valid is True

# -------------------------------------------------------------
# 4. SPATIAL GATE TESTS
# -------------------------------------------------------------
def test_spatial_gate():
    rules = [
        {
            "from_camera": "pantry",
            "transition": "pantry_yellow",
            "candidate_zones": ["waiting_lobby"]
        }
    ]
    gate = SpatialGate(transition_rules=rules)
    
    # Transition matches target
    is_valid, score, reason = gate.evaluate(
        source_camera="pantry",
        current_camera="waiting_lobby",
        last_transition_id="pantry_yellow",
        last_transition_direction="L2R",
        time_since_transition=2.0
    )
    assert is_valid is True
    assert score == 1.0
    
    # Missed transition line: spatial info is evidence, NOT hard blocker
    is_valid, score, reason = gate.evaluate(
        source_camera="pantry",
        current_camera="waiting_lobby",
        last_transition_id=None,
        last_transition_direction=None,
        time_since_transition=2.0
    )
    assert is_valid is True # Non-blocking!
    assert score < 1.0 # Penalized score

# -------------------------------------------------------------
# 5. CANDIDATE RESTRICTED GALLERY SEARCH
# -------------------------------------------------------------
def test_candidate_restricted_gallery_search(tmp_path):
    storage_path = str(tmp_path / "test_gallery.pkl")
    gallery = GlobalGallery(storage_path=storage_path, clear_on_start=True)
    
    # Populate gallery with 3 distinct identities
    emb_person1 = np.ones(512, dtype=np.float32)
    emb_person2 = -np.ones(512, dtype=np.float32)
    emb_person3 = np.zeros(512, dtype=np.float32)
    emb_person3[0] = 1.0
    
    class FakeEntry:
        def __init__(self, emb):
            self.embedding = emb
            self.quality = 0.9
            self.par_features = None
            self.viewpoint = "front"
            self.quality_metadata = {}
            self.par_attributes = {}
            
    gallery.update_identity("Person_1", [{"camera_id": "cctv1", "track_id": 1}], [FakeEntry(emb_person1)])
    gallery.update_identity("Person_2", [{"camera_id": "cctv1", "track_id": 2}], [FakeEntry(emb_person2)])
    gallery.update_identity("Person_3", [{"camera_id": "cctv1", "track_id": 3}], [FakeEntry(emb_person3)])
    
    # Query vector closest to Person_2
    query_vec = -np.ones(512, dtype=np.float32)
    
    # Search restricting ONLY to candidates ["Person_1", "Person_3"]
    # Even though Person_2 is a 1.0 match, it MUST NOT be returned!
    results = gallery.search_candidates(
        query_vector=query_vec,
        candidate_gids=["Person_1", "Person_3"],
        top_k=2
    )
    
    result_gids = list(results.keys())
    assert "Person_2" not in result_gids
    assert len(results) <= 2

# -------------------------------------------------------------
# 6. JOURNEY MANAGER & CONFLICT PREVENTION TESTS
# -------------------------------------------------------------
def test_journey_manager_lifecycle_and_retroactive_events(tmp_path):
    j_log = str(tmp_path / "journey.jsonl")
    d_log = str(tmp_path / "decisions.jsonl")
    store = JourneyStore(journey_log_path=j_log, association_log_path=d_log)
    jm = JourneyManager(store=store)
    
    # Step 1: Track appears in Pantry, emits pending ROI event before GID is known
    roi_evt = RoiEvent(
        event="ROI_ENTER",
        camera_id="pantry",
        zone="pantry",
        roi_id="sink",
        local_track_id=10,
        timestamp=5.0,
        frame_idx=100
    )
    jm.record_roi_event(roi_evt)
    
    # Step 2: Track is registered as new GID "Person_001"
    state = jm.register_new_identity(
        global_id="Person_001",
        camera_id="pantry",
        zone="pantry",
        local_track_id=10,
        timestamp=5.5,
        frame_idx=115
    )
    assert state.global_id == "Person_001"
    assert state.status == IdentityStatus.ACTIVE
    assert state.last_camera == "pantry"
    # Buffered ROI event should have been retroactively attached!
    assert any(e.event == "ROI_ENTER" and e.details.get("roi_id") == "sink" for e in state.journey_history)
    
    # Step 3: Conflict detection - cannot assign Person_001 simultaneously to a different active track in waiting_lobby
    is_confirmed = jm.confirm_association(
        global_id="Person_001",
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=99,
        timestamp=6.0,
        frame_idx=130,
        decision_details={"reason": "CONFLICT_TEST"}
    )
    # Must reject conflict!
    assert is_confirmed is False
    
    # Step 4: Track 10 disappears from pantry
    jm.on_track_lost("pantry", "pantry", 10, timestamp=10.0, frame_idx=300)
    assert state.status == IdentityStatus.TEMPORARILY_LOST
    
    # Step 5: Reappears in passage (valid next camera) -> reacquired
    is_reacquired = jm.confirm_association(
        global_id="Person_001",
        camera_id="passage",
        zone="passage",
        local_track_id=22,
        timestamp=12.0,
        frame_idx=360,
        decision_details={"reason": "REACQUIRED"}
    )
    assert is_reacquired is True
    assert state.status == IdentityStatus.ACTIVE
    assert state.current_camera == "passage"
    assert state.previous_camera == "pantry"

def test_association_manager_multiframe_and_hysteresis(tmp_path):
    import json
    from app.association.association_manager import AssociationManager
    from app.reid.matcher import GlobalMatcher

    j_log = str(tmp_path / "journey.jsonl")
    d_log = str(tmp_path / "decisions.jsonl")
    gallery_path = str(tmp_path / "gallery.pkl")

    store = JourneyStore(journey_log_path=j_log, association_log_path=d_log)
    jm = JourneyManager(store=store)
    gallery = GlobalGallery(storage_path=gallery_path, clear_on_start=True)

    config = {
        "topology": {"pantry": {"next_zones": ["waiting_lobby", "passage"]}},
        "temporal_constraints": {"pantry_to_waiting_lobby": {"min_seconds": 0.5, "max_seconds": 15.0}},
        "threshold_match": 0.85,
        "threshold_ambiguous": 0.50,
        "reid_confirm_frames": 1,
        "pending_confirm_frames": 3,
        "reid_new_id_confirm_frames": 5,
        "hysteresis_switch_threshold": 0.95,
        "hysteresis_consecutive_frames": 5,
        "weight_reid": 1.0,
        "weight_par": 0.0,
        "weight_context": 0.0
    }

    matcher = GlobalMatcher(gallery=gallery, config=config)
    topology_gate = TopologyGate(topology_config=config["topology"])
    temporal_gate = TemporalGate(temporal_config=config["temporal_constraints"])
    spatial_gate = SpatialGate()
    cand_filter = CandidateFilter(topology_gate, temporal_gate, spatial_gate)

    am = AssociationManager(
        gallery=gallery,
        matcher=matcher,
        candidate_filter=cand_filter,
        journey_manager=jm,
        store=store,
        config=config
    )

    # Setup an existing GID in pantry that transitioned
    state1 = jm.register_new_identity("Person_100", "pantry", "pantry", local_track_id=1, timestamp=1.0, frame_idx=10)
    jm.on_track_lost("pantry", "pantry", local_track_id=1, timestamp=2.0, frame_idx=30)
    
    # Store embedding for Person_100 in gallery (unit vector along dim 0)
    vec_cand = np.zeros(512, dtype=np.float32)
    vec_cand[0] = 1.0

    class FakeEntry:
        def __init__(self, emb):
            self.embedding = emb
            self.quality = 0.85
            self.par_features = None
            self.viewpoint = "front"
            self.quality_metadata = {}
            self.par_attributes = {}

    gallery.update_identity("Person_100", [{"camera_id": "pantry", "track_id": 1}], [FakeEntry(vec_cand)])

    # Query vector with dot product 0.70 with Person_100 (below threshold_match 0.85, above threshold_ambig 0.50)
    vec_query = np.zeros(512, dtype=np.float32)
    vec_query[0] = 0.70
    vec_query[1] = np.sqrt(1.0 - 0.70**2)

    # New track appears in waiting_lobby at t=3.0 (elapsed = 1.0s, in range 0.5 - 15.0s)
    # Frame 1: p_count = 1 -> PENDING
    gid, status, sim, _ = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        bank_embeddings=[FakeEntry(vec_query)],
        current_time=3.0,
        frame_idx=50,
        active_camera_gids=set()
    )
    assert status == "PENDING"

    # Frame 2: p_count = 2 -> PENDING
    gid, status, sim, _ = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        bank_embeddings=[FakeEntry(vec_query)],
        current_time=3.1,
        frame_idx=51,
        active_camera_gids=set()
    )
    assert status == "PENDING"

    # Frame 3: p_count = 3 >= pending_confirm_frames -> MATCHED
    gid, status, sim, _ = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        bank_embeddings=[FakeEntry(vec_query)],
        current_time=3.2,
        frame_idx=52,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert gid == "Person_100"

    # Frame 4: Track is now CONFIRMED. Check Hysteresis:
    # A single frame with a different vector does NOT switch the confirmed identity!
    vec_noise = np.zeros(512, dtype=np.float32)
    vec_noise[2] = 1.0
    gid, status, sim, details = am.associate_track(
        camera_id="waiting_lobby",
        zone="waiting_lobby",
        local_track_id=5,
        bank_embeddings=[FakeEntry(vec_noise)],
        current_time=3.3,
        frame_idx=53,
        active_camera_gids=set()
    )
    assert status == "MATCHED"
    assert gid == "Person_100"
    assert details.get("status") == "RETAINED"

    # Verify decision log was written
    with open(d_log, "r") as f:
        lines = f.readlines()
    assert len(lines) >= 3
    last_decision = json.loads(lines[-1])
    assert "timestamp" in last_decision

