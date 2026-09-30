import os
import shutil
import tempfile
import numpy as np
import pytest

from app.violation.line_crossing import LineCrossingDetector, CrossingEvent
from app.violation.attire_smoother import AttireStabilityManager
from app.violation.person_state import PersonStateManager
from app.violation.route_policy import RoutePolicy
from app.violation.evidence_manager import EvidenceManager
from app.violation.violation_engine import ViolationEngine, ViolationEvent
from app.violation.handwash_detector import (
    HandwashDetector,
    box_center,
    bottom_center,
    point_distance,
    point_inside_expanded_box,
    hand_is_near_sink,
    assign_person_to_sink
)

# Standard camera configs matching business requirements
CAMERA_CONFIGS = {
    "waiting_lobby": {
        "zone": "waiting_lobby",
        "pathway": "exit",
        "line": {"enabled": True, "start": [120, 250], "end": [120, 500]}
    },
    "pantry": {
        "zone": "pantry",
        "pathway": "pantry",
        "line": {"enabled": False}
    },
    "passage_1": {
        "zone": "passage_1",
        "pathway": "passage_1",
        "line": {"enabled": True, "start": [550, 600], "end": [800, 600]}
    }
}


# ============================================================
# 1. Point / Line Geometry Tests
# ============================================================

def test_point_line_side_calculation():
    start = (0, 100)
    end = (200, 100)
    assert LineCrossingDetector.calculate_side((100, 50), start, end) == "SIDE_B"
    assert LineCrossingDetector.calculate_side((100, 150), start, end) == "SIDE_A"
    assert LineCrossingDetector.calculate_side((100, 100), start, end) == "COLINEAR"

    start_v = (100, 0)
    end_v = (100, 200)
    assert LineCrossingDetector.calculate_side((50, 100), start_v, end_v) == "SIDE_A"
    assert LineCrossingDetector.calculate_side((150, 100), start_v, end_v) == "SIDE_B"
    assert LineCrossingDetector.calculate_side((100, 100), start_v, end_v) == "COLINEAR"


def test_line_crossing_detection():
    detector = LineCrossingDetector("waiting_lobby", line_start=(100, 0), line_end=(100, 500), enabled=True)
    track_id = 1
    
    # Frame 1: Person on Side A
    bbox1 = (40, 100, 60, 200)
    assert detector.check_crossing(track_id, bbox1) is None
    
    # Frame 2: Person moves to Side B -> Crossing triggered
    bbox2 = (140, 100, 160, 200)
    evt = detector.check_crossing(track_id, bbox2)
    assert evt is not None
    assert evt.local_track_id == 1
    assert evt.camera_id == "waiting_lobby"
    assert evt.previous_side == "SIDE_A"
    assert evt.current_side == "SIDE_B"
    assert evt.crossing_direction == "SIDE_A_TO_SIDE_B"


def test_no_crossing_when_staying_on_same_side():
    detector = LineCrossingDetector("waiting_lobby", line_start=(100, 0), line_end=(100, 500), enabled=True)
    track_id = 2
    bboxes = [
        (40, 100, 60, 200),
        (30, 110, 50, 210),
        (20, 120, 40, 220),
        (50, 90, 70, 190)
    ]
    for b in bboxes:
        assert detector.check_crossing(track_id, b) is None


# ============================================================
# 2. Core Business Scenarios (Section 18)
# ============================================================

# Test 1: Person A -> Pantry -> Handwash detected -> Passage 1 -> VALID
def test_scenario_1_handwashed_passage_1_valid():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        # In Pantry: Person A (GID 12) confirmed HANDWASHED
        person_state.update_handwash_status("GID_12", "HANDWASHED")
        assert person_state.get_handwash_status("GID_12") == "HANDWASHED"

        # In Passage 1: Crosses virtual line
        crossing = CrossingEvent(
            camera_id="passage_1",
            local_track_id=1,
            timestamp="2026-09-30T10:00:00",
            previous_side="SIDE_B",
            current_side="SIDE_A",
            crossing_direction="SIDE_B_TO_SIDE_A",
            foot_point=(650.0, 620.0),
            line_start=(550, 600),
            line_end=(800, 600)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = engine.process_crossing(
            crossing_event=crossing,
            global_id="GID_12",
            frame=dummy_frame,
            bbox=(600, 400, 700, 620)
        )

        # Must NOT be a violation -> VALID
        assert event is None
        p_state = person_state.get_state("GID_12")
        assert p_state["validation_status"] == "VALID"
        assert p_state["violation_status"] is False
        assert p_state["pathway"] == "PASSAGE_1"

        val_info = engine.get_validation("passage_1", 1)
        assert val_info is not None
        assert val_info["status"] == "VALID"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test 2: Person B -> Pantry -> No handwash -> Passage 1 -> VIOLATION
def test_scenario_2_not_handwashed_passage_1_violation():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        # In Pantry: Person B (GID 18) marked NOT_HANDWASHED
        person_state.update_handwash_status("GID_18", "NOT_HANDWASHED")
        assert person_state.get_handwash_status("GID_18") == "NOT_HANDWASHED"

        # In Passage 1: Crosses virtual line
        crossing = CrossingEvent(
            camera_id="passage_1",
            local_track_id=2,
            timestamp="2026-09-30T10:01:00",
            previous_side="SIDE_B",
            current_side="SIDE_A",
            crossing_direction="SIDE_B_TO_SIDE_A",
            foot_point=(650.0, 620.0),
            line_start=(550, 600),
            line_end=(800, 600)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = engine.process_crossing(
            crossing_event=crossing,
            global_id="GID_18",
            frame=dummy_frame,
            bbox=(600, 400, 700, 620)
        )

        # Must be VIOLATION
        assert event is not None
        assert event.validation_result == "VIOLATION"
        assert event.handwash_status == "NOT_HANDWASHED"
        assert event.pathway == "PASSAGE_1"
        assert event.global_id == "GID_18"

        p_state = person_state.get_state("GID_18")
        assert p_state["validation_status"] == "VIOLATION"
        assert p_state["violation_status"] is True

        vio = engine.get_violation("passage_1", 2)
        assert vio is not None
        assert vio.global_id == "GID_18"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test 3: Person C -> Pantry -> Handwash detected -> Exit pathway -> VIOLATION
def test_scenario_3_handwashed_exit_violation():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        # Person C (GID 25) washed hands
        person_state.update_handwash_status("GID_25", "HANDWASHED")

        # In Exit / Waiting Lobby: Crosses line
        crossing = CrossingEvent(
            camera_id="waiting_lobby",
            local_track_id=3,
            timestamp="2026-09-30T10:02:00",
            previous_side="SIDE_A",
            current_side="SIDE_B",
            crossing_direction="SIDE_A_TO_SIDE_B",
            foot_point=(150.0, 300.0),
            line_start=(120, 250),
            line_end=(120, 500)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = engine.process_crossing(
            crossing_event=crossing,
            global_id="GID_25",
            frame=dummy_frame,
            bbox=(130, 100, 170, 300)
        )

        # HANDWASHED + EXIT = VIOLATION
        assert event is not None
        assert event.validation_result == "VIOLATION"
        assert event.handwash_status == "HANDWASHED"
        assert event.pathway == "EXIT"
        assert event.global_id == "GID_25"
        assert person_state.get_state("GID_25")["violation_status"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test 4: Person D -> Pantry -> No handwash -> Exit pathway -> VALID
def test_scenario_4_not_handwashed_exit_valid():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        # Person D (GID 21) did not wash hands
        person_state.update_handwash_status("GID_21", "NOT_HANDWASHED")

        # In Exit pathway: Crosses line
        crossing = CrossingEvent(
            camera_id="waiting_lobby",
            local_track_id=4,
            timestamp="2026-09-30T10:03:00",
            previous_side="SIDE_A",
            current_side="SIDE_B",
            crossing_direction="SIDE_A_TO_SIDE_B",
            foot_point=(150.0, 300.0),
            line_start=(120, 250),
            line_end=(120, 500)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = engine.process_crossing(
            crossing_event=crossing,
            global_id="GID_21",
            frame=dummy_frame,
            bbox=(130, 100, 170, 300)
        )

        # NOT_HANDWASHED + EXIT = VALID
        assert event is None
        p_state = person_state.get_state("GID_21")
        assert p_state["validation_status"] == "VALID"
        assert p_state["violation_status"] is False
        assert p_state["pathway"] == "EXIT"

        val = engine.get_validation("waiting_lobby", 4)
        assert val is not None
        assert val["status"] == "VALID"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test 5: Person E -> Pantry -> Handwash initially UNKNOWN -> Later confirmed HANDWASHED -> Passage 1 -> VALID
def test_scenario_5_initially_unknown_then_handwashed_valid():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        # Initially UNKNOWN
        person_state.get_or_create("GID_50")
        assert person_state.get_handwash_status("GID_50") == "UNKNOWN"

        # If person were evaluated while still UNKNOWN, do not classify as violation
        res_unknown = policy.evaluate_compliance("passage_1", "UNKNOWN", "PASSAGE_1")
        assert res_unknown["result"] == "UNKNOWN"
        assert res_unknown["is_violation"] is False

        # Later confirmed HANDWASHED in pantry
        person_state.update_handwash_status("GID_50", "HANDWASHED")
        assert person_state.get_handwash_status("GID_50") == "HANDWASHED"

        # Now crosses Passage 1 line
        crossing = CrossingEvent(
            camera_id="passage_1",
            local_track_id=5,
            timestamp="2026-09-30T10:05:00",
            previous_side="SIDE_B",
            current_side="SIDE_A",
            crossing_direction="SIDE_B_TO_SIDE_A",
            foot_point=(650.0, 620.0),
            line_start=(550, 600),
            line_end=(800, 600)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        event = engine.process_crossing(
            crossing_event=crossing,
            global_id="GID_50",
            frame=dummy_frame,
            bbox=(600, 400, 700, 620)
        )

        assert event is None
        assert person_state.get_state("GID_50")["validation_status"] == "VALID"
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Test 6 — ID switching protection
def test_scenario_6_id_switching_protection():
    person_state = PersonStateManager(CAMERA_CONFIGS)

    # Person A is established as HANDWASHED
    person_state.update_handwash_status("GID_A", "HANDWASHED", confidence=0.95)
    # Person B is established as NOT_HANDWASHED
    person_state.update_handwash_status("GID_B", "NOT_HANDWASHED", confidence=0.90)

    assert person_state.get_handwash_status("GID_A") == "HANDWASHED"
    assert person_state.get_handwash_status("GID_B") == "NOT_HANDWASHED"

    # Attempt an inadvertent downgrade or swap onto GID_A
    person_state.update_handwash_status("GID_A", "NOT_HANDWASHED")
    # Protection must preserve HANDWASHED!
    assert person_state.get_handwash_status("GID_A") == "HANDWASHED"

    # Also attempt updating with UNKNOWN
    person_state.update_handwash_status("GID_A", "UNKNOWN")
    assert person_state.get_handwash_status("GID_A") == "HANDWASHED"

    # GID_B remains NOT_HANDWASHED
    assert person_state.get_handwash_status("GID_B") == "NOT_HANDWASHED"


# Test 7 — Temporary Occlusion resilience
def test_scenario_7_occlusion_resilience():
    person_state = PersonStateManager(CAMERA_CONFIGS)

    # Person (GID_12) in pantry gets washed
    person_state.update_handwash_status("GID_12", "HANDWASHED")
    person_state.link_track("pantry", 101, "GID_12")

    # Person moves to passage camera, tracked as local track 205
    person_state.link_track("passage_1", 205, "GID_12")
    assert person_state.get_global_id("passage_1", 205) == "GID_12"
    assert person_state.get_handwash_status("GID_12") == "HANDWASHED"

    # Occlusion occurs: track 205 drops. A new track 210 appears.
    # PersonStateManager state for GID_12 remains completely intact!
    assert person_state.get_handwash_status("GID_12") == "HANDWASHED"

    # Once re-associated to GID_12:
    person_state.link_track("passage_1", 210, "GID_12")
    assert person_state.get_global_id("passage_1", 210) == "GID_12"
    assert person_state.get_handwash_status("GID_12") == "HANDWASHED"


# ============================================================
# 3. System & Edge Case Tests
# ============================================================

def test_pantry_no_pathway_violation():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res_hw = policy.evaluate_compliance("pantry", "HANDWASHED", "PANTRY")
    assert res_hw["is_violation"] is False
    assert res_hw["result"] is None

    res_nhw = policy.evaluate_compliance("pantry", "NOT_HANDWASHED", "PANTRY")
    assert res_nhw["is_violation"] is False
    assert res_nhw["result"] is None


def test_unknown_state_no_premature_violation():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res_p1 = policy.evaluate_compliance("passage_1", "UNKNOWN", "PASSAGE_1")
    assert res_p1["is_violation"] is False
    assert res_p1["result"] == "UNKNOWN"

    res_exit = policy.evaluate_compliance("waiting_lobby", "UNKNOWN", "EXIT")
    assert res_exit["is_violation"] is False
    assert res_exit["result"] == "UNKNOWN"


def test_duplicate_violation_prevention():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        person_state.update_handwash_status("GID_001", "NOT_HANDWASHED")

        crossing = CrossingEvent(
            camera_id="passage_1",
            local_track_id=17,
            timestamp="2026-09-30T12:00:00",
            previous_side="SIDE_B",
            current_side="SIDE_A",
            crossing_direction="SIDE_B_TO_SIDE_A",
            foot_point=(650.0, 620.0),
            line_start=(550, 600),
            line_end=(800, 600)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # First crossing evaluation -> Generates VIO_0001
        event1 = engine.process_crossing(crossing, "GID_001", "NOT_HANDWASHED", dummy_frame, (600, 400, 700, 620))
        assert event1 is not None
        assert event1.event_id == "VIO_0001"

        # Subsequent call with same crossing -> None (duplicate prevented)
        event2 = engine.process_crossing(crossing, "GID_001", "NOT_HANDWASHED", dummy_frame, (600, 400, 700, 620))
        assert event2 is None
        assert len(engine.events) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_evidence_file_creation():
    temp_dir = tempfile.mkdtemp()
    try:
        evidence = EvidenceManager(output_dir=temp_dir, enabled=True, save_image=True, save_metadata=True)
        event = ViolationEvent(
            event_id="VIO_0001",
            global_id="GID_001",
            local_track_id=17,
            camera_id="passage_1",
            zone="passage_1",
            pathway="PASSAGE_1",
            handwash_status="NOT_HANDWASHED",
            validation_result="VIOLATION",
            violation_type="handwash_pathway_violation",
            timestamp="2026-09-30T12:42:51",
            crossing_direction="SIDE_B_TO_SIDE_A",
            reason="not_handwashed_passage_1_violation"
        )
        dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        bbox = (600, 400, 700, 620)
        line = ((550, 600), (800, 600))

        img_path = evidence.save_evidence(event, dummy_frame, bbox, line)
        assert img_path is not None
        assert os.path.exists(img_path)
        assert "VIO_0001_GID_001_NOT_HANDWASHED_PASSAGE_1.jpg" in img_path

        # Check metadata files
        daily_dir = os.path.dirname(img_path)
        assert os.path.exists(os.path.join(daily_dir, "events.json"))
        assert os.path.exists(os.path.join(daily_dir, "VIO_0001.json"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_handwash_detector_geometry():
    # Test box_center and bottom_center
    box = (100, 200, 200, 400)
    assert box_center(box) == (150, 300)
    assert bottom_center(box) == (150, 400)

    # Test point distance
    assert point_distance((0, 0), (3, 4)) == 5.0

    # Test point inside expanded box
    assert point_inside_expanded_box((150, 300), box) is True
    assert point_inside_expanded_box((90, 190), box) is True  # Inside expansion
    assert point_inside_expanded_box((50, 50), box) is False

    # Test hand near sink
    sink_box = (300, 300, 500, 500)
    # Hand directly above sink
    hand_in_sink = (350, 320, 400, 370)
    assert hand_is_near_sink(hand_in_sink, sink_box) is True
    # Hand far away
    hand_far = (50, 50, 80, 80)
    assert hand_is_near_sink(hand_far, sink_box) is False

    # Test assign_person_to_sink
    person_box = (320, 200, 480, 550)  # Person right in front of sink
    sink_boxes = [(sink_box, 0.9)]
    assert assign_person_to_sink(person_box, sink_boxes) == 0


def test_handwash_detector_state_lifecycle():
    detector = HandwashDetector(
        config={
            "min_washing_duration": 1.0,
            "detection_confirmation_frames": 2,
            "reset_after_no_detection_seconds": 0.5,
            "require_hand_movement": False
        }
    )

    track_id = 99
    # Initial status
    assert detector.get_status(track_id) == "UNKNOWN"

    # Simulate person leaving pantry without washing -> NOT_HANDWASHED
    final_status = detector.on_track_lost(track_id)
    assert final_status == "NOT_HANDWASHED"

    # If person achieved HANDWASHED, on_track_lost preserves HANDWASHED
    track_id2 = 100
    detector.set_status(track_id2, "HANDWASHED")
    assert detector.on_track_lost(track_id2) == "HANDWASHED"
