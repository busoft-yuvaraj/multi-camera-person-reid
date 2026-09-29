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

# Standard camera configs matching requirements
CAMERA_CONFIGS = {
    "waiting_lobby": {
        "zone": "waiting_lobby",
        "allowed_attire": "informal",
        "line": {"enabled": True, "start": [120, 250], "end": [120, 500]}
    },
    "pantry": {
        "zone": "pantry",
        "allowed_attire": None,
        "line": {"enabled": False}
    },
    "passage_1": {
        "zone": "passage_1",
        "allowed_attire": "formal",
        "line": {"enabled": True, "start": [550, 600], "end": [800, 600]}
    }
}

# 1. Point/line side calculation
def test_point_line_side_calculation():
    # Horizontal line from (0, 100) to (200, 100)
    start = (0, 100)
    end = (200, 100)
    
    # Point above line (smaller y in image coordinates)
    assert LineCrossingDetector.calculate_side((100, 50), start, end) == "SIDE_B"
    # Point below line (larger y in image coordinates)
    assert LineCrossingDetector.calculate_side((100, 150), start, end) == "SIDE_A"
    # Collinear point
    assert LineCrossingDetector.calculate_side((100, 100), start, end) == "COLINEAR"

    # Vertical line from (100, 0) to (100, 200)
    start_v = (100, 0)
    end_v = (100, 200)
    # Point to the left (x < 100): d = (100-100)*(100-0) - (200-0)*(50-100) = +10000 > 0
    assert LineCrossingDetector.calculate_side((50, 100), start_v, end_v) == "SIDE_A"
    # Point to the right (x > 100): d = -200*(150-100) = -10000 < 0
    assert LineCrossingDetector.calculate_side((150, 100), start_v, end_v) == "SIDE_B"
    # Collinear point
    assert LineCrossingDetector.calculate_side((100, 100), start_v, end_v) == "COLINEAR"

    # Diagonal line from (0, 0) to (100, 100)
    start_d = (0, 0)
    end_d = (100, 100)
    assert LineCrossingDetector.calculate_side((20, 80), start_d, end_d) == "SIDE_A"
    assert LineCrossingDetector.calculate_side((80, 20), start_d, end_d) == "SIDE_B"
    assert LineCrossingDetector.calculate_side((50, 50), start_d, end_d) == "COLINEAR"


# 2. Line crossing detection
def test_line_crossing_detection():
    detector = LineCrossingDetector("waiting_lobby", line_start=(100, 0), line_end=(100, 500), enabled=True)
    track_id = 1
    
    # Frame 1: Person on Side A (foot_x = 50, foot_y = 200)
    bbox1 = (40, 100, 60, 200)
    evt1 = detector.check_crossing(track_id, bbox1)
    assert evt1 is None  # Initial registration, not crossing
    
    # Frame 2: Person moves to Side B (foot_x = 150, foot_y = 200)
    bbox2 = (140, 100, 160, 200)
    evt2 = detector.check_crossing(track_id, bbox2)
    assert evt2 is not None
    assert evt2.local_track_id == 1
    assert evt2.camera_id == "waiting_lobby"
    assert evt2.previous_side == "SIDE_A"
    assert evt2.current_side == "SIDE_B"
    assert evt2.crossing_direction == "SIDE_A_TO_SIDE_B"


# 3. No crossing when person stays on same side
def test_no_crossing_when_staying_on_same_side():
    detector = LineCrossingDetector("waiting_lobby", line_start=(100, 0), line_end=(100, 500), enabled=True)
    track_id = 2
    
    # Person wanders around on Side B for multiple frames
    bboxes = [
        (40, 100, 60, 200),
        (30, 110, 50, 210),
        (20, 120, 40, 220),
        (50, 90, 70, 190)
    ]
    for b in bboxes:
        evt = detector.check_crossing(track_id, b)
        assert evt is None


# 4. Formal + waiting_lobby => violation
def test_formal_waiting_lobby_violation():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res = policy.evaluate_violation("waiting_lobby", actual_attire="formal", attire_conf=0.95, min_conf=0.70)
    assert res["is_violation"] is True
    assert res["expected_attire"] == "informal"
    assert res["actual_attire"] == "formal"


# 5. Informal + waiting_lobby => allowed
def test_informal_waiting_lobby_allowed():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res = policy.evaluate_violation("waiting_lobby", actual_attire="informal", attire_conf=0.92, min_conf=0.70)
    assert res["is_violation"] is False
    assert res["expected_attire"] == "informal"
    assert res["actual_attire"] == "informal"


# 6. Formal + passage_1 => allowed
def test_formal_passage_1_allowed():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res = policy.evaluate_violation("passage_1", actual_attire="formal", attire_conf=0.90, min_conf=0.70)
    assert res["is_violation"] is False
    assert res["expected_attire"] == "formal"
    assert res["actual_attire"] == "formal"


# 7. Informal + passage_1 => violation
def test_informal_passage_1_violation():
    policy = RoutePolicy(CAMERA_CONFIGS)
    res = policy.evaluate_violation("passage_1", actual_attire="informal", attire_conf=0.88, min_conf=0.70)
    assert res["is_violation"] is True
    assert res["expected_attire"] == "formal"
    assert res["actual_attire"] == "informal"


# 8. Pantry => no attire violation
def test_pantry_no_attire_violation():
    policy = RoutePolicy(CAMERA_CONFIGS)
    # Formal in pantry
    res_f = policy.evaluate_violation("pantry", actual_attire="formal", attire_conf=0.95)
    assert res_f["is_violation"] is False
    assert res_f["expected_attire"] is None

    # Informal in pantry
    res_i = policy.evaluate_violation("pantry", actual_attire="informal", attire_conf=0.95)
    assert res_i["is_violation"] is False
    assert res_i["expected_attire"] is None


# 9. Duplicate violation prevention
def test_duplicate_violation_prevention():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=True, save_image=False, save_metadata=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        crossing_event = CrossingEvent(
            camera_id="waiting_lobby",
            local_track_id=17,
            timestamp="2026-09-29T12:00:00",
            previous_side="SIDE_A",
            current_side="SIDE_B",
            crossing_direction="SIDE_A_TO_SIDE_B",
            foot_point=(150.0, 300.0),
            line_start=(120, 250),
            line_end=(120, 500)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bbox = (130, 100, 170, 300)
        stable_attire = {"final_attire": "formal", "confidence": 0.90}

        # First crossing evaluation -> Generates VIO_0001
        event1 = engine.process_crossing(crossing_event, "GID_001", stable_attire, dummy_frame, bbox)
        assert event1 is not None
        assert event1.event_id == "VIO_0001"

        # Subsequent call with same crossing -> None (duplicate prevented)
        event2 = engine.process_crossing(crossing_event, "GID_001", stable_attire, dummy_frame, bbox)
        assert event2 is None
        assert len(engine.events) == 1
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 10. Evidence file creation
def test_evidence_file_creation():
    temp_dir = tempfile.mkdtemp()
    try:
        evidence = EvidenceManager(output_dir=temp_dir, enabled=True, save_image=True, save_metadata=True)
        event = ViolationEvent(
            event_id="VIO_0001",
            global_id="GID_001",
            local_track_id=17,
            camera_id="waiting_lobby",
            zone="waiting_lobby",
            actual_attire="formal",
            attire_confidence=0.93,
            expected_attire="informal",
            violation_type="wrong_route",
            timestamp="2026-09-29T12:42:51",
            crossing_direction="SIDE_A_TO_SIDE_B",
            evidence_image=None
        )
        dummy_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
        bbox = (100, 100, 200, 400)
        line = ((120, 250), (120, 500))

        img_path = evidence.save_evidence(event, dummy_frame, bbox, line)
        assert img_path is not None
        assert os.path.exists(img_path)
        assert img_path.endswith("VIO_0001_GID_001_FORMAL_WAITING_LOBBY.jpg")

        # Check metadata files
        daily_dir = os.path.dirname(img_path)
        assert os.path.exists(os.path.join(daily_dir, "events.json"))
        assert os.path.exists(os.path.join(daily_dir, "VIO_0001.json"))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 11. Global_ID included in event
def test_global_id_included_in_event():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=False)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        crossing_event = CrossingEvent(
            camera_id="passage_1",
            local_track_id=42,
            timestamp="2026-09-29T12:10:00",
            previous_side="SIDE_B",
            current_side="SIDE_A",
            crossing_direction="SIDE_B_TO_SIDE_A",
            foot_point=(650.0, 620.0),
            line_start=(550, 600),
            line_end=(800, 600)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        event = engine.process_crossing(
            crossing_event=crossing_event,
            global_id="GID_007",
            stable_attire={"final_attire": "informal", "confidence": 0.95},
            frame=dummy_frame,
            bbox=(600, 400, 700, 620)
        )
        assert event is not None
        assert event.global_id == "GID_007"
        assert event.local_track_id == 42
        assert event.to_dict()["global_id"] == "GID_007"
        assert person_state.get_state("GID_007")["violation_status"] is True
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# 12. Missing Global_ID handled safely
def test_missing_global_id_handled_safely():
    temp_dir = tempfile.mkdtemp()
    try:
        policy = RoutePolicy(CAMERA_CONFIGS)
        evidence = EvidenceManager(output_dir=temp_dir, enabled=True, save_image=True, save_metadata=True)
        person_state = PersonStateManager(CAMERA_CONFIGS)
        engine = ViolationEngine({}, policy, evidence, person_state)

        crossing_event = CrossingEvent(
            camera_id="waiting_lobby",
            local_track_id=99,
            timestamp="2026-09-29T12:15:00",
            previous_side="SIDE_A",
            current_side="SIDE_B",
            crossing_direction="SIDE_A_TO_SIDE_B",
            foot_point=(150.0, 320.0),
            line_start=(120, 250),
            line_end=(120, 500)
        )
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # None passed as global_id
        event = engine.process_crossing(
            crossing_event=crossing_event,
            global_id=None,
            stable_attire={"final_attire": "formal", "confidence": 0.88},
            frame=dummy_frame,
            bbox=(130, 100, 170, 320)
        )
        assert event is not None
        assert event.global_id is None
        assert event.local_track_id == 99
        assert event.evidence_image is not None
        assert os.path.exists(event.evidence_image)
        assert "GID_NONE" in event.evidence_image
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# Temporal Attire Stability test
def test_attire_temporal_stability():
    smoother = AttireStabilityManager(history_size=5, minimum_confidence=0.70, prediction_interval=2)
    track_id = 10
    
    # Needs prediction on first frame
    assert smoother.should_predict(0, track_id) is True
    smoother.add_prediction(track_id, {"label": "formal_dress", "conf": 0.90, "is_formal": True})

    # Frame 1: prediction_interval is 2 -> False
    assert smoother.should_predict(1, track_id) is False
    # Frame 2: True
    assert smoother.should_predict(2, track_id) is True

    # Add more predictions: [formal(0.9), formal(0.85), informal(0.60), formal(0.95)]
    smoother.add_prediction(track_id, {"label": "formal_dress", "conf": 0.85, "is_formal": True})
    smoother.add_prediction(track_id, {"label": "informal_dress", "conf": 0.60, "is_formal": False})
    smoother.add_prediction(track_id, {"label": "formal_dress", "conf": 0.95, "is_formal": True})

    stable = smoother.get_stable_attire(track_id)
    assert stable["final_attire"] == "formal"
    assert stable["is_formal"] is True
    assert stable["confidence"] > 0.80
    assert stable["is_confident"] is True
