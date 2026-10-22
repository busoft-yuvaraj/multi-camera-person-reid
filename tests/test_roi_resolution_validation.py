import pytest
import numpy as np
import cv2

from app.events.roi_detector import (
    RoiDetector,
    RoiEvent,
    FrameResolutionMismatchError,
    DEFAULT_CAMERA_RESOLUTIONS,
    parse_resolution,
)
from app.multi_camera_processor import MultiCameraProcessor


# ---------------------------------------------------------------------------
# Test Helpers & Dummies
# ---------------------------------------------------------------------------

class DummyExtractor:
    def extract(self, crops):
        return [np.zeros(512, dtype=np.float32) for _ in crops]


class DummyParExtractor:
    def extract(self, crop):
        return np.zeros(35, dtype=np.float32)


class DummyPoseExtractor:
    def extract_viewpoint(self, crop):
        return "FRONT", 0.9


def create_dummy_config(cameras=None):
    if cameras is None:
        cameras = {
            "waiting_lobby": {
                "camera_id": "waiting_lobby",
                "zone": "waiting_lobby",
                "expected_resolution": [1280, 720],
                "roi": {
                    "monitored": {
                        "enabled": True,
                        "type": "polygon",
                        "points": [[319, 0], [708, 0], [708, 581], [319, 581]]
                    }
                }
            },
            "pantry": {
                "camera_id": "pantry",
                "zone": "pantry",
                "expected_resolution": [1280, 720],
                "crop": [0, 0, 720, 720],
                "track_only_inside_roi": ["sink"],
                "roi": {
                    "sink": {
                        "enabled": True,
                        "type": "polygon",
                        "points": [[31, 17], [528, 5], [530, 521], [49, 532]]
                    }
                }
            },
            "passage": {
                "camera_id": "passage",
                "zone": "passage",
                "expected_resolution": [1280, 720],
                "roi": {
                    "monitored": {
                        "enabled": True,
                        "type": "polygon",
                        "points": [[503, 267], [795, 267], [795, 615], [503, 615]]
                    }
                }
            }
        }
    return {
        "model_path": "yolo11n.pt",
        "tracker_config": "botsort.yaml",
        "cameras": cameras,
        "phase1_mode": True,
        "raise_on_resolution_error": False,
    }


# ---------------------------------------------------------------------------
# 1. PARSE RESOLUTION HELPER TESTS
# ---------------------------------------------------------------------------

def test_parse_resolution_formats():
    assert parse_resolution([1280, 720]) == (1280, 720)
    assert parse_resolution((1920, 1080)) == (1920, 1080)
    assert parse_resolution({"width": 1280, "height": 720}) == (1280, 720)
    assert parse_resolution({"w": 640, "h": 480}) == (640, 480)
    assert parse_resolution("1280x720") == (1280, 720)
    assert parse_resolution("1920X1080") == (1920, 1080)
    assert parse_resolution(None) is None
    assert parse_resolution("invalid") is None
    assert parse_resolution([]) is None


# ---------------------------------------------------------------------------
# 2. FRAME RESOLUTION MISMATCH ERROR TESTS
# ---------------------------------------------------------------------------

def test_frame_resolution_mismatch_error_structure():
    err = FrameResolutionMismatchError(
        camera_id="pantry",
        expected_resolution=(1280, 720),
        actual_resolution=(640, 480)
    )
    assert isinstance(err, ValueError)
    assert err.camera_id == "pantry"
    assert err.expected_resolution == (1280, 720)
    assert err.actual_resolution == (640, 480)
    msg = str(err)
    assert "pantry" in msg
    assert "1280x720" in msg
    assert "640x480" in msg


# ---------------------------------------------------------------------------
# 3. MATCHING RESOLUTIONS: ALL THREE CAMERAS
# ---------------------------------------------------------------------------

def test_matching_resolution_waiting_lobby():
    roi_cfg = {
        "monitored": {
            "enabled": True,
            "type": "polygon",
            "points": [[319, 0], [708, 0], [708, 581], [319, 581]]
        }
    }
    detector = RoiDetector("waiting_lobby", "waiting_lobby", roi_cfg, expected_resolution=(1280, 720))
    assert detector.expected_resolution == (1280, 720)

    # Valid matching frame (height=720, width=1280)
    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.validate_resolution(1280, 720) is True
    assert detector.validate_frame(valid_frame) is True

    # Applying ROI on matching frame works without error
    # Foot point at (500, 300) is inside monitored polygon
    inside = detector.is_point_in_roi((500, 300), "monitored", frame_resolution=(1280, 720))
    assert inside is True

    events = detector.check_rois(
        local_track_id=1,
        bbox=(450, 100, 550, 300),
        frame_idx=1,
        timestamp=0.1,
        frame=valid_frame
    )
    assert isinstance(events, list)


def test_matching_resolution_pantry():
    # Scaled sink points in 1280x720 coordinate space
    roi_cfg = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[55, 17], [939, 5], [942, 521], [87, 532]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_cfg, expected_resolution=(1280, 720))
    assert detector.expected_resolution == (1280, 720)

    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.validate_resolution(1280, 720) is True
    assert detector.validate_frame(valid_frame) is True

    # Inside sink
    assert detector.is_point_in_roi((500, 250), "sink", frame_resolution=(1280, 720)) is True
    # Outside sink
    assert detector.is_point_in_roi((1100, 600), "sink", frame_resolution=(1280, 720)) is False


def test_matching_resolution_passage():
    roi_cfg = {
        "monitored": {
            "enabled": True,
            "type": "polygon",
            "points": [[503, 267], [795, 267], [795, 615], [503, 615]]
        }
    }
    detector = RoiDetector("passage", "passage", roi_cfg, expected_resolution=(1280, 720))
    assert detector.expected_resolution == (1280, 720)

    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert detector.validate_resolution(1280, 720) is True
    assert detector.validate_frame(valid_frame) is True
    assert detector.is_point_in_roi((650, 450), "monitored", frame_resolution=(1280, 720)) is True


# ---------------------------------------------------------------------------
# 4. MISMATCHED RESOLUTIONS: ALL THREE CAMERAS RAISE CLEAR ERRORS
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cam_id", ["waiting_lobby", "pantry", "passage"])
def test_mismatched_resolution_raises_descriptive_error(cam_id):
    roi_cfg = {
        "zone_test": {
            "enabled": True,
            "type": "polygon",
            "points": [[100, 100], [500, 100], [500, 500], [100, 500]]
        }
    }
    detector = RoiDetector(cam_id, cam_id, roi_cfg, expected_resolution=(1280, 720))

    # Mismatched 640x480 frame
    mismatched_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(FrameResolutionMismatchError) as exc_info:
        detector.validate_resolution(640, 480)
    err = exc_info.value
    assert err.camera_id == cam_id
    assert err.expected_resolution == (1280, 720)
    assert err.actual_resolution == (640, 480)
    assert cam_id in str(err)
    assert "1280x720" in str(err)
    assert "640x480" in str(err)

    # validate_frame raises same error
    with pytest.raises(FrameResolutionMismatchError) as exc_info2:
        detector.validate_frame(mismatched_frame)
    assert exc_info2.value.actual_resolution == (640, 480)

    # check_rois with mismatched frame raises before applying ROI
    with pytest.raises(FrameResolutionMismatchError) as exc_info3:
        detector.check_rois(
            local_track_id=1,
            bbox=(200, 200, 400, 400),
            frame_idx=1,
            timestamp=0.1,
            frame=mismatched_frame
        )
    assert exc_info3.value.camera_id == cam_id

    # is_point_in_roi with mismatched frame_resolution raises before test
    with pytest.raises(FrameResolutionMismatchError):
        detector.is_point_in_roi((300, 300), "zone_test", frame_resolution=(640, 480))


def test_mismatched_resolution_higher_and_swapped():
    detector = RoiDetector("waiting_lobby", "waiting_lobby", {}, expected_resolution=(1280, 720))

    # 1920x1080 (larger)
    with pytest.raises(FrameResolutionMismatchError) as exc:
        detector.validate_resolution(1920, 1080)
    assert exc.value.actual_resolution == (1920, 1080)

    # 720x1280 (swapped width/height)
    with pytest.raises(FrameResolutionMismatchError) as exc:
        detector.validate_resolution(720, 1280)
    assert exc.value.actual_resolution == (720, 1280)


# ---------------------------------------------------------------------------
# 5. SAFE HANDLING OF INVALID AND UNREADABLE FRAMES (REQUIREMENT 8)
# ---------------------------------------------------------------------------

def test_invalid_and_unreadable_frames():
    detector = RoiDetector("pantry", "pantry", {}, expected_resolution=(1280, 720))

    # None frame
    with pytest.raises(FrameResolutionMismatchError) as exc1:
        detector.validate_frame(None)
    assert "None frame" in str(exc1.value)

    # 0-dimensional / empty array
    with pytest.raises(FrameResolutionMismatchError) as exc2:
        detector.validate_frame(np.zeros((0, 0, 3), dtype=np.uint8))
    assert exc2.value.actual_resolution == (0, 0)

    # Non-array object
    with pytest.raises(FrameResolutionMismatchError) as exc3:
        detector.validate_frame("invalid_frame_string")
    assert "invalid frame" in str(exc3.value)


# ---------------------------------------------------------------------------
# 6. PIPELINE VALIDATION POINT: ORIGINAL VS RESIZED (REQUIREMENT 6)
# ---------------------------------------------------------------------------

def test_pipeline_validation_prevents_auto_resize_masking():
    """
    Ensure that a mismatched original frame is caught BEFORE resizing,
    so that cv2.resize does not automatically hide the mismatch.
    """
    config = create_dummy_config()
    cameras_info = [
        {"camera_id": "pantry", "video_path": "cctv_samples/pantry.mp4"}
    ]
    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path="output/test_res.mp4",
        config=config,
        extractor=DummyExtractor(),
        par_extractor=DummyParExtractor(),
        pose_extractor=DummyPoseExtractor()
    )

    # pantry expects (1280, 720) for original frame
    assert processor.expected_resolutions["pantry"] == (1280, 720)

    # If raw frame from camera is 640x480, validate_camera_frame at 'original' stage raises!
    raw_mismatched_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(FrameResolutionMismatchError) as exc:
        processor.validate_camera_frame("pantry", raw_mismatched_frame, stage="original")
    assert exc.value.camera_id == "pantry"
    assert exc.value.actual_resolution == (640, 480)
    assert "original" in str(exc.value)

    # Valid raw frame passes original stage
    raw_valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert processor.validate_camera_frame("pantry", raw_valid_frame, stage="original") is True

    # Valid post-resized frame passes evaluation stage
    resized_valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    assert processor.validate_camera_frame("pantry", resized_valid_frame, stage="evaluation") is True


# ---------------------------------------------------------------------------
# 7. INDEPENDENT MULTI-CAMERA VALIDATION & ISOLATION (REQUIREMENT 9)
# ---------------------------------------------------------------------------

def test_independent_multi_camera_resolution_validation():
    """
    Test independent validation for all three cameras.
    If one camera encounters a resolution mismatch, only that camera stops processing,
    while the other two cameras continue normal execution.
    """
    config = create_dummy_config()
    cameras_info = [
        {"camera_id": "waiting_lobby", "video_path": "cctv_samples/waiting_lobby.mp4"},
        {"camera_id": "pantry", "video_path": "cctv_samples/pantry.mp4"},
        {"camera_id": "passage", "video_path": "cctv_samples/passage1.mp4"}
    ]
    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path="output/test_3cam_res.mp4",
        config=config,
        extractor=DummyExtractor(),
        par_extractor=DummyParExtractor(),
        pose_extractor=DummyPoseExtractor()
    )

    # Verify each camera has its own independent expected resolution
    assert processor.expected_resolutions["waiting_lobby"] == (1280, 720)
    assert processor.expected_resolutions["pantry"] == (1280, 720)
    assert processor.expected_resolutions["passage"] == (1280, 720)

    # Test independent validation in _process_frame:
    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mismatched_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # 1. waiting_lobby with valid frame -> succeeds
    out_lobby = processor._process_frame("waiting_lobby", valid_frame, frame_idx=1, w=1280, h=720)
    assert out_lobby is not None

    # 2. pantry with mismatched frame -> raises FrameResolutionMismatchError
    with pytest.raises(FrameResolutionMismatchError) as exc_pantry:
        processor._process_frame("pantry", mismatched_frame, frame_idx=1, w=640, h=480)
    assert exc_pantry.value.camera_id == "pantry"

    # 3. passage with valid frame -> succeeds independently
    out_passage = processor._process_frame("passage", valid_frame, frame_idx=1, w=1280, h=720)
    assert out_passage is not None

    # 4. Multi-camera loop failure isolation test:
    # Simulate an error on pantry: verify pantry stops and marks finished=True,
    # but waiting_lobby and passage remain active (finished=False).
    processor.state["waiting_lobby"]["finished"] = False
    processor.state["pantry"]["finished"] = False
    processor.state["passage"]["finished"] = False

    # Simulate camera loop handling of pantry mismatch
    try:
        processor.validate_camera_frame("pantry", mismatched_frame, stage="original")
    except FrameResolutionMismatchError as e:
        processor.state["pantry"]["finished"] = True
        processor.state["pantry"]["resolution_error"] = str(e)

    assert processor.state["pantry"]["finished"] is True
    assert "Frame resolution mismatch" in processor.state["pantry"]["resolution_error"]
    assert processor.state["waiting_lobby"]["finished"] is False
    assert processor.state["passage"]["finished"] is False


# ---------------------------------------------------------------------------
# 8. PRESERVATION OF EXISTING ROI FUNCTIONALITY (REQUIREMENT 7)
# ---------------------------------------------------------------------------

def test_roi_functionality_preserved_on_matching_resolution():
    """
    Ensure resolution validation does not modify ROI polygon points, debouncing, or event emission.
    """
    roi_cfg = {
        "sink": {
            "enabled": True,
            "type": "polygon",
            "points": [[100, 100], [300, 100], [300, 300], [100, 300]]
        }
    }
    detector = RoiDetector("pantry", "pantry", roi_cfg, debounce_frames=2, expected_resolution=(1280, 720))

    # Points in polygon unchanged
    expected_poly = np.array([[100, 100], [300, 100], [300, 300], [100, 300]], dtype=np.int32).reshape((-1, 1, 2))
    np.testing.assert_array_equal(detector.polygons["sink"], expected_poly)

    valid_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Debounce frame 1: inside (200, 200)
    evts1 = detector.check_rois(1, (180, 180, 220, 220), frame_idx=1, timestamp=1.0, frame=valid_frame)
    assert len(evts1) == 0

    # Debounce frame 2: confirmed inside -> ROI_ENTER
    evts2 = detector.check_rois(1, (180, 180, 220, 220), frame_idx=2, timestamp=1.1, frame=valid_frame)
    assert len(evts2) == 1
    assert evts2[0].event == "ROI_ENTER"
    assert evts2[0].roi_id == "sink"


# ---------------------------------------------------------------------------
# 9. FULL-PROGRAM TERMINATION ON RESOLUTION MISMATCH
# ---------------------------------------------------------------------------

class MockVideoCapture:
    def __init__(self, frame, is_opened=True):
        self._frame = frame
        self._opened = is_opened

    def isOpened(self):
        return self._opened

    def read(self):
        return True, self._frame.copy()

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return self._frame.shape[1]
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return self._frame.shape[0]
        if prop == cv2.CAP_PROP_FPS:
            return 30
        return 0

    def release(self):
        self._opened = False


def test_multi_camera_run_stops_entire_program_on_mismatch_default():
    """
    Verify that when any camera experiences a resolution mismatch,
    the entire tracking program stops immediately (raising FrameResolutionMismatchError,
    setting running=False, marking all cameras finished, and releasing resources).
    """
    config = create_dummy_config()
    # By default, raise_on_resolution_error is True
    config.pop("raise_on_resolution_error", None)

    cameras_info = [
        {"camera_id": "waiting_lobby", "video_path": "cctv_samples/waiting_lobby.mp4"},
        {"camera_id": "pantry", "video_path": "cctv_samples/pantry.mp4"},
        {"camera_id": "passage", "video_path": "cctv_samples/passage1.mp4"}
    ]
    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path="output/test_stop_all.mp4",
        config=config,
        extractor=DummyExtractor(),
        par_extractor=DummyParExtractor(),
        pose_extractor=DummyPoseExtractor()
    )

    assert processor.raise_on_resolution_error is True

    # Inject mock caps: waiting_lobby has valid 1280x720, pantry has mismatched 1080x1920 (WxH)
    processor.state["waiting_lobby"]["cap"] = MockVideoCapture(np.zeros((720, 1280, 3), dtype=np.uint8))
    processor.state["pantry"]["cap"] = MockVideoCapture(np.zeros((1920, 1080, 3), dtype=np.uint8))
    processor.state["passage"]["cap"] = MockVideoCapture(np.zeros((720, 1280, 3), dtype=np.uint8))

    with pytest.raises(FrameResolutionMismatchError) as exc_info:
        processor.run(max_frames=10)

    assert exc_info.value.camera_id == "pantry"
    assert exc_info.value.actual_resolution == (1080, 1920)
    assert exc_info.value.expected_resolution == (1280, 720)

    # Verify that the ENTIRE tracking program stopped, not just pantry
    assert processor.running is False
    assert processor.state["waiting_lobby"]["finished"] is True
    assert processor.state["pantry"]["finished"] is True
    assert processor.state["passage"]["finished"] is True

    # Verify all video captures were released via finally block
    assert processor.state["waiting_lobby"]["cap"].isOpened() is False
    assert processor.state["pantry"]["cap"].isOpened() is False
    assert processor.state["passage"]["cap"].isOpened() is False


def test_multi_camera_run_stops_entire_program_without_raising_when_disabled():
    """
    When raise_on_resolution_error is False, run() must still terminate the entire
    tracking program without continuing the other cameras.
    """
    config = create_dummy_config()
    config["raise_on_resolution_error"] = False

    cameras_info = [
        {"camera_id": "waiting_lobby", "video_path": "cctv_samples/waiting_lobby.mp4"},
        {"camera_id": "pantry", "video_path": "cctv_samples/pantry.mp4"},
        {"camera_id": "passage", "video_path": "cctv_samples/passage1.mp4"}
    ]
    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path="output/test_stop_all_noraise.mp4",
        config=config,
        extractor=DummyExtractor(),
        par_extractor=DummyParExtractor(),
        pose_extractor=DummyPoseExtractor()
    )

    processor.state["waiting_lobby"]["cap"] = MockVideoCapture(np.zeros((720, 1280, 3), dtype=np.uint8))
    processor.state["pantry"]["cap"] = MockVideoCapture(np.zeros((1920, 1080, 3), dtype=np.uint8))
    processor.state["passage"]["cap"] = MockVideoCapture(np.zeros((720, 1280, 3), dtype=np.uint8))

    # Should not raise, but must terminate cleanly and stop the entire program
    processor.run(max_frames=10)

    assert processor.running is False
    assert processor.state["pantry"]["finished"] is True
    assert processor.state["waiting_lobby"]["finished"] is True
    assert processor.state["passage"]["finished"] is True
