import os
import yaml
import pytest
import cv2
import numpy as np

from app.multi_camera_processor import MultiCameraProcessor
from app.reid.global_gallery import GlobalGallery
from app.reid.matcher import GlobalMatcher
from app.reid.extractors.attire_extractor import AttireExtractor

class DummyExtractor:
    def extract(self, crops):
        return [np.zeros(512, dtype=np.float32) for _ in crops]

class DummyParExtractor:
    def extract(self, crop):
        return np.zeros(35, dtype=np.float32)

class DummyPoseExtractor:
    def extract_viewpoint(self, crop):
        return "FRONT", 0.9

def test_three_cameras_initialization_and_frames():
    with open("config/app_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Read camera setup from Phase 1 cameras config or legacy violation_detection
    if "cameras" in config:
        cams = config["cameras"]
        cameras_info = [
            {"camera_id": "waiting_lobby", "video_path": cams.get("waiting_lobby", {}).get("video_path", "cctv_samples/waiting_lobby.mp4")},
            {"camera_id": "pantry", "video_path": cams.get("pantry", {}).get("video_path", "cctv_samples/pantry.mp4")},
            {"camera_id": "passage", "video_path": cams.get("passage", {}).get("video_path", "cctv_samples/passage1.mp4")}
        ]
    else:
        v_cfg = config.get("violation_detection", {})
        cams = v_cfg.get("cameras", {})
        cameras_info = [
            {"camera_id": "waiting_lobby", "video_path": cams["waiting_lobby"]["video_path"]},
            {"camera_id": "pantry", "video_path": cams["pantry"]["video_path"]},
            {"camera_id": "passage_1", "video_path": cams.get("passage_1", {}).get("video_path", "cctv_samples/passage1.mp4")}
        ]

    # Verify video files exist
    for c in cameras_info:
        assert os.path.exists(c["video_path"]), f"Video not found: {c['video_path']}"

    gallery = GlobalGallery(storage_path="tests_gallery.pkl", clear_on_start=True)
    matcher = GlobalMatcher(gallery=gallery, config=config)

    attire_extractor = AttireExtractor(
        model_path=config.get("attire_model_path", "models/formal_attire_best.pt"),
        conf=0.25,
        device="cpu"
    )

    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path="output/test_combined_3cams.mp4",
        config=config,
        extractor=DummyExtractor(),
        par_extractor=DummyParExtractor(),
        pose_extractor=DummyPoseExtractor(),
        attire_extractor=attire_extractor,
        gallery=gallery,
        matcher=matcher
    )

    # 1. Check independent state per camera
    assert len(processor.state) == 3
    for cam in cameras_info:
        c_id = cam["camera_id"]
        assert c_id in processor.state
        assert processor.state[c_id]["cap"].isOpened()
        assert processor.state[c_id]["model"] is not None

    # 2. Check transition detectors / line detectors
    assert "waiting_lobby" in processor.transition_detectors
    assert len(processor.transition_detectors["waiting_lobby"].lines) > 0

    # 3. Process 3 frames across all 3 cameras
    w = 1280
    h = 720
    for frame_idx in range(3):
        display_frames = []
        for cam in cameras_info:
            c_id = cam["camera_id"]
            st = processor.state[c_id]
            ret, frame = st["cap"].read()
            assert ret, f"Could not read frame from {c_id}"
            processed = processor._process_frame(c_id, frame, frame_idx, w, h)
            assert processed is not None
            assert processed.shape == (720, 1280, 3)
            display_frames.append(processed)

        # 4. Check horizontal stack across 3 cameras (3840 x 720)
        combined = np.hstack(display_frames)
        assert combined.shape == (720, 3840, 3)

    # Clean up captures
    for st in processor.state.values():
        st["cap"].release()
    if os.path.exists("tests_gallery.pkl"):
        os.remove("tests_gallery.pkl")
