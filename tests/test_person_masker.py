import pytest
import numpy as np
from app.reid.masker import PersonMasker

def test_person_masker_fallback_when_disabled():
    masker = PersonMasker(enabled=False)
    assert not masker.is_enabled()
    crop = np.full((100, 50, 3), 200, dtype=np.uint8)
    masked_crop, was_masked = masker.get_masked_crop(crop, "cam1", 1, 0)
    assert not was_masked
    assert np.array_equal(crop, masked_crop)

def test_person_masker_invalid_crop():
    masker = PersonMasker(enabled=True, model_path="models/yolo11n-seg.pt")
    # Empty crop
    empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
    out, masked = masker.get_masked_crop(empty_crop, "cam1", 1, 0)
    assert not masked

    # Tiny crop
    tiny_crop = np.zeros((10, 5, 3), dtype=np.uint8)
    out, masked = masker.get_masked_crop(tiny_crop, "cam1", 1, 0)
    assert not masked
    assert np.array_equal(tiny_crop, out)

def test_person_masker_on_person_crop():
    masker = PersonMasker(
        model_path="models/yolo11n-seg.pt",
        enabled=True,
        imgsz=160,
        bg_fill=(128, 128, 128),
        mask_reuse_frames=6
    )
    if not masker.is_enabled():
        pytest.skip("yolo11n-seg.pt not available")

    # Synthetic realistic person crop
    crop = np.full((200, 100, 3), 220, dtype=np.uint8) # background
    # Add a person shape in center
    crop[30:170, 30:70] = [30, 40, 150] # body

    masked_crop, was_masked = masker.get_masked_crop(crop, "cam1", 1, frame_idx=0)
    assert masked_crop.shape == crop.shape
    assert masked_crop.dtype == np.uint8

    # Test cache reuse on frame_idx = 3 (within mask_reuse_frames=6)
    masked_crop_2, was_masked_2 = masker.get_masked_crop(crop, "cam1", 1, frame_idx=3)
    assert masked_crop_2.shape == crop.shape

    # Reset track
    masker.reset_track("cam1", 1)
    assert ("cam1", 1) not in masker._mask_cache
