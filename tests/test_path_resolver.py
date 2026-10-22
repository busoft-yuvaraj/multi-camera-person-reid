import os
import pytest
from app.utils.path_resolver import resolve_video_path, resolve_output_path

def test_resolve_video_path_existing_relative():
    # An empty or nonexistent path without D: match returns original
    assert resolve_video_path("") == ""
    assert resolve_video_path("nonexistent_path_xyz123.mp4") == "nonexistent_path_xyz123.mp4"

def test_resolve_video_path_d_drive():
    # If cctv_samples is on D:, resolving cctv_samples/pantry.mp4 should find it on D:
    resolved = resolve_video_path("cctv_samples/pantry.mp4")
    if os.path.exists(r"D:\cctv_samples\pantry.mp4"):
        assert "D:" in resolved or "d:" in resolved
        assert os.path.exists(resolved)

def test_resolve_video_path_test8():
    resolved = resolve_video_path("cctv_samples/test8/waitinglobby.mp4")
    if os.path.exists(r"D:\cctv_samples\test8\waitinglobby.mp4"):
        assert "D:" in resolved or "d:" in resolved
        assert os.path.exists(resolved)

def test_resolve_output_path():
    # Relative path starting with output should point to D: if D: exists
    if os.path.exists(r"D:\output") or os.path.exists(r"D:\\"):
        out = resolve_output_path("output/combined_output_test.mp4")
        assert out.startswith("D:/output") or out.startswith("D:\\output")
        
        # Absolute custom temp path should remain untouched
        temp_path = os.path.abspath("temp_dir_xyz")
        assert resolve_output_path(temp_path) == temp_path
