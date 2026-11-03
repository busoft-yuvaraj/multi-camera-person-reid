"""
CLI Runner for Single-Camera Physical Calibration & Tracking Demonstration
Usage:
  python tools/run_single_camera_test.py --camera waiting_lobby --max_frames 150
  python tools/run_single_camera_test.py --camera passage1 --max_frames 150
  python tools/run_single_camera_test.py --camera pantry --max_frames 150
"""

import os
import sys
import argparse
import logging

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.calibration.single_camera_processor import SingleCameraCalibrationProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SingleCamTest")


def main():
    parser = argparse.ArgumentParser(description="Run single-camera physical calibration tracking test")
    parser.add_argument("--camera", type=str, default="passage1",
                        choices=["passage1", "pantry", "waiting_lobby"],
                        help="Camera ID to test")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to CCTV video (defaults to cctv_samples/{camera}.mp4)")
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Start frame offset in video")
    parser.add_argument("--max_frames", type=int, default=150,
                        help="Maximum frames to process (None for full video)")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold for YOLO11")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Tracker config (bytetrack.yaml or botsort.yaml)")

    args = parser.parse_args()

    video_path = args.video
    if not video_path:
        video_path = f"cctv_samples/{args.camera}.mp4"

    assert os.path.exists(video_path), f"Video file not found: {video_path}"

    print("=" * 78)
    print("PHASE 2: SINGLE-CAMERA VIDEO CALIBRATION TEST")
    print("=" * 78)
    print(f"Camera ID:        {args.camera}")
    print(f"Video Path:       {video_path}")
    print(f"Start Frame:      {args.start_frame}")
    print(f"Max Frames:       {args.max_frames}")
    print(f"YOLO11 Conf:      {args.conf}")
    print(f"Tracker:          {args.tracker}")
    print("-" * 78)

    processor = SingleCameraCalibrationProcessor(
        config_path="config/camera_calibration.yaml",
        model_name="yolo11n.pt",
        output_dir="calibration_result",
        logs_dir="outputs"
    )

    results = processor.process_video(
        camera_id=args.camera,
        video_path=video_path,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        conf_threshold=args.conf,
        save_videos=True
    )

    print("\n" + "=" * 78)
    print("SINGLE-CAMERA CALIBRATION & TRACKING EXECUTION REPORT")
    print("=" * 78)
    print(f"  Camera ID:              {results['camera_id']}")
    print(f"  Frames Processed:       {results['frames_processed']}")
    print(f"  Elapsed Time:           {results['elapsed_seconds']} s")
    print(f"  Processing Speed:       {results['processing_fps']} FPS")
    print(f"  Unique Tracks Detected: {results['total_tracks']}")
    print(f"  Observations Logged:    {results['total_observations']}")
    print("\n  Exported Artifacts:")
    print(f"    - Processed CCTV:     {results['output_videos']['live_cctv']}")
    print(f"    - Ground-Plane Video: {results['output_videos']['ground_plane']}")
    print(f"    - Dual Synchronized:  {results['output_videos']['synchronized_dual']}")
    print(f"    - Tracking Logs:      {results['output_logs']['tracking_logs']}")
    print(f"    - Calibration Meta:   {results['output_logs']['calibration_metadata']}")
    print("=" * 78)


if __name__ == "__main__":
    main()
