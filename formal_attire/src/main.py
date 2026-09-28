"""
Entry point for the Formal Attire Detection pipeline.

Pipeline: VideoStream -> FormalAttireDetector (YOLO + ByteTrack) ->
annotated frame -> live cv2 window + saved video in outputs/.

Run via the project-root `run.py`, e.g.:
    python run.py --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import cv2

from src.detector import FormalAttireDetector
from src.utils import ensure_dir, load_config, setup_logging
from src.video_stream import OutputWriter, VideoStream


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal Attire Detection (YOLO + ByteTrack)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override the video source from config (webcam index, file path, or RTSP URL)",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live cv2 window (useful for headless/server runs)",
    )
    return parser.parse_args()


def run(config_path: str, source_override: str | None, no_display: bool) -> None:
    cfg = load_config(config_path)
    logger = setup_logging(cfg["logging"])
    logger.info("Starting Formal Attire Detection pipeline")

    ensure_dir(cfg["output"]["save_dir"])

    if source_override is not None:
        # allow numeric strings to still mean a webcam index
        cfg["video"]["source"] = int(source_override) if source_override.isdigit() else source_override

    show_live = cfg["display"].get("show_live", True) and not no_display
    window_name = cfg["display"].get("window_name", "Formal Attire Detection")
    draw_fps = cfg["display"].get("draw_fps", True)

    detector = FormalAttireDetector(cfg["model"], cfg["tracker"])
    stream = VideoStream(cfg["video"])

    try:
        stream.open()
    except RuntimeError as e:
        logger.error(str(e))
        return

    frame_size = stream.get_frame_size()
    fps = stream.get_fps()
    writer = OutputWriter(cfg["output"], frame_size, fps)

    save_frames = cfg["output"].get("save_annotated_frames", False)
    frame_save_interval = cfg["output"].get("frame_save_interval", 30)
    frames_dir = None
    if save_frames:
        frames_dir = ensure_dir(str(Path(cfg["output"]["save_dir"]) / "frames"))

    frame_count = 0
    t_prev = time.time()

    try:
        while True:
            ok, frame = stream.read()
            if not ok:
                logger.info("End of stream / cannot read frame. Exiting loop.")
                break

            result = detector.track(frame)
            annotated = detector.annotate(frame, result)

            if draw_fps:
                t_now = time.time()
                inst_fps = 1.0 / max(t_now - t_prev, 1e-6)
                t_prev = t_now
                cv2.putText(
                    annotated,
                    f"FPS: {inst_fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )

            writer.write(annotated)

            if save_frames and frames_dir is not None and frame_count % frame_save_interval == 0:
                cv2.imwrite(str(frames_dir / f"frame_{frame_count:06d}.jpg"), annotated)

            if show_live:
                cv2.imshow(window_name, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit key pressed. Stopping.")
                    break

            frame_count += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl+C).")
    finally:
        stream.release()
        writer.release()
        if show_live:
            cv2.destroyAllWindows()
        logger.info("Pipeline stopped. Processed %d frames.", frame_count)


def main() -> None:
    args = parse_args()
    run(args.config, args.source, args.no_display)


if __name__ == "__main__":
    main()
