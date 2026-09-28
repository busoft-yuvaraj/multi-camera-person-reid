"""
VideoStream: thin wrapper around cv2.VideoCapture / cv2.VideoWriter so
main.py doesn't need to know about camera indices vs. file paths vs.
RTSP URLs, and so the output writer is only created once we know the
real frame size coming off the source.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

logger = logging.getLogger("formal_attire_detection")


class VideoStream:
    def __init__(self, video_cfg: Dict[str, Any]):
        self.source = video_cfg.get("source", 0)
        self.req_width = video_cfg.get("frame_width", 1280)
        self.req_height = video_cfg.get("frame_height", 720)
        self.req_fps = video_cfg.get("fps", 30)
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.source)
        if self.req_width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
        if self.req_height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        if self.req_fps:
            self.cap.set(cv2.CAP_PROP_FPS, self.req_fps)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.source!r}")

        logger.info("Video source opened: %s", self.source)

    def read(self):
        if self.cap is None:
            raise RuntimeError("VideoStream.open() must be called before read()")
        return self.cap.read()

    def get_frame_size(self) -> Tuple[int, int]:
        """Return the ACTUAL (width, height) the source is delivering,
        which may differ from the requested size."""
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.req_width
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.req_height
        return width, height

    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 1 else float(self.req_fps or 30)

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            logger.info("Video source released")


class OutputWriter:
    """Wraps cv2.VideoWriter, saving the annotated stream into the
    configured output directory."""

    def __init__(self, output_cfg: Dict[str, Any], frame_size: Tuple[int, int], fps: float):
        self.enabled = output_cfg.get("save_video", True)
        self.save_dir = Path(output_cfg.get("save_dir", "outputs"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filepath = self.save_dir / output_cfg.get("video_filename", "output.mp4")
        self.writer: Optional[cv2.VideoWriter] = None

        if self.enabled:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.filepath), fourcc, fps, frame_size)
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open output video writer at {self.filepath}")
            logger.info("Saving annotated output to: %s", self.filepath)

    def write(self, frame) -> None:
        if self.enabled and self.writer is not None:
            self.writer.write(frame)

    def release(self) -> None:
        if self.writer is not None:
            self.writer.release()
