"""
FormalAttireDetector: loads the trained YOLO weights (best.pt) and runs
frame-by-frame detection + ByteTrack tracking via ultralytics' built-in
`model.track(...)` API, driven by the bytetrack.yaml config.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from ultralytics import YOLO

from .utils import resolve_path

logger = logging.getLogger("formal_attire_detection")


class FormalAttireDetector:
    def __init__(self, model_cfg: Dict[str, Any], tracker_cfg: Dict[str, Any]):
        weights_path = resolve_path(model_cfg["weights_path"])
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}. "
                "Place your trained best.pt there or update config/config.yaml."
            )

        self.conf = model_cfg.get("confidence_threshold", 0.5)
        self.iou = model_cfg.get("iou_threshold", 0.45)
        self.device = model_cfg.get("device", "cpu")

        self.tracker_yaml = str(resolve_path(tracker_cfg["config_path"]))
        self.persist = tracker_cfg.get("persist", True)

        logger.info("Loading model weights from %s", weights_path)
        self.model = YOLO(str(weights_path))
        self.class_names = self.model.names

    def track(self, frame):
        """Run detection + ByteTrack tracking on a single BGR frame.

        Returns the ultralytics Results object for the frame (results[0]),
        which carries boxes, track IDs, confidences and class ids.
        """
        results = self.model.track(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            tracker=self.tracker_yaml,
            persist=self.persist,
            verbose=False,
        )
        return results[0]

    def annotate(self, frame, result):
        """Draw ultralytics' built-in box/label/track-id annotations
        onto the frame and return the annotated image."""
        return result.plot()

    def get_class_name(self, class_id: int) -> str:
        return self.class_names.get(int(class_id), str(class_id))
