import os
import cv2
import numpy as np
import logging
from ultralytics import YOLO

logger = logging.getLogger("reid.attire_extractor")

class AttireExtractor:
    def __init__(self, model_path="models/formal_attire_best.pt", conf=0.25, device="cpu"):
        self.device = device
        self.conf = conf
        self.model = None

        if not os.path.exists(model_path):
            logger.warning(f"Formal attire model not found at '{model_path}'. Attire detection will be disabled.")
            return

        try:
            logger.info(f"Loading Formal Attire model from {model_path} on {device}...")
            self.model = YOLO(model_path)
            self.class_names = self.model.names
            logger.info(f"Formal Attire classes loaded: {self.class_names}")
        except Exception as e:
            logger.error(f"Failed to load formal attire model: {e}")
            self.model = None

    def is_available(self) -> bool:
        return self.model is not None

    def extract_attire(self, crop: np.ndarray) -> dict:
        """
        Runs attire detection on a person crop.
        Returns:
            {
                "label": "formal_dress" | "informal_dress" | "unknown",
                "conf": float,
                "is_formal": bool | None
            }
        """
        if not self.is_available() or crop is None or crop.size == 0:
            return {"label": "unknown", "conf": 0.0, "is_formal": None}

        # Check minimal size
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return {"label": "unknown", "conf": 0.0, "is_formal": None}

        try:
            results = self.model.predict(
                source=crop,
                conf=self.conf,
                device=self.device,
                verbose=False
            )

            if not results or not results[0].boxes or len(results[0].boxes) == 0:
                return {"label": "unknown", "conf": 0.0, "is_formal": None}

            boxes = results[0].boxes
            best_idx = int(boxes.conf.argmax())
            cls_id = int(boxes.cls[best_idx].item())
            conf_val = float(boxes.conf[best_idx].item())
            label = self.class_names.get(cls_id, str(cls_id))

            is_formal = "formal" in label.lower() and "informal" not in label.lower()

            return {
                "label": label,
                "conf": round(conf_val, 2),
                "is_formal": is_formal
            }
        except Exception as e:
            logger.debug(f"Attire inference failed on crop: {e}")
            return {"label": "unknown", "conf": 0.0, "is_formal": None}
