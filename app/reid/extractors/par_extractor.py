import os
import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
from app.utils.similarity import cosine_similarity

try:
    import torch
except ImportError:
    torch = None

class ParExtractor:
    """
    Pedestrian Attribute Recognition (PAR) and Visual Attribute Extractor (Phase 6).
    Extracts visual attributes:
      - Upper-body garment color (shirt/jacket)
      - Lower-body garment color (pants/skirt)
      - Deep appearance features (MobileNet PAR)
    Compares attributes with graceful handling of uncertain/missing attributes.
    """
    def __init__(self, model_path: str = "models/mobilenet_par.pt", device: str = "cpu"):
        self.device = device
        self.model = None
        self.is_onnx = False
        
        # Color definitions in HSV
        self.color_ranges = [
            ("black", 0, 180, 0, 255, 0, 55),
            ("white", 0, 180, 0, 40, 195, 255),
            ("grey", 0, 180, 0, 45, 55, 195),
            ("blue", 85, 135, 45, 255, 40, 255),
            ("green", 35, 85, 45, 255, 40, 255),
            ("red_1", 0, 10, 45, 255, 50, 255),
            ("red_2", 168, 180, 45, 255, 50, 255),
            ("orange_brown", 11, 25, 45, 255, 40, 255),
            ("yellow", 26, 35, 45, 255, 50, 255),
            ("purple", 136, 167, 45, 255, 40, 255)
        ]

        if not os.path.exists(model_path):
            alt_path = model_path.replace('.pt', '.onnx') if '.pt' in model_path else model_path.replace('.onnx', '.pt')
            if os.path.exists(alt_path):
                model_path = alt_path

        if os.path.exists(model_path) and torch is not None:
            try:
                if model_path.endswith('.onnx'):
                    import onnxruntime as ort
                    self.is_onnx = True
                    self.model = ort.InferenceSession(model_path)
                else:
                    self.model = torch.jit.load(model_path, map_location=device)
                    self.model.eval()
            except Exception as e:
                logging.error(f"Failed to load PAR model from {model_path}: {e}")
                self.model = None
        else:
            logging.warning(f"PAR model not found at {model_path}. Using color-attribute extractor.")

    def preprocess(self, crop: np.ndarray) -> np.ndarray:
        img = cv2.resize(crop, (128, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img -= np.array([0.485, 0.456, 0.406], dtype=np.float32)
        img /= np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def extract(self, crop: np.ndarray) -> np.ndarray:
        """
        Legacy extractor compatibility: returns normalized 256-d feature vector.
        """
        if crop is None or crop.size == 0:
            return np.zeros(256, dtype=np.float32)

        if self.model is None:
            # Color histogram fallback
            h, w = crop.shape[:2]
            top = crop[:h//2, :]
            bottom = crop[h//2:, :]
            top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            bottom_hsv = cv2.cvtColor(bottom, cv2.COLOR_BGR2HSV)
            top_hist = cv2.calcHist([top_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            bottom_hist = cv2.calcHist([bottom_hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
            cv2.normalize(top_hist, top_hist)
            cv2.normalize(bottom_hist, bottom_hist)
            features = np.concatenate((top_hist.flatten(), bottom_hist.flatten()))
            return features.astype(np.float32)

        try:
            input_data = self.preprocess(crop)
            if self.is_onnx:
                input_name = self.model.get_inputs()[0].name
                features = self.model.run(None, {input_name: input_data})[0]
            else:
                with torch.no_grad():
                    tensor = torch.from_numpy(input_data).to(self.device)
                    features = self.model(tensor).cpu().numpy()

            features = features.flatten()
            norm = np.linalg.norm(features)
            if norm > 1e-8:
                features = features / norm
            return features.astype(np.float32)
        except Exception as e:
            logging.debug(f"PAR extract error: {e}")
            return np.zeros(256, dtype=np.float32)

    def extract_attributes(self, crop: np.ndarray) -> Dict[str, Any]:
        """
        Extracts stable visual attributes from person crop (Phase 6):
          - Upper-body garment dominant color
          - Lower-body garment dominant color
          - Feature vector
        """
        if crop is None or crop.size == 0:
            return {
                "upper_color": "unknown",
                "lower_color": "unknown",
                "features": None,
                "confidence": 0.0
            }

        h, w = crop.shape[:2]
        # Torso / shirt area: 15% to 55% of height, center 70% of width
        y_torso_1, y_torso_2 = int(0.15 * h), int(0.55 * h)
        x_pad = int(0.15 * w)
        torso_crop = crop[y_torso_1:y_torso_2, x_pad:w - x_pad]

        # Legs / pants area: 55% to 90% of height, center 70% of width
        y_legs_1, y_legs_2 = int(0.55 * h), int(0.90 * h)
        legs_crop = crop[y_legs_1:y_legs_2, x_pad:w - x_pad]

        upper_color, upper_conf = self._detect_dominant_color(torso_crop)
        lower_color, lower_conf = self._detect_dominant_color(legs_crop)
        feat = self.extract(crop)

        return {
            "upper_color": upper_color,
            "lower_color": lower_color,
            "upper_conf": upper_conf,
            "lower_conf": lower_conf,
            "features": feat,
            "confidence": (upper_conf + lower_conf) / 2.0
        }

    def _detect_dominant_color(self, sub_crop: np.ndarray) -> Tuple[str, float]:
        if sub_crop is None or sub_crop.size == 0:
            return "unknown", 0.0

        try:
            hsv = cv2.cvtColor(sub_crop, cv2.COLOR_BGR2HSV)
            total_pixels = hsv.shape[0] * hsv.shape[1]
            if total_pixels == 0:
                return "unknown", 0.0

            color_counts = {}
            for name, h_min, h_max, s_min, s_max, v_min, v_max in self.color_ranges:
                mask = cv2.inRange(hsv, (h_min, s_min, v_min), (h_max, s_max, v_max))
                count = int(cv2.countNonZero(mask))
                unified_name = "red" if name.startswith("red") else name
                color_counts[unified_name] = color_counts.get(unified_name, 0) + count

            if not color_counts:
                return "unknown", 0.0

            best_color, best_count = max(color_counts.items(), key=lambda x: x[1])
            ratio = best_count / float(total_pixels)
            if ratio < 0.20:
                return "unknown", 0.30

            return best_color, min(1.0, ratio * 1.5)
        except Exception:
            return "unknown", 0.0

    @staticmethod
    def compare_attributes(attr_a: Dict[str, Any], attr_b: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Compares two visual attribute profiles (Phase 6 & Phase 10).
        Missing or uncertain attributes do NOT penalize as mismatches.
        """
        def _color_similarity(c1: str, c2: str) -> float:
            if not c1 or not c2 or c1 == "unknown" or c2 == "unknown":
                return 0.70 # Neutral for unknown

            if c1 == c2:
                return 1.0

            # Similar color families
            similar_groups = [
                {"black", "grey", "dark"},
                {"white", "grey", "light"},
                {"blue", "black"},
                {"orange_brown", "yellow"}
            ]
            for g in similar_groups:
                if c1 in g and c2 in g:
                    return 0.85

            return 0.15 # Definite mismatch

        upper_sim = _color_similarity(attr_a.get("upper_color", "unknown"), attr_b.get("upper_color", "unknown"))
        lower_sim = _color_similarity(attr_a.get("lower_color", "unknown"), attr_b.get("lower_color", "unknown"))

        feat_sim = 0.70
        feat_a = attr_a.get("features")
        feat_b = attr_b.get("features")
        if feat_a is not None and feat_b is not None:
            fa = np.asarray(feat_a, dtype=np.float32).flatten()
            fb = np.asarray(feat_b, dtype=np.float32).flatten()
            if len(fa) > 0 and len(fb) > 0:
                feat_sim = max(0.0, float(cosine_similarity(fa, fb)))

        # Weighted combination: 45% upper color + 45% lower color + 10% deep feature
        par_score = 0.45 * upper_sim + 0.45 * lower_sim + 0.10 * feat_sim
        components = {
            "upper_similarity": upper_sim,
            "lower_similarity": lower_sim,
            "feature_similarity": feat_sim,
            "par_total": par_score
        }
        return float(par_score), components
