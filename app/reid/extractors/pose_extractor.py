import os
import cv2
import math
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class PoseExtractor:
    """
    Pose and Viewpoint Estimation Layer (Phase 4 & Phase 5).
    Estimates viewpoint context (FRONT, SIDE, BACK, UNKNOWN) and observation difficulty.
    Does NOT predict identity.
    """
    def __init__(self, model_path: str = "models/yolo11n-pose.pt", device: str = "cpu"):
        self.device = device
        self.model = None
        
        if not os.path.exists(model_path):
            alt_path = model_path.replace('.pt', '.onnx') if '.pt' in model_path else model_path.replace('.onnx', '.pt')
            if os.path.exists(alt_path):
                model_path = alt_path
                
        if os.path.exists(model_path) and YOLO is not None:
            try:
                self.model = YOLO(model_path)
            except Exception as e:
                logging.error(f"Failed to load YOLO pose model from {model_path}: {e}")
                self.model = None
        else:
            logging.warning(f"Pose model not found at {model_path}. Operating in fallback mode.")

    def extract_viewpoint(self, crop: np.ndarray) -> Tuple[str, float]:
        """
        Takes a BGR image crop of a person.
        Returns:
            viewpoint (str): 'FRONT', 'BACK', 'SIDE', or 'UNKNOWN'
            confidence (float): confidence of viewpoint estimation [0.0, 1.0]
        """
        if crop is None or crop.size == 0 or self.model is None:
            return "UNKNOWN", 0.0
            
        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return "UNKNOWN", 0.0

        try:
            results = self.model(crop, verbose=False, device=self.device)
            if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
                return "UNKNOWN", 0.0
                
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            if len(keypoints) < 17:
                return "UNKNOWN", 0.0
        except Exception as e:
            logging.debug(f"Pose estimation error: {e}")
            return "UNKNOWN", 0.0

        # COCO 17 Keypoints:
        # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        # 5: L-Shoulder, 6: R-Shoulder
        # 11: L-Hip, 12: R-Hip
        nose_conf = float(keypoints[0][2])
        l_eye_conf = float(keypoints[1][2])
        r_eye_conf = float(keypoints[2][2])
        l_ear_conf = float(keypoints[3][2])
        r_ear_conf = float(keypoints[4][2])

        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        l_hip = keypoints[11]
        r_hip = keypoints[12]

        # Check shoulder keypoint validity
        if l_shoulder[2] < 0.20 and r_shoulder[2] < 0.20:
            return "UNKNOWN", 0.0

        shoulder_dist = math.hypot(l_shoulder[0] - r_shoulder[0], l_shoulder[1] - r_shoulder[1])
        l_torso = math.hypot(l_shoulder[0] - l_hip[0], l_shoulder[1] - l_hip[1])
        r_torso = math.hypot(r_shoulder[0] - r_hip[0], r_shoulder[1] - r_hip[1])
        torso_height = (l_torso + r_torso) / 2.0

        if torso_height < 12.0:
            return "UNKNOWN", 0.0

        ratio = shoulder_dist / max(torso_height, 1e-4)

        # Profile / Side view: narrow shoulder profile relative to height
        # Or one ear/eye visible while opposite side is completely hidden
        one_side_dominant = (
            (l_ear_conf > 0.40 and r_ear_conf < 0.15) or
            (r_ear_conf > 0.40 and l_ear_conf < 0.15)
        )
        if ratio < 0.38 or (ratio < 0.48 and one_side_dominant):
            vp_conf = min(0.92, max(0.70, 0.90 - ratio * 0.3))
            return "SIDE", float(vp_conf)

        # Broad shoulder view -> FRONT or BACK
        # Check facial keypoints
        face_conf = max(nose_conf, l_eye_conf, r_eye_conf)
        both_eyes_visible = (l_eye_conf > 0.35 and r_eye_conf > 0.35)

        if face_conf >= 0.45 or both_eyes_visible:
            conf = min(0.98, max(0.75, face_conf + 0.15))
            return "FRONT", float(conf)
        else:
            # Shoulders are wide, but face features are absent -> person is viewed from BACK
            back_conf = min(0.95, max(0.75, (1.0 - face_conf) * 0.9))
            return "BACK", float(back_conf)

    @staticmethod
    def is_cross_view(view_a: str, view_b: str) -> bool:
        """Returns True if the two views are different and neither is UNKNOWN."""
        va = view_a.upper() if view_a else "UNKNOWN"
        vb = view_b.upper() if view_b else "UNKNOWN"
        if va == "UNKNOWN" or vb == "UNKNOWN":
            return False
        return va != vb

    @staticmethod
    def view_difficulty(view_a: str, view_b: str) -> float:
        """
        Quantifies viewpoint difficulty factor between two observations (Phase 5).
        Returns:
            1.0: Same view (FRONT-FRONT, BACK-BACK, SIDE-SIDE) -> easiest
            0.85: Adjacent view (FRONT-SIDE, SIDE-BACK) -> medium difficulty
            0.65: Opposite view (FRONT-BACK) -> hard cross-view
            0.80: UNKNOWN involved -> neutral tolerance
        """
        va = view_a.upper() if view_a else "UNKNOWN"
        vb = view_b.upper() if view_b else "UNKNOWN"

        if va == "UNKNOWN" or vb == "UNKNOWN":
            return 0.80

        if va == vb:
            return 1.0

        if (va == "FRONT" and vb == "BACK") or (va == "BACK" and vb == "FRONT"):
            return 0.65

        # All other combinations are adjacent (FRONT-SIDE, SIDE-BACK)
        return 0.85
