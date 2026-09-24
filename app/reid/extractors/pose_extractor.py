import os
import cv2
import math
import numpy as np
from ultralytics import YOLO

class PoseExtractor:
    def __init__(self, model_path="models/yolo11n-pose.pt", device="cpu"):
        # Fallback to .pt if .onnx requested but missing, or vice versa
        if not os.path.exists(model_path):
            alt_path = model_path.replace('.pt', '.onnx') if '.pt' in model_path else model_path.replace('.onnx', '.pt')
            if os.path.exists(alt_path):
                model_path = alt_path
                
        self.model = YOLO(model_path)
        self.device = device
        
    def extract_viewpoint(self, crop):
        """
        Takes a BGR image crop of a person.
        Returns:
            viewpoint (str): 'FRONT', 'BACK', 'SIDE', or 'UNKNOWN'
            confidence (float): confidence of the viewpoint estimation
        """
        if crop is None or crop.size == 0:
            return "UNKNOWN", 0.0
            
        results = self.model(crop, verbose=False, device=self.device)
        
        if not results or not results[0].keypoints or len(results[0].keypoints) == 0:
            return "UNKNOWN", 0.0
            
        # Get keypoints for the first person detected in the crop
        # YOLOv8-pose keypoints shape: (1, 17, 3) -> (num_persons, num_keypoints, [x, y, conf])
        keypoints = results[0].keypoints.data[0].cpu().numpy()
        
        if len(keypoints) < 17:
            return "UNKNOWN", 0.0
            
        # Keypoint indices for COCO format:
        # 0: Nose, 1: L-Eye, 2: R-Eye, 3: L-Ear, 4: R-Ear
        # 5: L-Shoulder, 6: R-Shoulder
        # 11: L-Hip, 12: R-Hip
        
        nose_conf = keypoints[0][2]
        l_eye_conf = keypoints[1][2]
        r_eye_conf = keypoints[2][2]
        
        l_shoulder = keypoints[5]
        r_shoulder = keypoints[6]
        
        l_hip = keypoints[11]
        r_hip = keypoints[12]
        
        # Calculate Shoulder Width
        shoulder_dist = math.hypot(l_shoulder[0] - r_shoulder[0], l_shoulder[1] - r_shoulder[1])
        
        # Calculate Torso Height (average of L/R shoulder to hip)
        l_torso = math.hypot(l_shoulder[0] - l_hip[0], l_shoulder[1] - l_hip[1])
        r_torso = math.hypot(r_shoulder[0] - r_hip[0], r_shoulder[1] - r_hip[1])
        torso_height = (l_torso + r_torso) / 2.0
        
        # If torso height is 0 (bad detection), return UNKNOWN
        if torso_height < 1e-5:
            return "UNKNOWN", 0.0
            
        ratio = shoulder_dist / torso_height
        
        # If shoulders are very close together relative to torso height, it's likely a side profile
        if ratio < 0.35:
            return "SIDE", 0.8
            
        # If shoulders are wide, it's FRONT or BACK.
        # Check facial keypoints to determine FRONT vs BACK.
        # If nose and eyes are visible (confidence > 0.5), they are facing the camera.
        face_visible = nose_conf > 0.5 or (l_eye_conf > 0.5 and r_eye_conf > 0.5)
        
        if face_visible:
            return "FRONT", max(nose_conf, l_eye_conf, r_eye_conf)
        else:
            return "BACK", 0.8 # We assume back if face isn't visible but shoulders are wide
