import cv2
import numpy as np

def calculate_quality(crop, conf, min_conf, min_width, min_height, blur_thresh, frame_width, frame_height, bbox):
    """
    Calculate the ReID quality score based on confidence, size, sharpness, and boundary constraints.
    Returns (quality_score, is_accepted, reason, metadata)
    """
    h, w = crop.shape[:2]
    
    if conf < min_conf:
        return 0.0, False, "low_confidence", {}
        
    if w < min_width or h < min_height:
        return 0.0, False, "too_small", {}
        
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < blur_thresh:
        return 0.0, False, "too_blurry", {}
        
    x1, y1, x2, y2 = bbox
    boundary_score = 1.0
    if x1 <= 0 or y1 <= 0 or x2 >= frame_width or y2 >= frame_height:
        boundary_score = 0.5
        
    sharpness_score = min(blur_score / (blur_thresh * 5), 1.0)
    size_score = min((w * h) / (min_width * min_height * 4), 1.0)
    
    # Estimate lighting (brightness)
    lighting_score = np.mean(gray) / 255.0
    
    quality = (0.40 * conf) + (0.25 * size_score) + (0.20 * sharpness_score) + (0.15 * boundary_score)
    
    metadata = {
        "confidence": float(conf),
        "sharpness": float(sharpness_score),
        "visibility": float(boundary_score),
        "lighting": float(lighting_score)
    }
    
    return quality, True, "good", metadata
