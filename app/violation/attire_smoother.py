from collections import deque
from typing import Dict, List, Optional, Any
import numpy as np

class AttireStabilityManager:
    """
    Maintains a rolling temporal window of attire predictions per track to ensure stability.
    Applies confidence-weighted voting over the history window to prevent single-frame flickering.
    """
    def __init__(self, history_size: int = 10, minimum_confidence: float = 0.50, prediction_interval: int = 5):
        self.history_size = max(1, history_size)
        self.minimum_confidence = minimum_confidence
        self.prediction_interval = max(1, prediction_interval)
        
        # Mapping: local_track_id -> deque of prediction dicts
        # e.g., [{"label": "formal_dress", "conf": 0.85, "is_formal": True}]
        self.history: Dict[int, deque] = {}

    def should_predict(self, frame_idx: int, local_track_id: int) -> bool:
        """
        Determines if the attire extractor should run on this frame for this track.
        Always predicts if track has no history yet, otherwise respects prediction_interval.
        """
        if local_track_id not in self.history or len(self.history[local_track_id]) == 0:
            return True
        return (frame_idx % self.prediction_interval) == 0

    def add_prediction(self, local_track_id: int, prediction: Dict[str, Any]):
        """
        Appends a prediction to the track's history window.
        Expected prediction format:
            {
                "label": str,
                "conf": float,
                "is_formal": bool | None
            }
        """
        if local_track_id not in self.history:
            self.history[local_track_id] = deque(maxlen=self.history_size)
        
        if prediction.get("label") != "unknown" and prediction.get("is_formal") is not None:
            self.history[local_track_id].append(prediction)

    def get_stable_attire(self, local_track_id: int) -> Dict[str, Any]:
        """
        Computes the stable attire classification using confidence-weighted voting.
        Returns:
            {
                "final_attire": "formal" | "informal" | "unknown",
                "confidence": float,
                "is_formal": bool | None,
                "is_confident": bool,
                "recent_predictions": List[Dict[str, Any]]
            }
        """
        if local_track_id not in self.history or len(self.history[local_track_id]) == 0:
            return {
                "final_attire": "unknown",
                "confidence": 0.0,
                "is_formal": None,
                "is_confident": False,
                "recent_predictions": []
            }

        preds = list(self.history[local_track_id])
        valid_preds = [p for p in preds if p.get("is_formal") is not None]

        if not valid_preds:
            return {
                "final_attire": "unknown",
                "confidence": 0.0,
                "is_formal": None,
                "is_confident": False,
                "recent_predictions": preds
            }

        # Confidence-weighted score summation
        formal_conf_sum = sum(p.get("conf", 0.0) for p in valid_preds if p.get("is_formal") is True)
        informal_conf_sum = sum(p.get("conf", 0.0) for p in valid_preds if p.get("is_formal") is False)

        formal_count = sum(1 for p in valid_preds if p.get("is_formal") is True)
        informal_count = sum(1 for p in valid_preds if p.get("is_formal") is False)

        total_conf = formal_conf_sum + informal_conf_sum
        if total_conf > 0:
            is_formal = formal_conf_sum >= informal_conf_sum
        else:
            is_formal = formal_count >= informal_count

        relevant_preds = [p for p in valid_preds if p.get("is_formal") is is_formal]
        avg_conf = float(np.mean([p.get("conf", 0.0) for p in relevant_preds])) if relevant_preds else 0.0

        final_attire = "formal" if is_formal else "informal"
        is_confident = avg_conf >= self.minimum_confidence

        return {
            "final_attire": final_attire,
            "confidence": round(avg_conf, 2),
            "is_formal": is_formal,
            "is_confident": is_confident,
            "recent_predictions": preds
        }

    def reset_track(self, local_track_id: int):
        self.history.pop(local_track_id, None)
