import os
import torch
import numpy as np
from torchreid.reid.utils import FeatureExtractor

class OSNetExtractor:
    def __init__(self, model_name="osnet_x1_0", model_path="", device="cpu"):
        if os.path.isfile(model_name):
            model_path = model_name
            model_name = "osnet_x1_0"
        elif not model_path and os.path.isfile("models/osnet_x1_0.pth"):
            model_path = "models/osnet_x1_0.pth"
            model_name = "osnet_x1_0"

        self.extractor = FeatureExtractor(
            model_name=model_name,
            model_path=model_path,
            device=device
        )
        
    def extract(self, crops):
        """
        Extract features from a list of images.
        Returns L2-normalized numpy arrays.
        """
        if not crops:
            return []
            
        features = self.extractor(crops)
        
        normalized_features = []
        for feat in features:
            if isinstance(feat, torch.Tensor):
                feat = feat.cpu().numpy()
            
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm
            normalized_features.append(feat)
            
        return normalized_features
