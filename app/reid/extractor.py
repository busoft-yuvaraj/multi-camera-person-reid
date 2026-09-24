import torch
import numpy as np
from torchreid.reid.utils import FeatureExtractor

class OSNetExtractor:
    def __init__(self, model_name="osnet_x1_0", device="cpu"):
        self.extractor = FeatureExtractor(
            model_name=model_name,
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
