import numpy as np
from app.utils.similarity import cosine_similarity

class TrackEmbeddingBank:
    def __init__(self, camera_id, track_id, max_size_per_view=5, diversity_threshold=0.95):
        self.camera_id = camera_id
        self.track_id = track_id
        self.max_size_per_view = max_size_per_view
        self.diversity_threshold = diversity_threshold
        
        # Sub-banks for each viewpoint
        self.banks = {
            "FRONT": [],
            "BACK": [],
            "SIDE": [],
            "UNKNOWN": []
        }
        
    def add_embedding(self, embedding, quality, frame_id, viewpoint, par_features, quality_metadata):
        view_bank = self.banks.get(viewpoint, self.banks["UNKNOWN"])
        
        entry = {
            "embedding": embedding,
            "quality": quality,
            "frame_id": frame_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "viewpoint": viewpoint,
            "par_features": par_features,
            "metadata": quality_metadata
        }
        
        if not view_bank:
            view_bank.append(entry)
            return True, "stored", entry
            
        for existing in view_bank:
            sim = cosine_similarity(embedding, existing["embedding"])
            if sim >= self.diversity_threshold:
                return False, "too_similar", None
                
        if len(view_bank) >= self.max_size_per_view:
            view_bank.sort(key=lambda x: x["quality"])
            if quality > view_bank[0]["quality"]:
                view_bank[0] = entry
                return True, "replaced_low_quality", entry
            else:
                return False, "bank_full_and_low_quality", None
                
        view_bank.append(entry)
        return True, "stored", entry
        
    def get_all_embeddings(self):
        """Returns all embeddings across all viewpoints."""
        all_embs = []
        for view, bank in self.banks.items():
            all_embs.extend(bank)
        return all_embs
