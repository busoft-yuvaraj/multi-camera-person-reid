import time
import numpy as np
from typing import Dict, List, Any, Optional, Set
from app.utils.similarity import cosine_similarity

class GlobalIdentity:
    """
    Global Identity Profile (Phase 8).
    Maintains:
      - View-aware ReID embedding bank (FRONT, SIDE, BACK, UNKNOWN)
      - Stable visual attribute profile with temporal voting (shirt color, pants color, etc.)
      - Viewpoint observation statistics
      - Spatio-temporal context (last_camera, last_track_id, last_seen_time)
      - Identity lifecycle status: UNKNOWN, PENDING, CONFIRMED, LOST
    """
    def __init__(
        self,
        global_id: str,
        initial_camera: str = "",
        initial_track_id: int = -1,
        status: str = "CONFIRMED",
        confidence: float = 1.0,
        max_per_view: int = 5
    ):
        self.global_id = global_id
        self.status = status
        self.confidence = float(confidence)
        self.max_per_view = max_per_view
        
        # View-aware ReID embedding storage
        self.reid_bank: Dict[str, List[Dict[str, Any]]] = {
            "FRONT": [],
            "SIDE": [],
            "BACK": [],
            "UNKNOWN": []
        }
        
        # Viewpoint statistics
        self.view_statistics: Dict[str, int] = {
            "FRONT": 0,
            "SIDE": 0,
            "BACK": 0,
            "UNKNOWN": 0
        }
        
        # Attribute profile with temporal voting / counts
        self.par_profile: Dict[str, Any] = {
            "upper_color": "unknown",
            "lower_color": "unknown",
            "upper_votes": {},
            "lower_votes": {},
            "features": None, # Centroid of raw features if available
            "par_count": 0
        }
        
        # Spatio-temporal tracking
        self.last_camera: str = initial_camera
        self.last_track_id: int = initial_track_id
        self.last_seen_time: float = time.time()
        self.cameras_seen: Set[str] = set([initial_camera]) if initial_camera else set()
        self.tracks: List[Dict[str, Any]] = (
            [{"camera_id": initial_camera, "track_id": initial_track_id}]
            if initial_camera and initial_track_id >= 0 else []
        )
        
    def add_embedding(
        self,
        embedding: np.ndarray,
        viewpoint: str,
        quality: float,
        camera_id: str,
        track_id: int,
        frame_idx: int = 0,
        par_data: Optional[Dict[str, Any]] = None,
        diversity_threshold: float = 0.92
    ) -> bool:
        """
        Conditionally adds a high-quality embedding to the view-aware bank
        following diversity and quality policies (Phase 13).
        """
        vp = viewpoint.upper() if viewpoint else "UNKNOWN"
        if vp not in self.reid_bank:
            vp = "UNKNOWN"
            
        bank = self.reid_bank[vp]
        self.view_statistics[vp] = self.view_statistics.get(vp, 0) + 1
        
        # Ensure L2 normalization
        norm = np.linalg.norm(embedding)
        if norm > 1e-8:
            norm_emb = (embedding / norm).astype(np.float32)
        else:
            norm_emb = embedding.astype(np.float32)
            
        # Diversity check against existing embeddings in this viewpoint
        for existing in bank:
            sim = cosine_similarity(norm_emb, existing["embedding"])
            if sim >= diversity_threshold:
                # Too redundant; update quality/metadata if superior
                if quality > existing.get("quality", 0.0):
                    existing["quality"] = quality
                    existing["frame_idx"] = frame_idx
                self._update_spatiotemporal(camera_id, track_id, par_data)
                return False
                
        entry = {
            "embedding": norm_emb,
            "viewpoint": vp,
            "quality": float(quality),
            "camera_id": camera_id,
            "track_id": track_id,
            "frame_idx": frame_idx
        }
        
        if len(bank) >= self.max_per_view:
            # Replace lowest quality entry if current is higher
            bank.sort(key=lambda x: x.get("quality", 0.0))
            if quality > bank[0].get("quality", 0.0):
                bank[0] = entry
            else:
                self._update_spatiotemporal(camera_id, track_id, par_data)
                return False
        else:
            bank.append(entry)
            
        self._update_spatiotemporal(camera_id, track_id, par_data)
        return True
        
    def _update_spatiotemporal(self, camera_id: str, track_id: int, par_data: Optional[Dict[str, Any]]):
        if camera_id:
            self.last_camera = camera_id
            self.cameras_seen.add(camera_id)
        if track_id >= 0:
            self.last_track_id = track_id
        self.last_seen_time = time.time()
        
        track_entry = {"camera_id": camera_id, "track_id": track_id}
        if track_entry not in self.tracks and camera_id:
            self.tracks.append(track_entry)
            
        # Aggregate PAR attributes temporally (Phase 6)
        if par_data:
            self.update_par_attributes(par_data)

    def update_par_attributes(self, par_data: Dict[str, Any]):
        """
        Aggregates PAR attributes with temporal voting to prevent single-frame noise.
        """
        upper = par_data.get("upper_color", "")
        if upper and upper != "unknown":
            votes = self.par_profile["upper_votes"]
            votes[upper] = votes.get(upper, 0) + 1
            # Majority vote becomes stable profile
            self.par_profile["upper_color"] = max(votes.items(), key=lambda x: x[1])[0]
            
        lower = par_data.get("lower_color", "")
        if lower and lower != "unknown":
            votes = self.par_profile["lower_votes"]
            votes[lower] = votes.get(lower, 0) + 1
            self.par_profile["lower_color"] = max(votes.items(), key=lambda x: x[1])[0]
            
        feat = par_data.get("features")
        if feat is not None:
            feat_arr = np.asarray(feat, dtype=np.float32).flatten()
            n = self.par_profile["par_count"]
            curr = self.par_profile["features"]
            if curr is None:
                self.par_profile["features"] = feat_arr
            else:
                self.par_profile["features"] = (curr * n + feat_arr) / (n + 1)
            self.par_profile["par_count"] += 1

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Returns all embeddings across all viewpoints (Phase 3)."""
        embs = []
        for vp in ["FRONT", "SIDE", "BACK", "UNKNOWN"]:
            embs.extend(self.reid_bank.get(vp, []))
        return embs
        
    def get_view_embeddings(self, viewpoint: str) -> List[Dict[str, Any]]:
        return self.reid_bank.get(viewpoint.upper(), [])
        
    def has_viewpoint(self, viewpoint: str) -> bool:
        return len(self.reid_bank.get(viewpoint.upper(), [])) > 0
        
    def dominant_viewpoint(self) -> str:
        best_vp = "UNKNOWN"
        best_count = 0
        for vp, bank in self.reid_bank.items():
            if len(bank) > best_count:
                best_count = len(bank)
                best_vp = vp
        return best_vp

    # Dict-like compatibility interface for seamless integration with legacy gallery/qdrant
    def __getitem__(self, item: str):
        if item == "global_id":
            return self.global_id
        elif item == "embeddings":
            return self.get_all_embeddings()
        elif item == "tracks":
            return self.tracks
        elif item == "cameras_seen":
            return self.cameras_seen
        elif item == "status":
            return self.status
        elif item == "confidence":
            return self.confidence
        elif item == "par_profile":
            return self.par_profile
        elif item == "view_statistics":
            return self.view_statistics
        elif item == "last_camera":
            return self.last_camera
        elif item == "last_seen_time":
            return self.last_seen_time
        elif item in self.__dict__:
            return getattr(self, item)
        raise KeyError(f"GlobalIdentity has no key '{item}'")
        
    def get(self, item: str, default: Any = None):
        try:
            return self[item]
        except KeyError:
            return default
            
    def __contains__(self, item: str):
        return item in [
            "global_id", "embeddings", "tracks", "cameras_seen", "status",
            "confidence", "par_profile", "view_statistics", "last_camera", "last_seen_time"
        ]
        
    def to_dict(self) -> Dict[str, Any]:
        return {
            "global_id": self.global_id,
            "embeddings": self.get_all_embeddings(),
            "tracks": self.tracks,
            "cameras_seen": list(self.cameras_seen),
            "status": self.status,
            "confidence": self.confidence,
            "par_profile": {
                "upper_color": self.par_profile["upper_color"],
                "lower_color": self.par_profile["lower_color"],
                "par_count": self.par_profile["par_count"]
            },
            "view_statistics": dict(self.view_statistics),
            "last_camera": self.last_camera,
            "last_seen_time": self.last_seen_time
        }
