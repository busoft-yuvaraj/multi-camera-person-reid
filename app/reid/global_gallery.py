import threading
import pickle
import json
import os

class GlobalGallery:
    def __init__(self, storage_path="gallery/global_gallery.pkl", clear_on_start=True):
        self.lock = threading.Lock()
        self.storage_path = storage_path
        self.metadata_path = self.storage_path.replace(".pkl", "_metadata.json")
        self.identities = {}
        self.next_id = 1
        
        if clear_on_start:
            self.save()
        else:
            self.load()
            
    def load(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path, "rb") as f:
                self.identities = pickle.load(f)
                
    def save(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.identities, f)
            
        metadata = {}
        for gid, data in self.identities.items():
            metadata[gid] = {
                "tracks": data["tracks"],
                "cameras_seen": list(data["cameras_seen"]),
                "status": data["status"],
                "num_embeddings": len(data["embeddings"])
            }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
            
    def add_identity(self, tracks, embeddings, status="TENTATIVE"):
        with self.lock:
            gid = f"Person_{self.next_id:03d}"
            self.next_id += 1
            
            cameras_seen = set([t["camera_id"] for t in tracks])
            
            self.identities[gid] = {
                "global_id": gid,
                "embeddings": embeddings,
                "tracks": tracks,
                "cameras_seen": cameras_seen,
                "status": status
            }
            self.save()
            return gid
            
    def update_identity(self, gid, new_tracks, new_embeddings, status=None):
        with self.lock:
            if gid not in self.identities:
                return False
                
            ident = self.identities[gid]
            
            for t in new_tracks:
                if t not in ident["tracks"]:
                    ident["tracks"].append(t)
                    ident["cameras_seen"].add(t["camera_id"])
                    
            ident["embeddings"].extend(new_embeddings)
            
            if status:
                ident["status"] = status
                
            self.save()
            return True
            
    def get_identities(self):
        with self.lock:
            return dict(self.identities)
