import threading
import pickle
import json
import os
from typing import Dict, List, Any, Optional
from app.reid.identity import GlobalIdentity

class GlobalGallery:
    def __init__(self, storage_path="gallery/global_gallery.pkl", clear_on_start=True, max_per_view=5):
        self.lock = threading.Lock()
        self.storage_path = storage_path
        self.metadata_path = self.storage_path.replace(".pkl", "_metadata.json")
        self.identities: Dict[str, GlobalIdentity] = {}
        self.next_id = 1
        self.max_per_view = max_per_view
        
        if clear_on_start:
            self.save()
        else:
            self.load()
            
    def load(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, "rb") as f:
                    loaded = pickle.load(f)
                self.identities = {}
                for gid, data in loaded.items():
                    if isinstance(data, GlobalIdentity):
                        self.identities[gid] = data
                    else:
                        # Convert legacy dict to GlobalIdentity
                        ident = GlobalIdentity(
                            global_id=gid,
                            status=data.get("status", "CONFIRMED"),
                            max_per_view=self.max_per_view
                        )
                        for emb_dict in data.get("embeddings", []):
                            vec = emb_dict["embedding"] if isinstance(emb_dict, dict) else emb_dict
                            vp = emb_dict.get("viewpoint", "UNKNOWN") if isinstance(emb_dict, dict) else "UNKNOWN"
                            q = emb_dict.get("quality", 0.8) if isinstance(emb_dict, dict) else 0.8
                            cid = emb_dict.get("camera_id", "") if isinstance(emb_dict, dict) else ""
                            tid = emb_dict.get("track_id", -1) if isinstance(emb_dict, dict) else -1
                            par = emb_dict.get("par_attributes") if isinstance(emb_dict, dict) else None
                            ident.add_embedding(vec, vp, q, cid, tid, par_data=par)
                        self.identities[gid] = ident
                        
                # Update next_id
                for gid in self.identities.keys():
                    try:
                        num = int(gid.split("_")[1])
                        if num >= self.next_id:
                            self.next_id = num + 1
                    except Exception:
                        pass
            except Exception as e:
                self.identities = {}
                
    def save(self):
        dir_name = os.path.dirname(self.storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(self.storage_path, "wb") as f:
            pickle.dump(self.identities, f)
            
        metadata = {}
        for gid, ident in self.identities.items():
            metadata[gid] = {
                "tracks": ident.tracks if hasattr(ident, "tracks") else ident.get("tracks", []),
                "cameras_seen": list(ident.cameras_seen if hasattr(ident, "cameras_seen") else ident.get("cameras_seen", [])),
                "status": ident.status if hasattr(ident, "status") else ident.get("status", "CONFIRMED"),
                "num_embeddings": len(ident.get_all_embeddings() if hasattr(ident, "get_all_embeddings") else ident.get("embeddings", [])),
                "par_profile": {
                    "upper_color": str(ident.par_profile.get("upper_color", "unknown")) if hasattr(ident, "par_profile") else "unknown",
                    "lower_color": str(ident.par_profile.get("lower_color", "unknown")) if hasattr(ident, "par_profile") else "unknown",
                    "par_count": int(ident.par_profile.get("par_count", 0)) if hasattr(ident, "par_profile") else 0
                }
            }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)

        # Save dedicated PAR profiles file (Phase 6 & Storage)
        par_profiles = {}
        for gid, ident in self.identities.items():
            if hasattr(ident, "par_profile"):
                par_p = ident.par_profile
                par_profiles[gid] = {
                    "global_id": gid,
                    "status": ident.status,
                    "upper_color": str(par_p.get("upper_color", "unknown")),
                    "lower_color": str(par_p.get("lower_color", "unknown")),
                    "upper_votes": par_p.get("upper_votes", {}),
                    "lower_votes": par_p.get("lower_votes", {}),
                    "total_observations": int(par_p.get("par_count", 0)),
                    "cameras_seen": list(ident.cameras_seen),
                    "last_camera": ident.last_camera,
                    "last_seen_time": ident.last_seen_time
                }
            elif isinstance(ident, dict) and "par_profile" in ident:
                par_profiles[gid] = ident["par_profile"]

        par_file = os.path.join(dir_name if dir_name else "gallery", "par_profiles.json")
        try:
            with open(par_file, "w") as f:
                json.dump(par_profiles, f, indent=4)
        except Exception:
            pass
            
    def add_identity(self, tracks: List[Dict[str, Any]], embeddings: List[Any], status="CONFIRMED") -> str:
        with self.lock:
            gid = f"Person_{self.next_id:03d}"
            self.next_id += 1
            
            init_cam = tracks[0]["camera_id"] if tracks else ""
            init_tid = tracks[0]["track_id"] if tracks else -1
            
            ident = GlobalIdentity(
                global_id=gid,
                initial_camera=init_cam,
                initial_track_id=init_tid,
                status=status,
                max_per_view=self.max_per_view
            )
            
            for t in tracks:
                if t not in ident.tracks:
                    ident.tracks.append(t)
                    ident.cameras_seen.add(t["camera_id"])
                    
            for emb_entry in embeddings:
                if isinstance(emb_entry, dict):
                    vec = emb_entry.get("embedding")
                    vp = emb_entry.get("viewpoint", "UNKNOWN")
                    q = emb_entry.get("quality", 0.8)
                    cid = emb_entry.get("camera_id", init_cam)
                    tid = emb_entry.get("track_id", init_tid)
                    f_id = emb_entry.get("frame_id", 0)
                    par = emb_entry.get("par_attributes")
                    if par is None and emb_entry.get("par_features") is not None:
                        par = {"features": emb_entry.get("par_features")}
                    ident.add_embedding(vec, vp, q, cid, tid, frame_idx=f_id, par_data=par)
                else:
                    ident.add_embedding(emb_entry, "UNKNOWN", 0.8, init_cam, init_tid)
                    
            self.identities[gid] = ident
            self.save()
            return gid
            
    def update_identity(self, gid: str, new_tracks: List[Dict[str, Any]], new_embeddings: List[Any], status=None) -> bool:
        with self.lock:
            if gid not in self.identities:
                return False
                
            ident = self.identities[gid]
            if not isinstance(ident, GlobalIdentity):
                # Legacy dict fallback
                for t in new_tracks:
                    if t not in ident["tracks"]:
                        ident["tracks"].append(t)
                        ident["cameras_seen"].add(t["camera_id"])
                ident["embeddings"].extend(new_embeddings)
                if status:
                    ident["status"] = status
                self.save()
                return True
                
            for t in new_tracks:
                if t not in ident.tracks:
                    ident.tracks.append(t)
                    ident.cameras_seen.add(t["camera_id"])
                    
            for emb_entry in new_embeddings:
                if isinstance(emb_entry, dict):
                    vec = emb_entry.get("embedding")
                    vp = emb_entry.get("viewpoint", "UNKNOWN")
                    q = emb_entry.get("quality", 0.8)
                    cid = emb_entry.get("camera_id", ident.last_camera)
                    tid = emb_entry.get("track_id", ident.last_track_id)
                    f_id = emb_entry.get("frame_id", 0)
                    par = emb_entry.get("par_attributes")
                    if par is None and emb_entry.get("par_features") is not None:
                        par = {"features": emb_entry.get("par_features")}
                    ident.add_embedding(vec, vp, q, cid, tid, frame_idx=f_id, par_data=par)
                else:
                    ident.add_embedding(emb_entry, "UNKNOWN", 0.8, ident.last_camera, ident.last_track_id)
                    
            if status:
                ident.status = status
                
            self.save()
            return True
            
    def get_identities(self) -> Dict[str, Any]:
        with self.lock:
            return dict(self.identities)
