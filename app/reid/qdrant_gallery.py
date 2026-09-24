import threading
import uuid
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

class QdrantGallery:
    def __init__(self, url, api_key, collection_name="cctv-poc", clear_on_start=True, max_embeddings=15):
        self.lock = threading.Lock()
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = collection_name
        self.max_embeddings = max_embeddings
        
        self.identities = {}
        self.next_id = 1
        
        if clear_on_start:
            if self.client.collection_exists(self.collection_name):
                self.client.delete_collection(self.collection_name)
                
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
        else:
            self._load_from_qdrant()
            
    def _load_from_qdrant(self):
        # We scroll through all points and reconstruct the in-memory dict
        # This keeps the matcher fast (O(1) dict lookup instead of network calls on every frame)
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True,
                with_vectors=True,
                offset=offset
            )
            for record in records:
                gid = record.payload["global_id"]
                if gid not in self.identities:
                    self.identities[gid] = {
                        "global_id": gid,
                        "embeddings": [],
                        "tracks": record.payload.get("tracks", []),
                        "cameras_seen": set(record.payload.get("cameras_seen", [])),
                        "status": record.payload.get("status", "TENTATIVE"),
                        "point_ids": []
                    }
                # Reconstruct dict format for compatibility
                emb_dict = {
                    "embedding": record.vector,
                    "viewpoint": record.payload.get("viewpoint", "UNKNOWN"),
                    "par_features": record.payload.get("par_features", []),
                    "quality": record.payload.get("quality", 0.0),
                    "metadata": record.payload.get("metadata", {})
                }
                self.identities[gid]["embeddings"].append(emb_dict)
                self.identities[gid]["point_ids"].append(record.id)
                
            if offset is None:
                break
                
        # Update next_id
        for gid in self.identities.keys():
            try:
                num = int(gid.split("_")[1])
                if num >= self.next_id:
                    self.next_id = num + 1
            except:
                pass

    def add_identity(self, tracks, embeddings, status="TENTATIVE"):
        # embeddings here is actually all_embs from TrackEmbeddingBank (dicts)
        with self.lock:
            gid = f"Person_{self.next_id:03d}"
            self.next_id += 1
            
            cameras_seen = set([t["camera_id"] for t in tracks])
            
            # Limit to max_embeddings PER VIEWPOINT
            kept_embeddings = []
            view_groups = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
            for emb in embeddings:
                vp = emb.get("viewpoint", "UNKNOWN")
                if vp in view_groups:
                    view_groups[vp].append(emb)
                    
            for vp, embs in view_groups.items():
                embs.sort(key=lambda x: x.get("quality", 0.0), reverse=True)
                kept_embeddings.extend(embs[:self.max_embeddings])
            
            self.identities[gid] = {
                "global_id": gid,
                "embeddings": kept_embeddings,
                "tracks": tracks,
                "cameras_seen": cameras_seen,
                "status": status,
                "point_ids": []
            }
            
            self._upload_to_qdrant(gid)
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
                
            # Keep top 5 per viewpoint
            kept_embeddings = []
            view_groups = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
            for emb in ident["embeddings"]:
                vp = emb.get("viewpoint", "UNKNOWN")
                # Handle old data missing viewpoint by defaulting to UNKNOWN
                if vp not in view_groups:
                    vp = "UNKNOWN"
                view_groups[vp].append(emb)
                
            for vp, embs in view_groups.items():
                # Sort by quality (highest first)
                embs.sort(key=lambda x: x.get("quality", 0.0) if isinstance(x, dict) else 0.0, reverse=True)
                # Keep top N
                kept_embeddings.extend(embs[:self.max_embeddings])
                
            ident["embeddings"] = kept_embeddings
                
            self._upload_to_qdrant(gid)
            return True
            
    def _upload_to_qdrant(self, gid):
        ident = self.identities[gid]
        
        # Grab old point IDs to delete
        old_point_ids = ident.get("point_ids", [])
            
        ident["point_ids"] = []
        points = []
        
        for emb in ident["embeddings"]:
            point_id = str(uuid.uuid4())
            ident["point_ids"].append(point_id)
            
            vector = emb["embedding"] if isinstance(emb, dict) else emb
            payload = {
                "global_id": gid,
                "tracks": ident["tracks"],
                "cameras_seen": list(ident["cameras_seen"]),
                "status": ident["status"]
            }
            
            if isinstance(emb, dict):
                payload.update({
                    "viewpoint": emb.get("viewpoint", "UNKNOWN"),
                    "par_features": emb.get("par_features", []).tolist() if hasattr(emb.get("par_features"), "tolist") else emb.get("par_features", []),
                    "quality": float(emb.get("quality", 0.0)),
                    "metadata": emb.get("metadata", {})
                })
                
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    payload=payload
                )
            )
            
        # Do network I/O in a background thread to prevent blocking the video processing loop
        threading.Thread(target=self._do_network_upload, args=(old_point_ids, points), daemon=True).start()

    def _do_network_upload(self, old_point_ids, points):
        try:
            if old_point_ids:
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=old_point_ids
                )
            if points:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
        except Exception as e:
            print(f"Qdrant background upload error: {e}")

    def get_identities(self):
        with self.lock:
            return dict(self.identities)
