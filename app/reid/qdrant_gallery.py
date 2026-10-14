import os
import json
import logging
import threading
import uuid
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from qdrant_client import QdrantClient
from qdrant_client.http import models
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
                        "point_ids": [],
                        "par_profile": record.payload.get("stable_par_profile", {
                            "upper_color": record.payload.get("upper_color", "unknown"),
                            "lower_color": record.payload.get("lower_color", "unknown"),
                            "par_count": 1
                        })
                    }
                # Reconstruct dict format for compatibility
                par_attr = record.payload.get("par_attributes") or {
                    "upper_color": record.payload.get("upper_color", "unknown"),
                    "lower_color": record.payload.get("lower_color", "unknown")
                }
                emb_dict = {
                    "embedding": record.vector,
                    "viewpoint": record.payload.get("viewpoint", "UNKNOWN"),
                    "par_features": record.payload.get("par_features", []),
                    "par_attributes": par_attr,
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
            
    def _update_par_profile(self, ident):
        upper_votes = {}
        lower_votes = {}
        for emb in ident.get("embeddings", []):
            if isinstance(emb, dict):
                par_attr = emb.get("par_attributes") or {}
                u = par_attr.get("upper_color")
                l = par_attr.get("lower_color")
                if u and u != "unknown":
                    upper_votes[u] = upper_votes.get(u, 0) + 1
                if l and l != "unknown":
                    lower_votes[l] = lower_votes.get(l, 0) + 1

        ident["par_profile"] = {
            "upper_color": max(upper_votes, key=upper_votes.get) if upper_votes else "unknown",
            "lower_color": max(lower_votes, key=lower_votes.get) if lower_votes else "unknown",
            "upper_votes": upper_votes,
            "lower_votes": lower_votes,
            "par_count": len(ident.get("embeddings", []))
        }

    def _export_par_profiles(self):
        try:
            os.makedirs("gallery", exist_ok=True)
            par_profiles = {}
            metadata = {}
            for gid, ident in self.identities.items():
                par_p = ident.get("par_profile", {})
                par_profiles[gid] = {
                    "global_id": gid,
                    "status": ident.get("status", "TENTATIVE"),
                    "upper_color": str(par_p.get("upper_color", "unknown")),
                    "lower_color": str(par_p.get("lower_color", "unknown")),
                    "upper_votes": par_p.get("upper_votes", {}),
                    "lower_votes": par_p.get("lower_votes", {}),
                    "total_observations": int(par_p.get("par_count", 0)),
                    "cameras_seen": list(ident.get("cameras_seen", []))
                }
                metadata[gid] = {
                    "tracks": ident.get("tracks", []),
                    "cameras_seen": list(ident.get("cameras_seen", [])),
                    "status": ident.get("status", "TENTATIVE"),
                    "num_embeddings": len(ident.get("embeddings", [])),
                    "par_profile": {
                        "upper_color": str(par_p.get("upper_color", "unknown")),
                        "lower_color": str(par_p.get("lower_color", "unknown")),
                        "par_count": int(par_p.get("par_count", 0))
                    }
                }
            with open("gallery/par_profiles.json", "w") as f:
                json.dump(par_profiles, f, indent=4)
            with open("gallery/global_gallery_metadata.json", "w") as f:
                json.dump(metadata, f, indent=4)
        except Exception as e:
            logging.debug(f"Failed to export par_profiles.json: {e}")

    def _upload_to_qdrant(self, gid):
        ident = self.identities[gid]
        self._update_par_profile(ident)
        
        # Grab old point IDs to delete
        old_point_ids = ident.get("point_ids", [])
            
        ident["point_ids"] = []
        points = []
        
        for emb in ident["embeddings"]:
            point_id = str(uuid.uuid4())
            ident["point_ids"].append(point_id)
            
            vector = emb["embedding"] if isinstance(emb, dict) else emb
            meta = emb.get("metadata", {}) if isinstance(emb, dict) else {}
            last_track = ident["tracks"][-1] if ident.get("tracks") else {}
            payload = {
                "global_id": gid,
                "camera_id": meta.get("camera_id", last_track.get("camera_id", "")),
                "zone": meta.get("zone", last_track.get("zone", last_track.get("camera_id", ""))),
                "local_track_id": meta.get("track_id", last_track.get("track_id", -1)),
                "timestamp": meta.get("timestamp", 0.0),
                "tracks": ident["tracks"],
                "cameras_seen": list(ident["cameras_seen"]),
                "status": ident["status"]
            }
            
            if isinstance(emb, dict):
                par_attr = emb.get("par_attributes") or {}
                payload.update({
                    "viewpoint": emb.get("viewpoint", "UNKNOWN"),
                    "upper_color": par_attr.get("upper_color", "unknown"),
                    "lower_color": par_attr.get("lower_color", "unknown"),
                    "par_attributes": {
                        "upper_color": par_attr.get("upper_color", "unknown"),
                        "lower_color": par_attr.get("lower_color", "unknown"),
                        "confidence": float(par_attr.get("confidence", 0.0))
                    },
                    "par_features": emb.get("par_features", []).tolist() if hasattr(emb.get("par_features"), "tolist") else emb.get("par_features", []),
                    "quality": float(emb.get("quality", 0.0)),
                    "metadata": meta
                })

            if "par_profile" in ident:
                payload["stable_par_profile"] = {
                    "upper_color": str(ident["par_profile"].get("upper_color", "unknown")),
                    "lower_color": str(ident["par_profile"].get("lower_color", "unknown")),
                    "upper_votes": ident["par_profile"].get("upper_votes", {}),
                    "lower_votes": ident["par_profile"].get("lower_votes", {}),
                    "par_count": int(ident["par_profile"].get("par_count", 0))
                }
                
            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist() if hasattr(vector, "tolist") else vector,
                    payload=payload
                )
            )
            
        self._export_par_profiles()
            
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

    def search_candidates(
        self,
        query_vector: np.ndarray,
        candidate_gids: List[str],
        top_k: int = 5
    ) -> Dict[str, float]:
        """
        Candidate-restricted vector search in Qdrant (Phase 1).
        Only searches across valid candidate Global IDs (as filtered by Topology,
        Temporal, and Spatial gating).
        Returns:
            Dict[global_id -> max_similarity_score]
        """
        if not candidate_gids or query_vector is None:
            return {}

        vec = query_vector.tolist() if hasattr(query_vector, "tolist") else list(query_vector)
        scores: Dict[str, float] = {}

        try:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="global_id",
                        match=models.MatchAny(any=candidate_gids)
                    )
                ]
            )
            if hasattr(self.client, "query_points"):
                query_res = self.client.query_points(
                    collection_name=self.collection_name,
                    query=vec,
                    query_filter=query_filter,
                    limit=top_k * len(candidate_gids),
                    with_payload=True
                )
                results = query_res.points
            else:
                results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=vec,
                    query_filter=query_filter,
                    limit=top_k * len(candidate_gids),
                    with_payload=True
                )
            for res in results:
                gid = res.payload.get("global_id")
                sim = float(res.score)
                if gid:
                    scores[gid] = max(scores.get(gid, 0.0), sim)
        except Exception as e:
            logging.debug(f"[QDRANT] Direct search error ({e}), falling back to candidate-restricted in-memory evaluation")
            with self.lock:
                for gid in candidate_gids:
                    ident = self.identities.get(gid)
                    if not ident:
                        continue
                    best_sim = 0.0
                    for emb_entry in ident.get("embeddings", []):
                        e_vec = emb_entry.get("embedding") if isinstance(emb_entry, dict) else emb_entry
                        if e_vec is not None:
                            dot = float(np.dot(query_vector, e_vec))
                            norm = float((np.linalg.norm(query_vector) * np.linalg.norm(e_vec)) + 1e-8)
                            sim = dot / norm
                            best_sim = max(best_sim, sim)
                    if best_sim > 0.0:
                        scores[gid] = best_sim

        return scores
