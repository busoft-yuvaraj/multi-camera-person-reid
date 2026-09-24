import numpy as np
from app.utils.similarity import cosine_similarity

class GlobalMatcher:
    def __init__(self, gallery, config):
        self.gallery = gallery
        self.config = config
        
    def match(self, track_embeddings, exclude_gids=None):
        if not track_embeddings:
            return None, 0.0, 0.0, "NO_MATCH"
            
        identities = self.gallery.get_identities()
        if not identities:
            return None, 0.0, 0.0, "NO_MATCH"
            
        exclude_gids = exclude_gids or set()
        
        # Group track embeddings by viewpoint
        t_views = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
        for emb in track_embeddings:
            vp = emb.get("viewpoint", "UNKNOWN")
            t_views[vp].append(emb)
            
        # Get dominant track viewpoint (one with most embeddings)
        dom_t_view = max(t_views.items(), key=lambda x: len(x[1]))[0]
        if len(t_views[dom_t_view]) == 0:
            dom_t_view = "UNKNOWN"
            
        t_embs = t_views[dom_t_view] if dom_t_view != "UNKNOWN" else track_embeddings
        if not t_embs:
            return None, 0.0, 0.0, "NO_MATCH"
            
        # Track ReID Prototype for dominant view
        t_reid_vectors = [e["embedding"] if isinstance(e, dict) else e for e in t_embs]
        t_centroid = np.mean(t_reid_vectors, axis=0)
        t_prototype = t_centroid / (np.linalg.norm(t_centroid) + 1e-8)
        
        # Track PAR Prototype for dominant view
        t_par_vectors = [e["par_features"] for e in t_embs if isinstance(e, dict) and len(e.get("par_features", [])) > 0]
        t_par_prototype = None
        if t_par_vectors:
            t_par_centroid = np.mean(t_par_vectors, axis=0)
            t_par_prototype = t_par_centroid / (np.linalg.norm(t_par_centroid) + 1e-8)
            
        best_match = None
        best_sim = 0.0
        second_best_sim = 0.0
        best_components = {}
        
        weight_reid = self.config.get("weight_reid", 0.60)
        weight_par = self.config.get("weight_par", 0.25)
        weight_pose = self.config.get("weight_pose", 0.15)
        
        for gid, data in identities.items():
            if gid in exclude_gids:
                continue
                
            gallery_embeddings = data["embeddings"]
            if not gallery_embeddings:
                continue
                
            # Filter gallery to compatible viewpoint
            g_views = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
            for emb in gallery_embeddings:
                vp = emb.get("viewpoint", "UNKNOWN") if isinstance(emb, dict) else "UNKNOWN"
                if vp in g_views:
                    g_views[vp].append(emb)
                    
            g_embs = g_views.get(dom_t_view, [])
            if not g_embs:
                g_embs = gallery_embeddings # fallback to all if specific viewpoint missing
                
            # 1. ReID Score
            g_reid_vectors = [e["embedding"] if isinstance(e, dict) else e for e in g_embs]
            g_centroid = np.mean(g_reid_vectors, axis=0)
            g_prototype = g_centroid / (np.linalg.norm(g_centroid) + 1e-8)
            reid_score = cosine_similarity(t_prototype, g_prototype)
            
            # 2. PAR Score
            par_score = 0.0
            has_par = False
            if t_par_prototype is not None:
                g_par_vectors = [e["par_features"] for e in g_embs if isinstance(e, dict) and len(e.get("par_features", [])) > 0]
                if g_par_vectors:
                    g_par_centroid = np.mean(g_par_vectors, axis=0)
                    g_par_prototype = g_par_centroid / (np.linalg.norm(g_par_centroid) + 1e-8)
                    par_score = cosine_similarity(t_par_prototype, g_par_prototype)
                    has_par = True
                    
            # 3. Pose/Viewpoint Score (1.0 if views match exactly, else lower)
            pose_score = 0.0
            g_vp_set = set([e.get("viewpoint", "UNKNOWN") for e in gallery_embeddings if isinstance(e, dict)])
            if dom_t_view != "UNKNOWN" and dom_t_view in g_vp_set:
                pose_score = 1.0
            elif dom_t_view != "UNKNOWN":
                pose_score = 0.5 # Has some pose, but mismatched
                
            # Combine
            active_weight_par = weight_par if has_par else 0.0
            total_weight = weight_reid + active_weight_par + weight_pose
            
            total_score = (reid_score * weight_reid + par_score * active_weight_par + pose_score * weight_pose) / total_weight if total_weight > 0 else 0.0
            
            if total_score > best_sim:
                second_best_sim = best_sim
                best_sim = total_score
                best_match = gid
                best_components = {"reid": reid_score, "par": par_score, "pose": pose_score}
            elif total_score > second_best_sim:
                second_best_sim = total_score
                
        # Use new configurable thresholds
        thresh_match = self.config.get("threshold_match", 0.85)
        thresh_ambig = self.config.get("threshold_ambiguous", 0.70)
        
        status = "NO_MATCH"
        if best_match:
            if best_sim >= thresh_match:
                status = "MATCH"
            elif best_sim >= thresh_ambig:
                status = "AMBIGUOUS"
                
        # Optional: Print debug logic for scores (can be caught by logger)
        if best_match:
            import logging
            logging.info(f"MATCH_DEBUG [{best_match}]: Total={best_sim:.2f} (ReID={best_components.get('reid',0):.2f}, PAR={best_components.get('par',0):.2f}, Pose={best_components.get('pose',0):.2f}) -> {status}")
                
        return best_match, best_sim, second_best_sim, status
