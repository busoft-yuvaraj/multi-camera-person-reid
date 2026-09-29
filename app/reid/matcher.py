import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from scipy.optimize import linear_sum_assignment
from app.utils.similarity import cosine_similarity

class GlobalMatcher:
    def __init__(self, gallery, config: Dict[str, Any]):
        self.gallery = gallery
        self.config = config

    def _extract_track_prototypes(self, track_embeddings: List[Any]):
        """
        Extracts dominant viewpoint, ReID prototype, and PAR prototype from track embeddings.
        """
        if not track_embeddings:
            return None, None, None, "UNKNOWN"

        t_views = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
        for emb in track_embeddings:
            vp = emb.get("viewpoint", "UNKNOWN") if isinstance(emb, dict) else "UNKNOWN"
            t_views[vp].append(emb)

        dom_t_view = max(t_views.items(), key=lambda x: len(x[1]))[0]
        if len(t_views[dom_t_view]) == 0:
            dom_t_view = "UNKNOWN"

        t_embs = t_views[dom_t_view] if dom_t_view != "UNKNOWN" else track_embeddings
        if not t_embs:
            return None, None, None, "UNKNOWN"

        t_reid_vectors = [e["embedding"] if isinstance(e, dict) else e for e in t_embs]
        t_centroid = np.mean(t_reid_vectors, axis=0)
        t_prototype = t_centroid / (np.linalg.norm(t_centroid) + 1e-8)

        t_par_vectors = [e["par_features"] for e in t_embs if isinstance(e, dict) and len(e.get("par_features", [])) > 0]
        t_par_prototype = None
        if t_par_vectors:
            t_par_centroid = np.mean(t_par_vectors, axis=0)
            t_par_prototype = t_par_centroid / (np.linalg.norm(t_par_centroid) + 1e-8)

        return t_prototype, t_par_prototype, t_embs, dom_t_view

    def compute_similarity(
        self,
        t_prototype: np.ndarray,
        t_par_prototype: Optional[np.ndarray],
        dom_t_view: str,
        gallery_embeddings: List[Any]
    ) -> Tuple[float, Dict[str, float]]:
        """
        Computes composite similarity score between track prototypes and gallery embeddings.
        """
        if not gallery_embeddings or t_prototype is None:
            return 0.0, {"reid": 0.0, "par": 0.0, "pose": 0.0}

        # Filter gallery to compatible viewpoint
        g_views = {"FRONT": [], "SIDE": [], "BACK": [], "UNKNOWN": []}
        for emb in gallery_embeddings:
            vp = emb.get("viewpoint", "UNKNOWN") if isinstance(emb, dict) else "UNKNOWN"
            if vp in g_views:
                g_views[vp].append(emb)

        g_embs = g_views.get(dom_t_view, [])
        if not g_embs:
            g_embs = gallery_embeddings  # fallback to all embeddings if viewpoint missing

        # 1. ReID Appearance Score
        g_reid_vectors = [e["embedding"] if isinstance(e, dict) else e for e in g_embs]
        g_centroid = np.mean(g_reid_vectors, axis=0)
        g_prototype = g_centroid / (np.linalg.norm(g_centroid) + 1e-8)
        reid_score = float(cosine_similarity(t_prototype, g_prototype))

        # 2. PAR Attribute Score
        par_score = 0.0
        has_par = False
        if t_par_prototype is not None:
            g_par_vectors = [e["par_features"] for e in g_embs if isinstance(e, dict) and len(e.get("par_features", [])) > 0]
            if g_par_vectors:
                g_par_centroid = np.mean(g_par_vectors, axis=0)
                g_par_prototype = g_par_centroid / (np.linalg.norm(g_par_centroid) + 1e-8)
                par_score = float(cosine_similarity(t_par_prototype, g_par_prototype))
                has_par = True

        # 3. Pose / Viewpoint Score
        # When weight_pose is 0.0, we completely drop it.
        # Previously returned 1.0 because 'dom_t_view in g_vp_set' was always True for accumulated gallery banks.
        weight_reid = float(self.config.get("weight_reid", 0.85))
        weight_par = float(self.config.get("weight_par", 0.15))
        weight_pose = float(self.config.get("weight_pose", 0.0))

        pose_score = 0.0
        if weight_pose > 0.0:
            g_vp_set = set([e.get("viewpoint", "UNKNOWN") for e in gallery_embeddings if isinstance(e, dict)])
            if dom_t_view != "UNKNOWN" and dom_t_view in g_vp_set:
                pose_score = 1.0
            elif dom_t_view != "UNKNOWN":
                pose_score = 0.5

        active_weight_par = weight_par if has_par else 0.0
        active_weight_pose = weight_pose if weight_pose > 0.0 else 0.0

        total_weight = weight_reid + active_weight_par + active_weight_pose
        if total_weight > 0.0:
            total_score = (reid_score * weight_reid + par_score * active_weight_par + pose_score * active_weight_pose) / total_weight
        else:
            total_score = 0.0

        components = {"reid": reid_score, "par": par_score, "pose": pose_score}
        return float(total_score), components

    def match(self, track_embeddings: List[Any], exclude_gids: Optional[Set[str]] = None) -> Tuple[Optional[str], float, float, str]:
        """
        Compares a single track's embeddings against the gallery.
        Returns:
            (best_match_gid, best_sim, second_best_sim, status)
        """
        if not track_embeddings:
            return None, 0.0, 0.0, "NO_MATCH"

        identities = self.gallery.get_identities()
        if not identities:
            return None, 0.0, 0.0, "NO_MATCH"

        exclude_gids = exclude_gids or set()
        t_prototype, t_par_prototype, _, dom_t_view = self._extract_track_prototypes(track_embeddings)
        if t_prototype is None:
            return None, 0.0, 0.0, "NO_MATCH"

        best_match = None
        best_sim = 0.0
        second_best_sim = 0.0
        best_components = {}

        for gid, data in identities.items():
            if gid in exclude_gids:
                continue

            gallery_embeddings = data.get("embeddings", [])
            if not gallery_embeddings:
                continue

            total_score, components = self.compute_similarity(
                t_prototype, t_par_prototype, dom_t_view, gallery_embeddings
            )

            if total_score > best_sim:
                second_best_sim = best_sim
                best_sim = total_score
                best_match = gid
                best_components = components
            elif total_score > second_best_sim:
                second_best_sim = total_score

        thresh_match = float(self.config.get("threshold_match", 0.83))
        thresh_ambig = float(self.config.get("threshold_ambiguous", 0.80))
        ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))

        status = "NO_MATCH"
        if best_match:
            margin = best_sim - second_best_sim
            if best_sim >= thresh_match and (margin >= ambig_margin or second_best_sim == 0.0):
                status = "MATCH"
            elif best_sim >= thresh_ambig:
                status = "AMBIGUOUS"

            logging.debug(
                f"MATCH_DEBUG [{best_match}]: Total={best_sim:.2f}, 2nd={second_best_sim:.2f}, "
                f"margin={margin:.2f} (ReID={best_components.get('reid', 0):.2f}, "
                f"PAR={best_components.get('par', 0):.2f}) -> {status}"
            )

        return best_match, best_sim, second_best_sim, status

    def match_batch(
        self,
        tracks_dict: Dict[int, List[Any]],
        exclude_gids: Optional[Set[str]] = None
    ) -> Dict[int, Tuple[Optional[str], float, float, str]]:
        """
        Applies Hungarian matching across all active tracks in a camera against gallery identities.
        Ensures ONE global ID per camera at a time — no two tracks can ever share a Global ID.

        Returns:
            Dict[track_id -> (assigned_gid, similarity, second_best_sim, status)]
        """
        if not tracks_dict:
            return {}

        identities = self.gallery.get_identities()
        if not identities:
            return {tid: (None, 0.0, 0.0, "NO_MATCH") for tid in tracks_dict}

        exclude_gids = exclude_gids or set()
        gallery_gids = [gid for gid in identities.keys() if gid not in exclude_gids]

        if not gallery_gids:
            return {tid: (None, 0.0, 0.0, "NO_MATCH") for tid in tracks_dict}

        track_ids = list(tracks_dict.keys())
        num_tracks = len(track_ids)
        num_gallery = len(gallery_gids)

        sim_matrix = np.zeros((num_tracks, num_gallery), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            t_embeddings = tracks_dict[tid]
            t_prototype, t_par_prototype, _, dom_t_view = self._extract_track_prototypes(t_embeddings)
            if t_prototype is None:
                continue

            for j, gid in enumerate(gallery_gids):
                g_embs = identities[gid].get("embeddings", [])
                score, _ = self.compute_similarity(t_prototype, t_par_prototype, dom_t_view, g_embs)
                sim_matrix[i, j] = score

        # Cost matrix for Hungarian Assignment: higher similarity -> lower cost
        cost_matrix = 1.0 - np.clip(sim_matrix, 0.0, 1.0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Mapping of track_index -> assigned gallery_index
        assignment_map = dict(zip(row_ind, col_ind))

        thresh_match = float(self.config.get("threshold_match", 0.83))
        thresh_ambig = float(self.config.get("threshold_ambiguous", 0.80))
        ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))

        results = {}
        for i, tid in enumerate(track_ids):
            if i in assignment_map:
                assigned_j = assignment_map[i]
                assigned_gid = gallery_gids[assigned_j]
                sim = float(sim_matrix[i, assigned_j])

                # Second best among all other gallery candidates
                other_sims = [sim_matrix[i, j] for j in range(num_gallery) if j != assigned_j]
                second_sim = float(max(other_sims)) if other_sims else 0.0
                margin = sim - second_sim

                status = "NO_MATCH"
                if sim >= thresh_match and (margin >= ambig_margin or second_sim == 0.0):
                    status = "MATCH"
                elif sim >= thresh_ambig:
                    status = "AMBIGUOUS"

                results[tid] = (assigned_gid, sim, second_sim, status)
            else:
                results[tid] = (None, 0.0, 0.0, "NO_MATCH")

        return results
