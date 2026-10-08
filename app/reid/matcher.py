import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from scipy.optimize import linear_sum_assignment
from app.utils.similarity import cosine_similarity
from app.reid.extractors.pose_extractor import PoseExtractor
from app.reid.extractors.par_extractor import ParExtractor
from app.reid.identity import GlobalIdentity

class GlobalMatcher:
    """
    Robust Multi-Camera Global ID Matcher (Phases 3, 5, 7, 9, 10, 11, 12, 14, 15).
    Core Principles:
      - OSNet is the PRIMARY identity feature (Phase 3).
      - Embedding bank is VIEW-AWARE for storage, CROSS-VIEW for matching (Phase 3).
      - Pose provides viewpoint context and observation difficulty, NOT identity (Phase 4 & 5).
      - PAR provides supporting appearance evidence with temporal profile stability (Phase 6).
      - Handles CCTV1 BACK -> CCTV2 SIDE -> CCTV2 FRONT seamlessly (Phase 7).
      - PENDING identity state prevents premature ID fragmentation (Phase 12).
      - Identity contamination prevention (Phase 14).
    """
    def __init__(self, gallery, config: Dict[str, Any]):
        self.gallery = gallery
        self.config = config
        self.last_match_details: Dict[int, Dict[str, Any]] = {}
        
        # Camera transition topology (Phase 10)
        self.camera_transitions = self.config.get("camera_transitions", {
            "waiting_lobby": ["pantry", "passage_1"],
            "pantry": ["waiting_lobby", "passage_1"],
            "passage_1": ["waiting_lobby", "pantry"],
            "cctv1": ["cctv2", "cctv3"],
            "cctv2": ["cctv1", "cctv3"]
        })

    def _extract_track_prototypes(self, track_embeddings: List[Any]) -> Tuple[Optional[np.ndarray], Dict[str, Any], str, float]:
        """
        Extracts robust ReID prototype, aggregated PAR attributes, and dominant viewpoint.
        Does NOT discard embeddings from non-dominant viewpoints (Phase 3 fix).
        """
        if not track_embeddings:
            return None, {}, "UNKNOWN", 0.0

        all_vectors = []
        view_counts: Dict[str, int] = {"FRONT": 0, "SIDE": 0, "BACK": 0, "UNKNOWN": 0}
        upper_colors: Dict[str, int] = {}
        lower_colors: Dict[str, int] = {}
        par_features_list = []

        for item in track_embeddings:
            if isinstance(item, dict):
                emb = item.get("embedding")
                vp = item.get("viewpoint", "UNKNOWN")
                par_attr = item.get("par_attributes") or {}
                par_feat = item.get("par_features")
            elif hasattr(item, "embedding"):
                emb = getattr(item, "embedding")
                vp = getattr(item, "viewpoint", "UNKNOWN")
                par_attr = getattr(item, "par_attributes", {}) or {}
                par_feat = getattr(item, "par_features", None)
            else:
                emb = item
                vp = "UNKNOWN"
                par_attr = {}
                par_feat = None

            if emb is not None:
                norm = np.linalg.norm(emb)
                if norm > 1e-8:
                    all_vectors.append(emb / norm)
                else:
                    all_vectors.append(emb)

            vp_norm = vp.upper() if vp else "UNKNOWN"
            view_counts[vp_norm] = view_counts.get(vp_norm, 0) + 1

            # Aggregate PAR data
            u_col = par_attr.get("upper_color", "")
            if u_col and u_col != "unknown":
                upper_colors[u_col] = upper_colors.get(u_col, 0) + 1
            l_col = par_attr.get("lower_color", "")
            if l_col and l_col != "unknown":
                lower_colors[l_col] = lower_colors.get(l_col, 0) + 1
            if par_feat is not None and len(par_feat) > 0:
                par_features_list.append(par_feat)

        if not all_vectors:
            return None, {}, "UNKNOWN", 0.0

        # Global track centroid across ALL collected observations
        centroid = np.mean(all_vectors, axis=0)
        norm = np.linalg.norm(centroid)
        t_prototype = (centroid / (norm + 1e-8)).astype(np.float32)

        # Dominant viewpoint & confidence
        dom_t_view = max(view_counts.items(), key=lambda x: x[1])[0]
        total_obs = max(1, sum(view_counts.values()))
        view_conf = view_counts[dom_t_view] / float(total_obs)

        # Stable track attribute profile
        dom_upper = max(upper_colors.items(), key=lambda x: x[1])[0] if upper_colors else "unknown"
        dom_lower = max(lower_colors.items(), key=lambda x: x[1])[0] if lower_colors else "unknown"
        avg_par_feat = np.mean(par_features_list, axis=0) if par_features_list else None

        t_par_data = {
            "upper_color": dom_upper,
            "lower_color": dom_lower,
            "features": avg_par_feat
        }

        return t_prototype, t_par_data, dom_t_view, view_conf

    def compute_candidate_score(
        self,
        t_prototype: np.ndarray,
        t_par_data: Dict[str, Any],
        dom_t_view: str,
        candidate_identity: Any,
        current_camera: str = "",
        consecutive_candidate_count: int = 1
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Computes multi-signal matching score against a candidate Global Identity (Phase 10 & 11).
        Signals:
          1. Cross-View ReID Score (max cosine across all stored embeddings)
          2. Stable PAR Attribute Score (color agreement + appearance)
          3. Viewpoint Context (difficulty calibration)
          4. Camera Transition Validity
          5. Temporal Consistency
        """
        # Retrieve candidate embeddings
        if isinstance(candidate_identity, GlobalIdentity):
            cand_embs = candidate_identity.get_all_embeddings()
            cand_par = candidate_identity.par_profile
            cand_last_cam = candidate_identity.last_camera
            gid = candidate_identity.global_id
        elif isinstance(candidate_identity, dict):
            cand_embs = candidate_identity.get("embeddings", [])
            cand_par = candidate_identity.get("par_profile", {})
            cand_last_cam = candidate_identity.get("last_camera", "")
            gid = candidate_identity.get("global_id", "UNKNOWN")
        else:
            return 0.0, {}

        if not cand_embs or t_prototype is None:
            return 0.0, {"reid": 0.0, "par": 0.0, "context": 0.0}

        # -----------------------------------------------------------------
        # 1. PRIMARY SIGNAL: OSNet Cross-View ReID Score (Phase 3 & 10)
        # -----------------------------------------------------------------
        # Compare current observation against ALL stored representations.
        best_reid = -1.0
        best_stored_view = "UNKNOWN"
        for e in cand_embs:
            vec = e.get("embedding") if isinstance(e, dict) else e
            vp = e.get("viewpoint", "UNKNOWN") if isinstance(e, dict) else "UNKNOWN"
            if vec is not None:
                sim = float(cosine_similarity(t_prototype, vec))
                if sim > best_reid:
                    best_reid = sim
                    best_stored_view = vp

        reid_score = max(0.0, float(best_reid))

        # -----------------------------------------------------------------
        # 2. SUPPORTING SIGNAL: PAR Attribute Score (Phase 6 & 10)
        # -----------------------------------------------------------------
        par_score, par_breakdown = ParExtractor.compare_attributes(t_par_data, cand_par)

        # -----------------------------------------------------------------
        # 3. CONTEXT SIGNAL: Viewpoint Difficulty & Transition (Phase 5 & 10)
        # -----------------------------------------------------------------
        is_cross = PoseExtractor.is_cross_view(dom_t_view, best_stored_view)
        difficulty = PoseExtractor.view_difficulty(dom_t_view, best_stored_view)

        # Camera transition plausibility
        transition_score = 0.85
        if current_camera and cand_last_cam:
            if current_camera == cand_last_cam:
                transition_score = 1.0 # Same camera continuation
            else:
                allowed = self.camera_transitions.get(cand_last_cam, [])
                if current_camera in allowed:
                    transition_score = 1.0 # Valid cross-camera transition
                else:
                    transition_score = 0.50 # Unconfigured or unexpected transition

        # Temporal consistency bonus
        temporal_score = min(1.0, 0.50 + 0.10 * min(5, consecutive_candidate_count))

        # Combined context score
        context_score = 0.45 * transition_score + 0.35 * temporal_score + 0.20 * difficulty

        # -----------------------------------------------------------------
        # 4. COMPOSITE SCORING (Phase 11)
        # -----------------------------------------------------------------
        weight_reid = float(self.config.get("weight_reid", 0.60))
        weight_par = float(self.config.get("weight_par", 0.25))
        weight_context = float(self.config.get("weight_context", 0.15))

        total_weight = weight_reid + weight_par + weight_context
        w_reid = weight_reid / total_weight
        w_par = weight_par / total_weight
        w_ctx = weight_context / total_weight

        # Cross-view difficulty adjustment:
        # If cross-view (e.g. CCTV1 BACK -> CCTV2 SIDE), OSNet similarity is inherently lower.
        # Calibrate effective ReID score according to difficulty factor for candidate evaluation
        effective_reid = reid_score
        if is_cross and difficulty < 1.0:
            effective_reid = min(1.0, reid_score / max(difficulty, 0.65))

        raw_composite = (w_reid * effective_reid) + (w_par * par_score) + (w_ctx * context_score)

        # -----------------------------------------------------------------
        # 5. ANTI-VETO SAFEGUARDS (Phase 11)
        # -----------------------------------------------------------------
        # Strong contradictory ReID cannot be overpowered by matching dark pants
        if reid_score < 0.40:
            final_score = min(raw_composite, 0.45)
        # Overwhelming ReID similarity cannot be blocked by minor attribute noise
        elif reid_score >= 0.85:
            final_score = max(raw_composite, reid_score * 0.95)
        else:
            final_score = raw_composite

        details = {
            "gid": gid,
            "reid_raw": reid_score,
            "reid_effective": effective_reid,
            "par_score": par_score,
            "context_score": context_score,
            "transition_score": transition_score,
            "temporal_score": temporal_score,
            "difficulty": difficulty,
            "is_cross_view": is_cross,
            "dom_view": dom_t_view,
            "stored_view": best_stored_view,
            "par_breakdown": par_breakdown,
            "final_score": float(final_score)
        }
        return float(final_score), details

    def match(
        self,
        track_embeddings: List[Any],
        exclude_gids: Optional[Set[str]] = None,
        current_camera: str = "",
        track_id: int = -1,
        consecutive_candidate_count: int = 1
    ) -> Tuple[Optional[str], float, float, str]:
        """
        Matches an active track's observations against the Global Gallery (Phase 9 & 12).
        Returns:
            (best_match_gid, best_sim, second_best_sim, status)
        Where status is one of:
            'MATCH'     -> Confirmed Global ID match
            'PENDING'   -> Plausible candidate awaiting further confirmation (Phase 12)
            'NO_MATCH'  -> No plausible gallery identity found
        """
        if not track_embeddings:
            return None, 0.0, 0.0, "NO_MATCH"

        identities = self.gallery.get_identities()
        if not identities:
            return None, 0.0, 0.0, "NO_MATCH"

        exclude_gids = exclude_gids or set()
        t_prototype, t_par_data, dom_t_view, view_conf = self._extract_track_prototypes(track_embeddings)
        if t_prototype is None:
            return None, 0.0, 0.0, "NO_MATCH"

        # -------------------------------------------------------------
        # Phase 9: Candidate Generation
        # Retrieve candidate identities and score with multi-signal matching
        # -------------------------------------------------------------
        candidate_scores = []
        for gid, ident in identities.items():
            if gid in exclude_gids:
                continue

            score, details = self.compute_candidate_score(
                t_prototype=t_prototype,
                t_par_data=t_par_data,
                dom_t_view=dom_t_view,
                candidate_identity=ident,
                current_camera=current_camera,
                consecutive_candidate_count=consecutive_candidate_count
            )
            candidate_scores.append((gid, score, details))

        if not candidate_scores:
            return None, 0.0, 0.0, "NO_MATCH"

        # Sort candidates by final score descending
        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        best_gid, best_sim, best_details = candidate_scores[0]
        second_best_sim = candidate_scores[1][1] if len(candidate_scores) > 1 else 0.0
        margin = best_sim - second_best_sim

        if track_id >= 0:
            self.last_match_details[track_id] = best_details

        # -------------------------------------------------------------
        # Phase 12: Decision Logic with PENDING State
        # -------------------------------------------------------------
        thresh_match = float(self.config.get("threshold_match", 0.80))
        thresh_ambig = float(self.config.get("threshold_ambiguous", 0.68))
        thresh_cross = float(self.config.get("threshold_cross_view", 0.62))
        ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))

        is_cross = best_details.get("is_cross_view", False)
        status = "NO_MATCH"

        # Confirmed match
        if best_sim >= thresh_match and (margin >= ambig_margin or second_best_sim == 0.0):
            status = "MATCH"
        # PENDING state: plausible cross-view (CCTV1 BACK -> CCTV2 SIDE) or borderline match
        elif (is_cross and best_sim >= thresh_cross) or (best_sim >= thresh_ambig):
            status = "PENDING"
        else:
            status = "NO_MATCH"

        # Phase 15: Detailed Logging
        logging.info(
            f"[GLOBAL MATCH EVAL] Track={track_id}, Cam={current_camera}, View={dom_t_view} | "
            f"Top={best_gid} (Final={best_sim:.2f}, ReID_raw={best_details.get('reid_raw', 0):.2f}, "
            f"PAR={best_details.get('par_score', 0):.2f}, Cross={is_cross}), 2nd={second_best_sim:.2f}, "
            f"Margin={margin:.2f} -> {status}"
        )

        return best_gid, float(best_sim), float(second_best_sim), status

    def match_batch(
        self,
        tracks_dict: Dict[int, List[Any]],
        exclude_gids: Optional[Set[str]] = None,
        current_camera: str = ""
    ) -> Dict[int, Tuple[Optional[str], float, float, str]]:
        """
        Batch matching using Hungarian algorithm for global 1-to-1 camera assignment.
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
            t_embs = tracks_dict[tid]
            t_proto, t_par, dom_vp, _ = self._extract_track_prototypes(t_embs)
            if t_proto is None:
                continue

            for j, gid in enumerate(gallery_gids):
                score, _ = self.compute_candidate_score(
                    t_prototype=t_proto,
                    t_par_data=t_par,
                    dom_t_view=dom_vp,
                    candidate_identity=identities[gid],
                    current_camera=current_camera
                )
                sim_matrix[i, j] = score

        cost_matrix = 1.0 - np.clip(sim_matrix, 0.0, 1.0)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        assignment_map = dict(zip(row_ind, col_ind))

        thresh_match = float(self.config.get("threshold_match", 0.80))
        thresh_ambig = float(self.config.get("threshold_ambiguous", 0.68))
        thresh_cross = float(self.config.get("threshold_cross_view", 0.62))
        ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))

        results = {}
        for i, tid in enumerate(track_ids):
            if i in assignment_map:
                assigned_j = assignment_map[i]
                assigned_gid = gallery_gids[assigned_j]
                sim = float(sim_matrix[i, assigned_j])

                other_sims = [sim_matrix[i, j] for j in range(num_gallery) if j != assigned_j]
                second_sim = float(max(other_sims)) if other_sims else 0.0
                margin = sim - second_sim

                status = "NO_MATCH"
                if sim >= thresh_match and (margin >= ambig_margin or second_sim == 0.0):
                    status = "MATCH"
                elif sim >= thresh_cross or sim >= thresh_ambig:
                    status = "PENDING"

                results[tid] = (assigned_gid, sim, second_sim, status)
            else:
                results[tid] = (None, 0.0, 0.0, "NO_MATCH")

        return results
