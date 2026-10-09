import logging
from typing import Dict, List, Tuple, Optional, Set, Any
import numpy as np

from .candidate_filter import CandidateFilter
from app.journey.journey_manager import JourneyManager
from app.journey.journey_store import JourneyStore
from app.reid.matcher import GlobalMatcher


class AssociationManager:
    """
    Floor-Plan-Aware Multi-Camera Global ID Association Manager (Phase 1).
    Orchestrates the entire decision pipeline:
      1. Spatial, Temporal, & Topology Candidate Gating
      2. Candidate-Restricted Qdrant Vector Search
      3. Multi-Signal ReID, PAR, & Viewpoint Scoring
      4. Multi-Frame Confirmation & PENDING State
      5. Conflict Prevention & Reacquisition
      6. Hysteresis Stability (Anti-Switching)
      7. Journey & Diagnostic Explainability Logging
    """
    def __init__(
        self,
        gallery: Any,
        matcher: GlobalMatcher,
        candidate_filter: CandidateFilter,
        journey_manager: JourneyManager,
        store: JourneyStore,
        config: Dict[str, Any]
    ):
        self.gallery = gallery
        self.matcher = matcher
        self.candidate_filter = candidate_filter
        self.journey_manager = journey_manager
        self.store = store
        self.config = config

        # Tracking state per (camera_id, local_track_id)
        # -> current pending candidate GID
        self.pending_candidate: Dict[Tuple[str, int], str] = {}
        # -> consecutive frame count for pending candidate
        self.pending_counters: Dict[Tuple[str, int], int] = {}
        # -> consecutive frame count of NO_MATCH / no candidates
        self.no_match_counters: Dict[Tuple[str, int], int] = {}
        # -> hysteresis tracking: candidate wishing to switch
        self.hysteresis_candidate: Dict[Tuple[str, int], str] = {}
        self.hysteresis_counters: Dict[Tuple[str, int], int] = {}

    def associate_track(
        self,
        camera_id: str,
        zone: str,
        local_track_id: int,
        bank_embeddings: List[Any],
        current_time: float,
        frame_idx: int,
        active_camera_gids: Set[str]
    ) -> Tuple[Optional[str], str, float, Dict[str, Any]]:
        """
        Processes a local track and returns:
            (assigned_gid, status, similarity, decision_details)
        status:
            'MATCHED' -> Confirmed Global ID
            'PENDING' -> Awaiting further evidence (candidate identified or collecting)
            'NEW'     -> New Global ID spawned after sustained absence of candidate match
        """
        key = (camera_id, local_track_id)
        current_gid = self.journey_manager.active_tracks.get(key)
        if current_gid is None:
            # Fallback check across identities
            for gid, ident in self.journey_manager.identities.items():
                if ident.current_camera == camera_id and ident.current_local_track_id == local_track_id:
                    current_gid = gid
                    self.journey_manager._set_active_mapping(camera_id, local_track_id, gid)
                    break

        # -------------------------------------------------------------
        # Case A: Track is ALREADY CONFIRMED to a Global ID -> Check Hysteresis / Lock
        # -------------------------------------------------------------
        if current_gid is not None:
            self.journey_manager.update_track_presence(camera_id, local_track_id, current_time, frame_idx)
            
            # If lock_confirmed_id is enabled (default True), NEVER switch the ID once assigned!
            lock_confirmed = bool(self.config.get("lock_confirmed_id", True))
            if lock_confirmed:
                return current_gid, "MATCHED", 1.0, {"status": "RETAINED"}

            # Check if an overwhelming candidate wants to switch (Hysteresis Guard)
            h_thresh = float(self.config.get("hysteresis_switch_threshold", 0.92))
            h_margin = float(self.config.get("hysteresis_switch_margin", 0.12))
            h_frames = int(self.config.get("hysteresis_consecutive_frames", 10))

            # Extract prototype to evaluate
            t_proto, t_par, dom_view, _ = self.matcher._extract_track_prototypes(bank_embeddings)
            if t_proto is not None:
                # Filter candidates for switch (must also be valid!)
                identities = self.gallery.get_identities()
                exclude_gids = set(active_camera_gids)
                if current_gid in exclude_gids:
                    exclude_gids.remove(current_gid)

                valid_gids, evals, rejections = self.candidate_filter.filter_candidates(
                    identities=identities,
                    current_camera=camera_id,
                    current_zone=zone,
                    current_time=current_time,
                    exclude_gids=exclude_gids
                )

                if valid_gids:
                    q_scores = self.gallery.search_candidates(t_proto, valid_gids, top_k=5)
                    # Score candidates
                    cand_scores = []
                    for c_gid in valid_gids:
                        if c_gid == current_gid:
                            continue
                        score, details = self.matcher.compute_candidate_score(
                            t_prototype=t_proto,
                            t_par_data=t_par,
                            dom_t_view=dom_view,
                            candidate_identity=identities[c_gid],
                            current_camera=camera_id
                        )
                        cand_scores.append((c_gid, score, details))

                    if cand_scores:
                        cand_scores.sort(key=lambda x: x[1], reverse=True)
                        best_switch_gid, best_switch_score, _ = cand_scores[0]
                        second_switch_score = cand_scores[1][1] if len(cand_scores) > 1 else 0.0

                        if (best_switch_score >= h_thresh and 
                            (best_switch_score - second_switch_score) >= h_margin and 
                            best_switch_gid not in active_camera_gids):
                            
                            if self.hysteresis_candidate.get(key) == best_switch_gid:
                                self.hysteresis_counters[key] = self.hysteresis_counters.get(key, 0) + 1
                            else:
                                self.hysteresis_candidate[key] = best_switch_gid
                                self.hysteresis_counters[key] = 1

                            if self.hysteresis_counters[key] >= h_frames:
                                # Confirmed switch!
                                logging.info(f"[HYSTERESIS SWITCH] Track {local_track_id} in {camera_id}: {current_gid} -> {best_switch_gid}")
                                self.journey_manager.confirm_association(
                                    global_id=best_switch_gid,
                                    camera_id=camera_id,
                                    zone=zone,
                                    local_track_id=local_track_id,
                                    timestamp=current_time,
                                    frame_idx=frame_idx,
                                    decision_reason="HYSTERESIS_CONFIRMED_SWITCH"
                                )
                                self.hysteresis_candidate[key] = None
                                self.hysteresis_counters[key] = 0
                                return best_switch_gid, "MATCHED", best_switch_score, {"status": "HYSTERESIS_SWITCH"}
                        else:
                            self.hysteresis_candidate[key] = None
                            self.hysteresis_counters[key] = 0

            return current_gid, "MATCHED", 1.0, {"status": "RETAINED"}

        # -------------------------------------------------------------
        # Case B: Track is NOT YET CONFIRMED
        # -------------------------------------------------------------
        confirm_frames = int(self.config.get("reid_confirm_frames", 3))
        if len(bank_embeddings) < confirm_frames:
            return None, "PENDING", 0.0, {
                "decision": "COLLECTING_EVIDENCE",
                "observations": len(bank_embeddings),
                "required": confirm_frames
            }

        t_proto, t_par, dom_view, view_conf = self.matcher._extract_track_prototypes(bank_embeddings)
        if t_proto is None:
            return None, "PENDING", 0.0, {"decision": "NO_PROTOTYPE"}

        # -------------------------------------------------------------
        # Step 1: Candidate Filtering (Topology + Temporal + Spatial)
        # -------------------------------------------------------------
        identities = self.gallery.get_identities()
        camera_occupied_gids = set(active_camera_gids)

        valid_gids, evals, rejections = self.candidate_filter.filter_candidates(
            identities=identities,
            current_camera=camera_id,
            current_zone=zone,
            current_time=current_time,
            exclude_gids=camera_occupied_gids
        )

        decision_data: Dict[str, Any] = {
            "timestamp": round(current_time, 3),
            "frame_idx": frame_idx,
            "camera_id": camera_id,
            "zone": zone,
            "local_track_id": local_track_id,
            "evidence_count": len(bank_embeddings),
            "dominant_view": dom_view,
            "all_gallery_gids": list(identities.keys()),
            "valid_candidates": valid_gids,
            "rejected_candidates": rejections
        }

        # -------------------------------------------------------------
        # Step 2: If NO valid candidates pass spatial/temporal/topology gates
        # -------------------------------------------------------------
        if not valid_gids:
            self.no_match_counters[key] = self.no_match_counters.get(key, 0) + 1
            new_id_confirm = int(self.config.get("reid_new_id_confirm_frames", 15))

            decision_data["decision"] = "NO_VALID_CANDIDATES"
            decision_data["no_match_count"] = self.no_match_counters[key]
            decision_data["new_id_threshold"] = new_id_confirm

            # Only spawn new ID after sustained absence of any possible candidate
            if self.no_match_counters[key] >= new_id_confirm:
                new_gid = self.gallery.add_identity([{"camera_id": camera_id, "track_id": local_track_id}], bank_embeddings)
                self.journey_manager.register_new_identity(
                    global_id=new_gid,
                    camera_id=camera_id,
                    zone=zone,
                    local_track_id=local_track_id,
                    timestamp=current_time,
                    frame_idx=frame_idx
                )
                self.no_match_counters[key] = 0
                self.pending_candidate[key] = None
                self.pending_counters[key] = 0

                decision_data["selected_global_id"] = new_gid
                decision_data["decision"] = "NEW_GLOBAL_ID_CREATED"
                self.store.log_association_decision(decision_data)
                return new_gid, "MATCHED", 1.0, decision_data
            else:
                decision_data["decision"] = "PENDING_NEW (Awaiting more evidence)"
                self.store.log_association_decision(decision_data)
                return None, "PENDING", 0.0, decision_data

        # -------------------------------------------------------------
        # Step 3: Candidate-Restricted Qdrant Vector Search
        # -------------------------------------------------------------
        qdrant_scores = self.gallery.search_candidates(t_proto, valid_gids, top_k=5)
        decision_data["qdrant_scores"] = {k: round(v, 3) for k, v in qdrant_scores.items()}

        # -------------------------------------------------------------
        # Step 4: Multi-Signal Composite Scoring on Valid Candidates
        # -------------------------------------------------------------
        scored_candidates = []
        for cand_gid in valid_gids:
            ident = identities[cand_gid]
            p_count = self.pending_counters.get(key, 1)
            score, details = self.matcher.compute_candidate_score(
                t_prototype=t_proto,
                t_par_data=t_par,
                dom_t_view=dom_view,
                candidate_identity=ident,
                current_camera=camera_id,
                consecutive_candidate_count=p_count
            )
            # Factor in spatial & temporal gate scores as supporting multiplier (no harsh penalty on strong ReID)
            cand_eval = evals.get(cand_gid, {})
            t_score = cand_eval.get("temporal_score", 1.0)
            s_score = cand_eval.get("spatial_score", 1.0)
            gate_factor = 0.92 + 0.04 * min(1.0, t_score) + 0.04 * min(1.0, s_score)
            adjusted_score = score * gate_factor
            scored_candidates.append((cand_gid, float(adjusted_score), details))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_gid, best_sim, best_details = scored_candidates[0]
        second_sim = scored_candidates[1][1] if len(scored_candidates) > 1 else 0.0
        margin = best_sim - second_sim

        decision_data["top_candidate"] = best_gid
        decision_data["top_score"] = round(best_sim, 3)
        decision_data["second_score"] = round(second_sim, 3)
        decision_data["margin"] = round(margin, 3)
        decision_data["candidate_details"] = best_details

        # -------------------------------------------------------------
        # Step 5: Multi-Frame Confirmation & Promotion
        # -------------------------------------------------------------
        thresh_match = float(self.config.get("threshold_match", 0.80))
        thresh_ambig = float(self.config.get("threshold_ambiguous", 0.68))
        thresh_cross = float(self.config.get("threshold_cross_view", 0.62))
        ambig_margin = float(self.config.get("reid_ambiguous_margin", 0.04))
        pending_promote_frames = int(self.config.get("pending_confirm_frames", 3))

        is_cross = best_details.get("is_cross_view", False)

        cand_eval = evals.get(best_gid, {})
        has_matching_transition = (cand_eval.get("spatial_score", 0.0) >= 0.95)

        # Immediate Confirmation Check
        if (best_sim >= thresh_match or (has_matching_transition and best_sim >= 0.70)) and (margin >= ambig_margin or second_sim == 0.0):
            # Confirm immediately!
            confirmed = self.journey_manager.confirm_association(
                global_id=best_gid,
                camera_id=camera_id,
                zone=zone,
                local_track_id=local_track_id,
                timestamp=current_time,
                frame_idx=frame_idx,
                decision_reason=f"CONFIRMED_MATCH (Score={best_sim:.2f}, Margin={margin:.2f})",
                decision_details=decision_data
            )
            if confirmed:
                self.pending_candidate[key] = None
                self.pending_counters[key] = 0
                self.no_match_counters[key] = 0
                decision_data["decision"] = "CONFIRMED"
                decision_data["selected_global_id"] = best_gid
                self.store.log_association_decision(decision_data)
                return best_gid, "MATCHED", best_sim, decision_data
            else:
                self.pending_candidate[key] = best_gid
                self.pending_counters[key] = 1
                decision_data["decision"] = f"PENDING_CANDIDATE ({best_gid}, 1/{pending_promote_frames})"
                self.store.log_association_decision(decision_data)
                return best_gid, "PENDING", best_sim, decision_data

        # PENDING Candidate Check (borderline or cross-view awaiting evidence)
        elif (is_cross and best_sim >= thresh_cross) or (best_sim >= thresh_ambig) or (has_matching_transition and best_sim >= 0.60):
            prev_cand = self.pending_candidate.get(key)
            if prev_cand == best_gid:
                self.pending_counters[key] = self.pending_counters.get(key, 0) + 1
            else:
                self.pending_candidate[key] = best_gid
                self.pending_counters[key] = 1

            p_count = self.pending_counters[key]
            decision_data["pending_count"] = p_count
            decision_data["pending_threshold"] = pending_promote_frames

            if p_count >= pending_promote_frames or (dom_view == "FRONT" and best_sim >= 0.72) or (has_matching_transition and best_sim >= 0.65):
                confirmed = self.journey_manager.confirm_association(
                    global_id=best_gid,
                    camera_id=camera_id,
                    zone=zone,
                    local_track_id=local_track_id,
                    timestamp=current_time,
                    frame_idx=frame_idx,
                    decision_reason=f"CONFIRMED_FROM_PENDING ({p_count} sustained checks)",
                    decision_details=decision_data
                )
                if confirmed:
                    self.pending_candidate[key] = None
                    self.pending_counters[key] = 0
                    self.no_match_counters[key] = 0
                    decision_data["decision"] = "CONFIRMED_FROM_PENDING"
                    decision_data["selected_global_id"] = best_gid
                    self.store.log_association_decision(decision_data)
                    return best_gid, "MATCHED", best_sim, decision_data
                else:
                    decision_data["decision"] = f"CONFLICT_DEFERRED ({best_gid})"
                    self.store.log_association_decision(decision_data)
                    return best_gid, "PENDING", best_sim, decision_data
            else:
                self.no_match_counters[key] = 0
                decision_data["decision"] = f"PENDING_CANDIDATE ({best_gid}, {p_count}/{pending_promote_frames})"
                self.store.log_association_decision(decision_data)
                return best_gid, "PENDING", best_sim, decision_data

        # NO_MATCH: Below threshold
        self.no_match_counters[key] = self.no_match_counters.get(key, 0) + 1
        new_id_confirm = int(self.config.get("reid_new_id_confirm_frames", 15))
        decision_data["decision"] = "NO_MATCH_BELOW_THRESH"
        decision_data["no_match_count"] = self.no_match_counters[key]

        if self.no_match_counters[key] >= new_id_confirm:
            new_gid = self.gallery.add_identity([{"camera_id": camera_id, "track_id": local_track_id}], bank_embeddings)
            self.journey_manager.register_new_identity(
                global_id=new_gid,
                camera_id=camera_id,
                zone=zone,
                local_track_id=local_track_id,
                timestamp=current_time,
                frame_idx=frame_idx
            )
            self.no_match_counters[key] = 0
            self.pending_candidate[key] = None
            self.pending_counters[key] = 0
            decision_data["selected_global_id"] = new_gid
            decision_data["decision"] = "NEW_GLOBAL_ID_CREATED"
            self.store.log_association_decision(decision_data)
            return new_gid, "MATCHED", 1.0, decision_data

        self.store.log_association_decision(decision_data)
        return None, "PENDING", 0.0, decision_data
