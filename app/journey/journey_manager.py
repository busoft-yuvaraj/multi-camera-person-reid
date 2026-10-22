import logging
from typing import Dict, List, Optional, Tuple, Any
from .models import IdentityStatus, JourneyEvent, GlobalIdentityState
from .journey_store import JourneyStore
from app.events.transition_detector import TransitionEvent
from app.events.roi_detector import RoiEvent

class JourneyManager:
    """
    Central Journey and Global Identity State Manager for Phase 1.
    Responsibilities:
      - Tracks lifecycle & metadata for each Global ID across cameras and zones.
      - Maintains local-track to Global ID mappings.
      - Buffers pending ROI and transition events for local tracks, and retroactively
        attaches them when a Global ID is confirmed.
      - Detects impossible simultaneous assignments (GLOBAL_ID_CONFLICT).
      - Manages track disappearance, reacquisition, and expiration.
      - Emits structured events to JourneyStore.
    """
    def __init__(
        self,
        store: JourneyStore,
        reacquisition_timeout_seconds: float = 20.0,
        transition_rules: Optional[List[Dict[str, Any]]] = None,
        topology: Optional[Dict[str, Any]] = None,
        overlapping_camera_pairs: Optional[List[Any]] = None
    ):
        self.store = store
        self.reacquisition_timeout_seconds = reacquisition_timeout_seconds
        self.transition_rules = transition_rules or []
        self.topology = topology or {
            "pantry": ["waiting_lobby", "passage"],
            "waiting_lobby": ["pantry", "passage"],
            "passage": ["pantry", "waiting_lobby", "workspace"],
            "workspace": ["passage"]
        }
        raw_pairs = overlapping_camera_pairs or []
        self.overlapping_camera_pairs = [set(p) for p in raw_pairs]
        
        # Registry of all Global Identities: gid -> GlobalIdentityState
        self.identities: Dict[str, GlobalIdentityState] = {}
        
        # Currently active track assignments: (camera_id, local_track_id) -> gid
        self.active_tracks: Dict[Tuple[str, int], str] = {}
        
        # Reverse mapping: gid -> (camera_id, local_track_id) (latest primary)
        self.gid_to_active_track: Dict[str, Tuple[str, int]] = {}
        
        # Per-camera active mappings: gid -> {camera_id: local_track_id}
        self.gid_to_camera_tracks: Dict[str, Dict[str, int]] = {}
        
        # Buffer of pending events for unconfirmed tracks: (camera_id, local_track_id) -> list of JourneyEvents
        self.pending_events: Dict[Tuple[str, int], List[JourneyEvent]] = {}
        
        # Last observed timestamp per active track: (camera_id, local_track_id) -> timestamp
        self.track_last_seen: Dict[Tuple[str, int], float] = {}

    def get_identity(self, global_id: str) -> Optional[GlobalIdentityState]:
        return self.identities.get(global_id)

    def get_active_global_id(self, camera_id: str, local_track_id: int) -> Optional[str]:
        return self.active_tracks.get((camera_id, local_track_id))

    def get_all_identities(self) -> Dict[str, GlobalIdentityState]:
        return dict(self.identities)

    def register_new_identity(
        self,
        global_id: str,
        camera_id: str,
        zone: str,
        local_track_id: int,
        timestamp: float,
        frame_idx: int
    ) -> GlobalIdentityState:
        """
        Creates a new Global Identity in state CONFIRMED/ACTIVE.
        """
        state = GlobalIdentityState(
            global_id=global_id,
            status=IdentityStatus.ACTIVE,
            current_camera=camera_id,
            current_zone=zone,
            current_local_track_id=local_track_id,
            first_seen_timestamp=timestamp,
            last_seen_timestamp=timestamp
        )
        self.identities[global_id] = state
        self._set_active_mapping(camera_id, local_track_id, global_id, timestamp=timestamp)

        # Log creation event
        evt = JourneyEvent(
            event="TRACK_STARTED",
            global_id=global_id,
            camera_id=camera_id,
            zone=zone,
            local_track_id=local_track_id,
            timestamp=timestamp,
            frame_idx=frame_idx,
            details={"status": "NEW_GLOBAL_ID_CREATED"}
        )
        state.add_event(evt)
        self.store.log_event(evt)

        # Transfer any buffered events for this local track
        self._transfer_pending_events(camera_id, local_track_id, global_id)
        return state

    def is_overlapping_pair(self, cam1: Optional[str], cam2: Optional[str]) -> bool:
        if not cam1 or not cam2:
            return False
        return {cam1, cam2} in self.overlapping_camera_pairs

    def confirm_association(
        self,
        global_id: str,
        camera_id: str,
        zone: str,
        local_track_id: int,
        timestamp: float,
        frame_idx: int,
        decision_reason: str = "CONFIRMED_MATCH",
        decision_details: Optional[Dict[str, Any]] = None,
        is_overlapping_view: bool = False
    ) -> bool:
        """
        Confirms that local track belongs to existing Global ID.
        Checks for simultaneous assignment conflicts.
        """
        state = self.identities.get(global_id)
        if state is None:
            logging.error(f"[JOURNEY] Cannot confirm non-existent Global ID: {global_id}")
            return False

        # -------------------------------------------------------------
        # Conflict Detection: Is this Global ID already active elsewhere?
        # -------------------------------------------------------------
        active_entries = []
        if global_id in self.gid_to_camera_tracks and self.gid_to_camera_tracks[global_id]:
            for c_id, t_id in self.gid_to_camera_tracks[global_id].items():
                if (c_id, t_id) != (camera_id, local_track_id):
                    active_entries.append((c_id, t_id))
        else:
            current_active = self.gid_to_active_track.get(global_id)
            if current_active is not None and current_active != (camera_id, local_track_id):
                active_entries.append(current_active)

        has_overlap = (
            is_overlapping_view or
            bool(decision_details and decision_details.get("is_overlapping_view")) or
            (decision_reason in ["OVERLAPPING_VIEW", "SIMULTANEOUS_VISIBILITY"])
        )

        for active_cam, active_tid in active_entries:
            # Case 1: Same camera replacement (e.g. BoT-SORT dropped previous local track ID)
            if active_cam == camera_id:
                active_key = (active_cam, active_tid)
                last_active_time = self.track_last_seen.get(active_key, state.last_seen_timestamp)
                time_diff = abs(timestamp - last_active_time) if last_active_time is not None else 999.0
                # True conflict if active_tid is still active in self.active_tracks and was observed within 2.0s
                is_currently_active = (active_key in self.active_tracks) and (time_diff < 2.0)
                if is_currently_active:
                    conflict_evt = JourneyEvent(
                        event="GLOBAL_ID_CONFLICT",
                        global_id=global_id,
                        camera_id=camera_id,
                        zone=zone,
                        local_track_id=local_track_id,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        details={
                            "reason": f"Simultaneous assignment in same camera {camera_id}: already active on Track {active_tid} in current frame",
                            "conflict_camera": active_cam,
                            "conflict_track_id": active_tid
                        }
                    )
                    state.add_event(conflict_evt)
                    self.store.log_event(conflict_evt)
                    logging.error(
                        f"[CONFLICT] {global_id} is currently active in same camera {camera_id} Track {active_tid}. "
                        f"Cannot assign to Track {local_track_id} simultaneously!"
                    )
                    return False
                else:
                    logging.info(
                        f"[JOURNEY] Track replacement in {camera_id}: {global_id} moved from Track {active_tid} to Track {local_track_id}"
                    )
                    self.active_tracks.pop((active_cam, active_tid), None)
                    self.track_last_seen.pop((active_cam, active_tid), None)
            else:
                # Case 2: Cross-camera transition
                is_overlap = has_overlap or self.is_overlapping_pair(camera_id, active_cam)
                active_key = (active_cam, active_tid)
                last_active_time = self.track_last_seen.get(active_key, state.last_seen_timestamp)
                time_diff = abs(timestamp - last_active_time) if last_active_time is not None else 999.0

                # When camera ROIs do not overlap, simultaneous presence is a strict physical impossibility.
                if not is_overlap:
                    is_transition_handoff = False
                    if state.last_transition_source_camera == active_cam or (decision_details and decision_details.get("has_matching_transition")):
                        target_matches = (
                            state.last_transition_target_camera == camera_id or
                            camera_id in (state.last_transition_candidate_zones or []) or
                            bool(decision_details and decision_details.get("has_matching_transition"))
                        )
                        trans_time = state.last_transition_timestamp if state.last_transition_timestamp is not None else state.last_seen_timestamp
                        if target_matches and trans_time is not None:
                            trans_elapsed = abs(timestamp - trans_time)
                            if trans_elapsed <= 15.0:
                                is_transition_handoff = True

                    is_currently_active = (active_key in self.active_tracks) and (state.status == IdentityStatus.ACTIVE) and (time_diff < 1.0)
                    if is_currently_active and not is_transition_handoff:
                        conflict_evt = JourneyEvent(
                            event="GLOBAL_ID_CONFLICT",
                            global_id=global_id,
                            camera_id=camera_id,
                            zone=zone,
                            local_track_id=local_track_id,
                            timestamp=timestamp,
                            frame_idx=frame_idx,
                            details={
                                "reason": f"Simultaneous/un-transitioned assignment in non-overlapping ROIs: recently present in {active_cam} (Track {active_tid}, elapsed {time_diff:.1f}s)",
                                "conflict_camera": active_cam,
                                "conflict_track_id": active_tid
                            }
                        )
                        state.add_event(conflict_evt)
                        self.store.log_event(conflict_evt)
                        logging.error(
                            f"[CONFLICT] {global_id} is currently active or recently present in non-overlapping {active_cam} Track {active_tid}. "
                            f"Cannot assign to {camera_id} Track {local_track_id} without validated transition handoff!"
                        )
                        return False
                    else:
                        logging.info(
                            f"[JOURNEY] Cross-camera handoff: {global_id} transitioned from {active_cam} (Track {active_tid}) to {camera_id} (Track {local_track_id})"
                        )
                else:
                    logging.info(
                        f"[JOURNEY] Overlapping cross-camera association: {global_id} in {camera_id} (Track {local_track_id}) while also tracked in {active_cam} (Track {active_tid})"
                    )

        was_same_camera = (state.current_camera == camera_id)
        is_reacquisition = (not was_same_camera) or (state.status in [
            IdentityStatus.TEMPORARILY_LOST,
            IdentityStatus.OCCLUDED,
            IdentityStatus.REACQUIRABLE,
            IdentityStatus.DORMANT,
            IdentityStatus.EXPIRED
        ])

        # Update identity state
        if state.current_camera != camera_id:
            state.previous_camera = state.current_camera
            state.previous_zone = state.current_zone
            state.previous_local_track_id = state.current_local_track_id

        state.current_camera = camera_id
        state.current_zone = zone
        state.current_local_track_id = local_track_id
        state.last_seen_timestamp = timestamp
        state.status = IdentityStatus.ACTIVE

        # Consume transition handoff upon successful cross-camera confirmation
        if is_reacquisition:
            state.last_transition_id = None
            state.last_transition_direction = None
            state.last_transition_timestamp = None
            state.last_transition_target_camera = None
            state.last_transition_target_zone = None
            state.last_transition_candidate_zones = []

        self._set_active_mapping(camera_id, local_track_id, global_id, timestamp=timestamp)

        # Log confirmation, track recovery, or cross-camera reacquisition event
        if is_reacquisition and was_same_camera:
            evt_type = "TRACK_RECOVERED"
        elif is_reacquisition:
            evt_type = "GLOBAL_ID_REACQUIRED"
        else:
            evt_type = "GLOBAL_ID_CONFIRMED"
        evt = JourneyEvent(
            event=evt_type,
            global_id=global_id,
            camera_id=camera_id,
            zone=zone,
            local_track_id=local_track_id,
            timestamp=timestamp,
            frame_idx=frame_idx,
            details={
                "reason": decision_reason,
                "previous_camera": state.previous_camera,
                **(decision_details or {})
            }
        )
        state.add_event(evt)
        self.store.log_event(evt)

        # Transfer buffered events
        self._transfer_pending_events(camera_id, local_track_id, global_id)
        return True

    def update_track_presence(
        self,
        camera_id: str,
        local_track_id: int,
        timestamp: float,
        frame_idx: int
    ):
        """
        Updates the last seen timestamp for an active track.
        """
        key = (camera_id, local_track_id)
        self.track_last_seen[key] = timestamp
        gid = self.active_tracks.get(key)
        if gid and gid in self.identities:
            state = self.identities[gid]
            state.last_seen_timestamp = timestamp
            state.status = IdentityStatus.ACTIVE

    def on_track_lost(
        self,
        camera_id: str,
        zone: str,
        local_track_id: int,
        timestamp: float,
        frame_idx: int
    ):
        """
        Called when a local track disappears from camera view.
        Transitions the identity to TEMPORARILY_LOST / REACQUIRABLE rather than deleting it.
        """
        key = (camera_id, local_track_id)
        self.track_last_seen.pop(key, None)
        gid = self.active_tracks.pop(key, None)
        if gid:
            if gid in self.gid_to_camera_tracks and camera_id in self.gid_to_camera_tracks[gid]:
                if self.gid_to_camera_tracks[gid][camera_id] == local_track_id:
                    self.gid_to_camera_tracks[gid].pop(camera_id, None)

            if self.gid_to_active_track.get(gid) == key:
                if gid in self.gid_to_camera_tracks and self.gid_to_camera_tracks[gid]:
                    other_cam, other_tid = next(iter(self.gid_to_camera_tracks[gid].items()))
                    self.gid_to_active_track[gid] = (other_cam, other_tid)
                else:
                    self.gid_to_active_track.pop(gid, None)

            state = self.identities.get(gid)
            if state:
                has_other_active = bool(gid in self.gid_to_camera_tracks and self.gid_to_camera_tracks[gid])
                if not has_other_active:
                    state.status = IdentityStatus.OCCLUDED
                    state.last_seen_timestamp = timestamp

                evt = JourneyEvent(
                    event="CAMERA_DISAPPEARED",
                    global_id=gid,
                    camera_id=camera_id,
                    zone=zone,
                    local_track_id=local_track_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    details={"status": "OCCLUDED"}
                )
                state.add_event(evt)
                self.store.log_event(evt)

        # Clean up unconfirmed pending buffer if unused
        self.pending_events.pop(key, None)

    def record_transition(self, event: TransitionEvent):
        """
        Records a line transition event. If the track is already confirmed to a GID,
        updates GlobalIdentityState and logs immediately. Otherwise, buffers event.
        """
        key = (event.camera_id, event.local_track_id)
        gid = self.active_tracks.get(key) or event.global_id

        cand_dests = getattr(event, "candidate_destinations", None) or getattr(event, "candidate_zones", None) or []
        if not cand_dests and event.target_camera:
            cand_dests = [event.target_camera]
        status = getattr(event, "status", "PENDING")

        j_evt = JourneyEvent(
            event="TRANSITION_CROSSED",
            global_id=gid,
            camera_id=event.camera_id,
            zone=event.zone,
            local_track_id=event.local_track_id,
            timestamp=event.timestamp,
            frame_idx=event.frame_idx,
            details={
                "transition_id": event.transition_id,
                "direction": event.direction,
                "foot_point": [round(event.foot_point[0], 1), round(event.foot_point[1], 1)],
                "target_camera": event.target_camera,
                "target_zone": event.target_zone,
                "candidate_destinations": list(cand_dests),
                "status": status
            }
        )

        if gid and gid in self.identities:
            state = self.identities[gid]
            state.last_transition_source_camera = event.camera_id
            state.last_transition_id = event.transition_id
            state.last_transition_timestamp = event.timestamp
            state.last_transition_direction = event.direction
            state.last_transition_target_camera = event.target_camera
            state.last_transition_target_zone = event.target_zone
            state.last_transition_candidate_zones = list(cand_dests)
            state.last_transition_status = status
            state.add_event(j_evt)
            self.store.log_event(j_evt)
        else:
            if key not in self.pending_events:
                self.pending_events[key] = []
            self.pending_events[key].append(j_evt)

    def record_roi_event(self, event: RoiEvent):
        """
        Records ROI_ENTER or ROI_EXIT. If track is confirmed, updates state and logs immediately.
        Otherwise buffers event.
        """
        key = (event.camera_id, event.local_track_id)
        gid = self.active_tracks.get(key) or event.global_id

        details: Dict[str, Any] = {
            "roi_id": event.roi_id,
            "foot_point": [round(event.foot_point[0], 1), round(event.foot_point[1], 1)]
        }
        if event.duration_seconds is not None:
            details["duration_seconds"] = event.duration_seconds

        j_evt = JourneyEvent(
            event=event.event,
            global_id=gid,
            camera_id=event.camera_id,
            zone=event.zone,
            local_track_id=event.local_track_id,
            timestamp=event.timestamp,
            frame_idx=event.frame_idx,
            details=details
        )

        if gid and gid in self.identities:
            state = self.identities[gid]
            if event.event == "ROI_ENTER":
                state.current_roi = event.roi_id
                state.last_roi_enter_timestamp = event.timestamp
            elif event.event == "ROI_EXIT":
                state.last_roi = state.current_roi
                state.current_roi = None
                state.last_roi_exit_timestamp = event.timestamp

            state.add_event(j_evt)
            self.store.log_event(j_evt)
        else:
            if key not in self.pending_events:
                self.pending_events[key] = []
            self.pending_events[key].append(j_evt)

    def check_expirations(self, current_time: float, frame_idx: int):
        """
        Marks TEMPORARILY_LOST tracks as EXPIRED if unseen for longer than timeout.
        Also expires unresolved stale transitions (> 45s).
        """
        for gid, state in self.identities.items():
            if state.status in [IdentityStatus.OCCLUDED, IdentityStatus.TEMPORARILY_LOST]:
                elapsed = state.time_since_last_seen(current_time)
                if elapsed > self.reacquisition_timeout_seconds:
                    state.status = IdentityStatus.DORMANT
                    evt = JourneyEvent(
                        event="GLOBAL_ID_EXPIRED",
                        global_id=gid,
                        camera_id=state.current_camera or "UNKNOWN",
                        zone=state.current_zone or "UNKNOWN",
                        local_track_id=state.current_local_track_id or -1,
                        timestamp=current_time,
                        frame_idx=frame_idx,
                        details={"status": "DORMANT", "elapsed_seconds": round(elapsed, 2)}
                    )
                    state.add_event(evt)
                    self.store.log_event(evt)

            # Check transition expiration
            if state.last_transition_timestamp is not None:
                trans_elapsed = current_time - state.last_transition_timestamp
                if trans_elapsed > 45.0:
                    state.last_transition_status = "EXPIRED"
                    state.last_transition_id = None
                    state.last_transition_candidate_zones = []
                    state.last_transition_direction = None
                    state.last_transition_timestamp = None

    def _set_active_mapping(self, camera_id: str, local_track_id: int, global_id: str, timestamp: Optional[float] = None):
        key = (camera_id, local_track_id)
        
        # 1. If this track was previously mapped to an old GID, unmap that old GID for this camera
        old_gid = self.active_tracks.get(key)
        if old_gid and old_gid != global_id:
            if old_gid in self.gid_to_camera_tracks and camera_id in self.gid_to_camera_tracks[old_gid]:
                if self.gid_to_camera_tracks[old_gid][camera_id] == local_track_id:
                    self.gid_to_camera_tracks[old_gid].pop(camera_id, None)
                
        # 2. In THIS camera, if this GID was previously mapped to another track in this camera, unmap it
        if global_id in self.gid_to_camera_tracks and camera_id in self.gid_to_camera_tracks[global_id]:
            old_tid = self.gid_to_camera_tracks[global_id][camera_id]
            if old_tid != local_track_id:
                self.active_tracks.pop((camera_id, old_tid), None)
            
        # 3. Store mapping independently per camera
        if timestamp is not None:
            self.track_last_seen[key] = timestamp
        self.active_tracks[key] = global_id
        if global_id not in self.gid_to_camera_tracks:
            self.gid_to_camera_tracks[global_id] = {}
        self.gid_to_camera_tracks[global_id][camera_id] = local_track_id
        self.gid_to_active_track[global_id] = key

    def _transfer_pending_events(self, camera_id: str, local_track_id: int, global_id: str):
        key = (camera_id, local_track_id)
        buffered = self.pending_events.pop(key, [])
        state = self.identities.get(global_id)
        if state and buffered:
            for evt in buffered:
                evt.global_id = global_id
                state.add_event(evt)
                self.store.log_event(evt)
                
                # Apply transition / ROI attributes to state retroactively if present
                if evt.event == "TRANSITION_CROSSED":
                    state.last_transition_source_camera = evt.camera_id
                    state.last_transition_id = evt.details.get("transition_id")
                    state.last_transition_timestamp = evt.timestamp
                    state.last_transition_direction = evt.details.get("direction")
                    state.last_transition_target_camera = evt.details.get("target_camera")
                    state.last_transition_target_zone = evt.details.get("target_zone")
                elif evt.event == "ROI_ENTER":
                    state.current_roi = evt.details.get("roi_id")
                    state.last_roi_enter_timestamp = evt.timestamp
                elif evt.event == "ROI_EXIT":
                    state.last_roi = evt.details.get("roi_id")
                    state.current_roi = None
                    state.last_roi_exit_timestamp = evt.timestamp
