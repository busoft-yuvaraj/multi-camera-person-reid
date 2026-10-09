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
        transition_rules: Optional[List[Dict[str, Any]]] = None
    ):
        self.store = store
        self.reacquisition_timeout_seconds = reacquisition_timeout_seconds
        self.transition_rules = transition_rules or []
        
        # Registry of all Global Identities: gid -> GlobalIdentityState
        self.identities: Dict[str, GlobalIdentityState] = {}
        
        # Currently active track assignments: (camera_id, local_track_id) -> gid
        self.active_tracks: Dict[Tuple[str, int], str] = {}
        
        # Reverse mapping: gid -> (camera_id, local_track_id)
        self.gid_to_active_track: Dict[str, Tuple[str, int]] = {}
        
        # Buffer of pending events for unconfirmed tracks: (camera_id, local_track_id) -> list of JourneyEvents
        self.pending_events: Dict[Tuple[str, int], List[JourneyEvent]] = {}

    def get_identity(self, global_id: str) -> Optional[GlobalIdentityState]:
        return self.identities.get(global_id)

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
        self._set_active_mapping(camera_id, local_track_id, global_id)

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

    def confirm_association(
        self,
        global_id: str,
        camera_id: str,
        zone: str,
        local_track_id: int,
        timestamp: float,
        frame_idx: int,
        decision_reason: str = "CONFIRMED_MATCH",
        decision_details: Optional[Dict[str, Any]] = None
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
        current_active = self.gid_to_active_track.get(global_id)
        if current_active is not None and current_active != (camera_id, local_track_id):
            active_cam, active_tid = current_active
            
            # Case 1: Same camera replacement (e.g. BoT-SORT dropped previous local track ID)
            if active_cam == camera_id:
                logging.info(
                    f"[JOURNEY] Track replacement in {camera_id}: {global_id} moved from Track {active_tid} to Track {local_track_id}"
                )
                self.active_tracks.pop((active_cam, active_tid), None)
            else:
                # Case 2: Cross-camera transition
                # Real conflict only if seen simultaneously in two distinct cameras within < 1.0s
                # UNLESS last transition was specifically directed towards camera_id or target zone!
                is_expected_transition = False
                if getattr(state, "last_transition_target_camera", None) == camera_id:
                    is_expected_transition = True
                elif getattr(state, "last_transition_target_zone", None) == zone:
                    is_expected_transition = True
                elif self.transition_rules and state.last_transition_id:
                    for rule in self.transition_rules:
                        if rule.get("from_camera") == active_cam and rule.get("transition") == state.last_transition_id:
                            c_zones = rule.get("candidate_zones", [])
                            if camera_id in c_zones or zone in c_zones:
                                is_expected_transition = True
                                break
                time_diff = abs(timestamp - state.last_seen_timestamp)
                if not is_expected_transition and time_diff < 1.0 and state.status == IdentityStatus.ACTIVE and active_cam != camera_id:
                    conflict_evt = JourneyEvent(
                        event="GLOBAL_ID_CONFLICT",
                        global_id=global_id,
                        camera_id=camera_id,
                        zone=zone,
                        local_track_id=local_track_id,
                        timestamp=timestamp,
                        frame_idx=frame_idx,
                        details={
                            "reason": f"Simultaneous assignment: already active in {active_cam} (Track {active_tid})",
                            "conflict_camera": active_cam,
                            "conflict_track_id": active_tid
                        }
                    )
                    state.add_event(conflict_evt)
                    self.store.log_event(conflict_evt)
                    logging.error(
                        f"[CONFLICT] {global_id} already active in {active_cam} Track {active_tid}. "
                        f"Cannot assign to {camera_id} Track {local_track_id} simultaneously!"
                    )
                    return False
                else:
                    logging.info(
                        f"[JOURNEY] Transitioning {global_id} from {active_cam} (Track {active_tid}) to {camera_id} (Track {local_track_id})"
                    )
                    self.active_tracks.pop((active_cam, active_tid), None)

        is_reacquisition = (state.current_camera != camera_id) or (state.status in [IdentityStatus.TEMPORARILY_LOST, IdentityStatus.REACQUIRABLE])

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

        self._set_active_mapping(camera_id, local_track_id, global_id)

        # Log confirmation or reacquisition event
        evt_type = "GLOBAL_ID_REACQUIRED" if is_reacquisition else "GLOBAL_ID_CONFIRMED"
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
        gid = self.active_tracks.get((camera_id, local_track_id))
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
        gid = self.active_tracks.pop(key, None)
        if gid:
            if self.gid_to_active_track.get(gid) == key:
                self.gid_to_active_track.pop(gid, None)
            state = self.identities.get(gid)
            if state:
                state.status = IdentityStatus.TEMPORARILY_LOST
                state.last_seen_timestamp = timestamp

                evt = JourneyEvent(
                    event="CAMERA_DISAPPEARED",
                    global_id=gid,
                    camera_id=camera_id,
                    zone=zone,
                    local_track_id=local_track_id,
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    details={"status": "TEMPORARILY_LOST"}
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
                "target_zone": event.target_zone
            }
        )

        if gid and gid in self.identities:
            state = self.identities[gid]
            state.last_transition_id = event.transition_id
            state.last_transition_timestamp = event.timestamp
            state.last_transition_direction = event.direction
            state.last_transition_target_camera = event.target_camera
            state.last_transition_target_zone = event.target_zone
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
        """
        for gid, state in self.identities.items():
            if state.status == IdentityStatus.TEMPORARILY_LOST:
                elapsed = state.time_since_last_seen(current_time)
                if elapsed > self.reacquisition_timeout_seconds:
                    state.status = IdentityStatus.EXPIRED
                    evt = JourneyEvent(
                        event="GLOBAL_ID_EXPIRED",
                        global_id=gid,
                        camera_id=state.current_camera or "UNKNOWN",
                        zone=state.current_zone or "UNKNOWN",
                        local_track_id=state.current_local_track_id or -1,
                        timestamp=current_time,
                        frame_idx=frame_idx,
                        details={"elapsed_seconds": round(elapsed, 2)}
                    )
                    state.add_event(evt)
                    self.store.log_event(evt)

    def _set_active_mapping(self, camera_id: str, local_track_id: int, global_id: str):
        key = (camera_id, local_track_id)
        
        # If this track was previously mapped to an old GID, unmap that old GID
        old_gid = self.active_tracks.get(key)
        if old_gid and old_gid != global_id:
            if self.gid_to_active_track.get(old_gid) == key:
                self.gid_to_active_track.pop(old_gid, None)
                
        # If this GID was previously mapped to an old track, unmap that old track
        old_key = self.gid_to_active_track.get(global_id)
        if old_key and old_key != key:
            self.active_tracks.pop(old_key, None)
            
        self.active_tracks[key] = global_id
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
