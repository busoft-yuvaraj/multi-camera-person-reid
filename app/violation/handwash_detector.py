import cv2
import os
import math
import time
import logging
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Any

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


def box_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def bottom_center(box: Tuple[int, int, int, int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int(y2)


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def point_inside_expanded_box(
    point: Tuple[float, float],
    box: Tuple[int, int, int, int],
    expand_x: float = 0.15,
    expand_y: float = 0.20
) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    ex = int(width * expand_x)
    ey = int(height * expand_y)
    return (x1 - ex <= px <= x2 + ex) and (y1 - ey <= py <= y2 + ey)


def hand_is_near_sink(
    hand_box: Tuple[int, int, int, int],
    sink_box: Tuple[int, int, int, int],
    horizontal_margin: float = 0.30,
    top_margin: float = 1.50,
    bottom_margin: float = 0.20
) -> bool:
    sx1, sy1, sx2, sy2 = sink_box
    hand_x, hand_y = box_center(hand_box)

    sink_width = max(1, sx2 - sx1)
    sink_height = max(1, sy2 - sy1)

    allowed_x1 = sx1 - int(sink_width * horizontal_margin)
    allowed_x2 = sx2 + int(sink_width * horizontal_margin)
    horizontal_match = allowed_x1 <= hand_x <= allowed_x2

    allowed_y1 = sy1 - int(sink_height * top_margin)
    allowed_y2 = sy2 + int(sink_height * bottom_margin)
    vertical_match = allowed_y1 <= hand_y <= allowed_y2

    return horizontal_match and vertical_match


def assign_person_to_sink(
    person_box: Tuple[int, int, int, int],
    sink_boxes: List[Tuple[Tuple[int, int, int, int], float]],
    max_person_sink_dist_mult: float = 2.0
) -> Optional[int]:
    if not sink_boxes:
        return None

    person_point = bottom_center(person_box)
    best_index = None
    best_distance = float("inf")

    for sink_index, (sink_box, _conf) in enumerate(sink_boxes):
        sx1, sy1, sx2, sy2 = sink_box
        sink_center = int((sx1 + sx2) / 2), int((sy1 + sy2) / 2)
        dist = point_distance(person_point, sink_center)

        sink_width = max(1, sx2 - sx1)
        sink_height = max(1, sy2 - sy1)
        max_dist = max(sink_width, sink_height) * max_person_sink_dist_mult

        if dist <= max_dist and dist < best_distance:
            best_distance = dist
            best_index = sink_index

    return best_index


def get_current_step(step_result, step_model, step_confidence: float = 0.35):
    current_step = None
    current_confidence = 0.0
    current_box = None

    if step_result is None or step_result.boxes is None or len(step_result.boxes) == 0:
        return current_step, current_confidence, current_box

    for box, class_id, confidence in zip(
        step_result.boxes.xyxy,
        step_result.boxes.cls,
        step_result.boxes.conf,
    ):
        conf = float(confidence)
        if conf < step_confidence:
            continue
        cid = int(class_id)
        step_name = getattr(step_model, "names", {}).get(cid, f"step_{cid}")
        if conf > current_confidence:
            current_step = step_name
            current_confidence = conf
            current_box = tuple(box.int().tolist())

    return current_step, current_confidence, current_box


class HandwashDetector:
    """
    Modular Handwashing Detection Engine integrating:
    1. Sink detection model (models/yolo11n(best).pt)
    2. Hand/Hand-washing detection model (models/yolo_11n(best).pt)
    3. Step detection model (models/sink_model(best).pt)

    Reuses and maintains the proven temporal and geometric logic from hand_washing_with_log.py:
    - Person-to-sink spatial association
    - Hand-to-person association within expanded bounding box
    - Hand-to-sink proximity evaluation
    - Temporal confirmation of washing action
    - Hand movement tracking (exponential moving average)
    - Minimum washing duration requirement
    - Safe UNKNOWN -> HANDWASHED or NOT_HANDWASHED transitions
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        sink_model=None,
        hand_model=None,
        step_model=None,
        device: str = "cpu"
    ):
        self.config = config or {}
        self.device = device

        self.sink_model_path = self.config.get("sink_model_path", "models/yolo11n(best).pt")
        self.hand_model_path = self.config.get("hand_model_path", "models/yolo_11n(best).pt")
        self.step_model_path = self.config.get("step_model_path", "models/sink_model(best).pt")

        self.sink_confidence = float(self.config.get("sink_confidence", 0.30))
        self.hand_confidence = float(self.config.get("hand_confidence", 0.40))
        self.step_confidence = float(self.config.get("step_confidence", 0.35))

        self.min_washing_duration = float(self.config.get("min_washing_duration", 5.0))
        self.detection_confirmation_frames = int(self.config.get("detection_confirmation_frames", 3))
        self.reset_after_no_detection_seconds = float(self.config.get("reset_after_no_detection_seconds", 1.0))
        self.require_hand_movement = bool(self.config.get("require_hand_movement", True))
        self.min_hand_movement = float(self.config.get("min_hand_movement", 2.0))

        # Initialize or inject models
        self.sink_model = sink_model
        self.hand_model = hand_model
        self.step_model = step_model

        if YOLO is not None:
            if self.sink_model is None and os.path.exists(self.sink_model_path):
                try:
                    self.sink_model = YOLO(self.sink_model_path)
                    logging.info(f"[HandwashDetector] Loaded sink model from {self.sink_model_path}")
                except Exception as e:
                    logging.warning(f"[HandwashDetector] Failed loading sink model: {e}")

            if self.hand_model is None and os.path.exists(self.hand_model_path):
                try:
                    self.hand_model = YOLO(self.hand_model_path)
                    logging.info(f"[HandwashDetector] Loaded hand model from {self.hand_model_path}")
                except Exception as e:
                    logging.warning(f"[HandwashDetector] Failed loading hand model: {e}")

            if self.step_model is None and os.path.exists(self.step_model_path):
                try:
                    self.step_model = YOLO(self.step_model_path)
                    logging.info(f"[HandwashDetector] Loaded step model from {self.step_model_path}")
                except Exception as e:
                    logging.warning(f"[HandwashDetector] Failed loading step model: {e}")

        # Local track states: track_id -> dict
        self.person_states: Dict[int, Dict[str, Any]] = defaultdict(self._create_empty_state)

        # Cache of last detection results for rendering
        self.last_sink_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
        self.last_hand_assignments: List[Dict[str, Any]] = []
        self.last_step_info: Tuple[Optional[str], float, Optional[Tuple[int, int, int, int]]] = (None, 0.0, None)

    def _create_empty_state(self) -> Dict[str, Any]:
        return {
            "sink_index": None,
            "hand_centers": [],
            "movement_history": deque(maxlen=12),
            "movement_score": 0.0,
            "hand_near_sink": False,
            "step_detected": False,
            "washing": False,
            "action_state": "NOT_NEAR_SINK",
            "handwash_status": "UNKNOWN",  # "UNKNOWN", "HANDWASHED", "NOT_HANDWASHED"
            "positive_frames": 0,
            "no_detection_frames": 0,
            "entered_sink_frame": None,
            "washing_start_frame": None,
            "last_seen_frame": 0,
            "washing_duration": 0.0,
            "global_id": None
        }

    def process_frame(
        self,
        frame,
        tracks: List[Any],
        frame_idx: int,
        fps: float = 25.0,
        global_id_map: Optional[Dict[int, str]] = None
    ) -> Dict[int, Dict[str, Any]]:
        """
        Processes a video frame from the pantry camera:
        1. Identifies sinks and hands
        2. Associates hands to persons and sinks
        3. Tracks handwashing motion and duration
        4. Updates each person's Handwash state
        """
        if fps <= 0:
            fps = 25.0

        reset_frames = max(1, int(fps * self.reset_after_no_detection_seconds))
        global_id_map = global_id_map or {}

        # Parse person detections from tracks: list of [x1, y1, x2, y2, track_id, conf, ...]
        persons = []
        for t in tracks:
            x1, y1, x2, y2, tid = t[:5]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            tid = int(tid)
            persons.append({"id": tid, "box": (x1, y1, x2, y2)})

        # 1. SINK DETECTION
        sink_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
        if self.sink_model is not None and frame is not None:
            try:
                res = self.sink_model.predict(
                    frame,
                    imgsz=640,
                    conf=self.sink_confidence,
                    verbose=False,
                    device=self.device
                )[0]
                if res.boxes is not None:
                    for b, cid, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                        if int(cid) == 0:
                            sink_boxes.append((tuple(b.int().tolist()), float(conf)))
            except Exception as e:
                logging.debug(f"[HandwashDetector] Sink predict error: {e}")

        self.last_sink_boxes = sink_boxes

        # 2. HAND DETECTION
        hand_boxes: List[Tuple[Tuple[int, int, int, int], float]] = []
        if self.hand_model is not None and frame is not None:
            try:
                res = self.hand_model.predict(
                    frame,
                    imgsz=640,
                    conf=self.hand_confidence,
                    verbose=False,
                    device=self.device
                )[0]
                if res.boxes is not None:
                    for b, cid, conf in zip(res.boxes.xyxy, res.boxes.cls, res.boxes.conf):
                        if int(cid) == 0:
                            hand_boxes.append((tuple(b.int().tolist()), float(conf)))
            except Exception as e:
                logging.debug(f"[HandwashDetector] Hand predict error: {e}")

        # 3. STEP DETECTION (supporting evidence)
        detected_step, detected_step_conf, detected_step_box = None, 0.0, None
        if self.step_model is not None and frame is not None:
            try:
                res = self.step_model.predict(
                    frame,
                    imgsz=640,
                    conf=self.step_confidence,
                    verbose=False,
                    device=self.device
                )[0]
                detected_step, detected_step_conf, detected_step_box = get_current_step(
                    res, self.step_model, self.step_confidence
                )
            except Exception as e:
                logging.debug(f"[HandwashDetector] Step predict error: {e}")

        self.last_step_info = (detected_step, detected_step_conf, detected_step_box)

        # 4. ASSIGN EACH PERSON TO SINK
        for person in persons:
            pid = person["id"]
            state = self.person_states[pid]
            state["last_seen_frame"] = frame_idx
            state["sink_index"] = assign_person_to_sink(person["box"], sink_boxes)
            state["hand_centers"] = []
            state["hand_near_sink"] = False
            state["step_detected"] = False

            gid = global_id_map.get(pid)
            if gid and str(gid).upper() not in ["PENDING", "AMBIGUOUS", "UNKNOWN"]:
                state["global_id"] = gid

        # 5. ASSIGN HANDS TO PERSONS
        hand_assignments = []
        for hand_box, hand_conf in hand_boxes:
            h_center = box_center(hand_box)
            best_person_id = None
            best_dist = float("inf")

            for person in persons:
                p_box = person["box"]
                if not point_inside_expanded_box(h_center, p_box):
                    continue
                p_center = box_center(p_box)
                dist = point_distance(h_center, p_center)
                if dist < best_dist:
                    best_dist = dist
                    best_person_id = person["id"]

            hand_assignments.append({
                "box": hand_box,
                "confidence": hand_conf,
                "center": h_center,
                "person_id": best_person_id
            })

        self.last_hand_assignments = hand_assignments

        # 6. ASSOCIATE HANDS TO PERSON'S SINK
        for assignment in hand_assignments:
            pid = assignment["person_id"]
            if pid is None or pid not in self.person_states:
                continue

            h_box = assignment["box"]
            h_center = assignment["center"]
            state = self.person_states[pid]
            state["hand_centers"].append(h_center)

            sink_idx = state["sink_index"]
            if sink_idx is not None and sink_idx < len(sink_boxes):
                sink_box, _sconf = sink_boxes[sink_idx]
                if hand_is_near_sink(h_box, sink_box):
                    state["hand_near_sink"] = True

        # 7. HAND MOVEMENT CALCULATION
        for person in persons:
            pid = person["id"]
            state = self.person_states[pid]
            centers = state["hand_centers"]

            if centers:
                avg_x = sum(p[0] for p in centers) / len(centers)
                avg_y = sum(p[1] for p in centers) / len(centers)
                current_c = (int(avg_x), int(avg_y))
                state["movement_history"].append(current_c)

                if len(state["movement_history"]) >= 2:
                    movement = point_distance(
                        state["movement_history"][-1],
                        state["movement_history"][-2]
                    )
                    state["movement_score"] = 0.8 * state["movement_score"] + 0.2 * movement
            else:
                state["movement_score"] *= 0.8

        # 8. STEP DETECTION ASSOCIATION
        if detected_step_box is not None and persons:
            step_center = box_center(detected_step_box)
            best_person_id = None
            best_dist = float("inf")

            for person in persons:
                p_center = box_center(person["box"])
                dist = point_distance(step_center, p_center)
                if dist < best_dist:
                    best_dist = dist
                    best_person_id = person["id"]

            if best_person_id is not None:
                self.person_states[best_person_id]["step_detected"] = True

        # 9. EVALUATE WASHING DECISION PER PERSON
        for person in persons:
            pid = person["id"]
            state = self.person_states[pid]

            has_hand = len(state["hand_centers"]) >= 1
            near_sink = state["hand_near_sink"]
            moving = state["movement_score"] >= self.min_hand_movement

            if self.require_hand_movement:
                washing_evidence = has_hand and near_sink and moving
            else:
                washing_evidence = has_hand and near_sink

            if washing_evidence:
                state["positive_frames"] += 1
                state["no_detection_frames"] = 0
            else:
                state["positive_frames"] = 0
                state["no_detection_frames"] += 1

            # Near sink state
            if near_sink:
                if state["entered_sink_frame"] is None:
                    state["entered_sink_frame"] = frame_idx
                    if state["handwash_status"] != "HANDWASHED":
                        state["action_state"] = "NEAR_SINK_NOT_WASHING"

                # Confirm washing started
                if not state["washing"] and state["positive_frames"] >= self.detection_confirmation_frames:
                    state["washing"] = True
                    state["washing_start_frame"] = frame_idx
                    state["action_state"] = "WASHING"

                # If washing, track duration and promote to HANDWASHED once threshold reached
                if state["washing"] and state["washing_start_frame"] is not None:
                    duration = (frame_idx - state["washing_start_frame"]) / max(fps, 1.0)
                    state["washing_duration"] = duration
                    if duration >= self.min_washing_duration:
                        state["handwash_status"] = "HANDWASHED"
                        state["action_state"] = "WASHED"

            # Left sink area
            if not near_sink and state["entered_sink_frame"] is not None and state["no_detection_frames"] >= reset_frames:
                if state["washing"]:
                    duration = 0.0
                    if state["washing_start_frame"] is not None:
                        duration = (frame_idx - state["washing_start_frame"]) / max(fps, 1.0)
                    state["washing_duration"] = duration

                    if duration >= self.min_washing_duration:
                        state["handwash_status"] = "HANDWASHED"
                        state["action_state"] = "WASHED"
                    else:
                        # Session ended without reaching required duration
                        if state["handwash_status"] != "HANDWASHED":
                            state["handwash_status"] = "NOT_HANDWASHED"
                            state["action_state"] = "NOT_WASHED"
                    state["washing"] = False
                    state["washing_start_frame"] = None
                else:
                    # In sink area without washing
                    if state["handwash_status"] != "HANDWASHED":
                        state["handwash_status"] = "NOT_HANDWASHED"
                        state["action_state"] = "NOT_WASHED"

                state["entered_sink_frame"] = None
                state["positive_frames"] = 0
                state["no_detection_frames"] = 0

        return dict(self.person_states)

    def on_track_lost(self, track_id: int) -> str:
        """
        Called when a person track is lost from the pantry camera.
        If the person was in the pantry and never handwashed, transitions UNKNOWN -> NOT_HANDWASHED.
        Preserves HANDWASHED status.
        """
        state = self.person_states[track_id]
        if state["handwash_status"] == "UNKNOWN":
            state["handwash_status"] = "NOT_HANDWASHED"
            state["action_state"] = "NOT_WASHED"
        return state["handwash_status"]

    def get_status(self, track_id: int) -> str:
        """
        Returns the handwash status ("UNKNOWN", "HANDWASHED", "NOT_HANDWASHED") for a track.
        """
        if track_id in self.person_states:
            return self.person_states[track_id]["handwash_status"]
        return "UNKNOWN"

    def set_status(self, track_id: int, status: str):
        """
        Allows explicitly updating or mocking handwash status for a track.
        """
        state = self.person_states[track_id]
        norm = str(status).upper().strip()
        if state["handwash_status"] == "HANDWASHED" and norm != "HANDWASHED":
            return
        state["handwash_status"] = norm

    def get_track_state(self, track_id: int) -> Dict[str, Any]:
        return self.person_states[track_id]
