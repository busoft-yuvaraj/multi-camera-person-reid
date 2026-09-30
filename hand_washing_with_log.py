from ultralytics import YOLO
import cv2
import os
import time
import math
import torch
from collections import defaultdict, deque


# ============================================================
# CONFIG
# ============================================================

PERSON_MODEL_PATH = "yolo11n.pt"
PERSON_CLASS_ID = 0

HAND_MODEL_PATH = "yolo_11n_19_08_e-35.pt"
HAND_CLASS_ID = 0

SINK_MODEL_PATH = "sink_model.pt"
SINK_CLASS_ID = 0

HAND_STEP_MODEL_PATH = "best.pt"

VIDEO_PATH = "VID_20260903_161258.mp4"
OUTPUT_PATH = "output(7).mp4"
LOG_FILE_PATH = "handwashing_log.txt"

# Minimum confirmed washing duration required to mark a session as WASHED.
MIN_WASHING_DURATION = 5.0

DEVICE = 0 if torch.cuda.is_available() else "cpu"


# ============================================================
# CONFIDENCE
# ============================================================

PERSON_CONFIDENCE = 0.25
HAND_CONFIDENCE = 0.50
SINK_CONFIDENCE = 0.50
STEP_CONFIDENCE = 0.40


# ============================================================
# PROCESSING FPS
#
# IMPORTANT:
# This controls inference frequency, not the FPS of the
# saved output video.
#
# No person -> approximately 10 inference FPS
# Person present -> approximately 30 inference FPS
#
# If your source video itself is below 30 FPS, the maximum
# possible processing rate is the source FPS.
# ============================================================

NO_PERSON_PROCESS_FPS = 5.0
PERSON_PROCESS_FPS = 10.0


# ============================================================
# PERSON / HAND / SINK ASSOCIATION
# ============================================================

# A hand belongs to a person if the hand center is inside
# the expanded person bounding box.
PERSON_BOX_EXPAND_X = 0.15
PERSON_BOX_EXPAND_Y = 0.20

# A person is assigned to a sink using the closest sink
# to the person's lower-center point.
#
# Maximum distance is relative to the sink size.
MAX_PERSON_SINK_DISTANCE = 2.0

# Hand-to-sink relation uses the same idea as your original
# script, but is evaluated ONLY against that person's sink.
SINK_HORIZONTAL_MARGIN = 0.30
SINK_TOP_MARGIN = 1.50
SINK_BOTTOM_MARGIN = 0.20


# ============================================================
# WASHING DECISION
# ============================================================

# A hand must be associated with a person and that person's
# sink.
MIN_HANDS_FOR_WASHING = 1

# Number of consecutive positive frames before declaring
# washing.
DETECTION_CONFIRMATION_FRAMES = 3

# Frames without evidence before resetting the event.
RESET_AFTER_NO_DETECTION_SECONDS = 1.0

# Movement history for each person's hand.
MOVEMENT_HISTORY_SIZE = 12

# Minimum average hand movement in pixels/frame.
# You should tune this for your camera.
MIN_HAND_MOVEMENT = 2.0

# If step_1 ... step_6 are detected for the person, this
# increases confidence that the person is actually washing.
USE_STEP_MODEL_AS_SUPPORT = True

# Number of washing-related conditions needed.
#
# Condition A: person's hand is near that person's sink
# Condition B: hand movement is present
# Condition C: step model detects a step for that person
#
# We use A + B as the main decision.
# C is supporting evidence.
REQUIRE_HAND_MOVEMENT = True


# ============================================================
# DRAW COLORS - BGR
# ============================================================

SINK_COLOR = (0, 255, 0)
PERSON_COLOR = (255, 255, 0)
VALID_HAND_COLOR = (255, 0, 0)
INVALID_HAND_COLOR = (0, 0, 255)
STEP_COLOR = (255, 0, 255)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
WASHING_COLOR = (0, 255, 0)
NOT_WASHING_COLOR = (0, 0, 255)


# ============================================================
# HELPERS
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2),
    )


def bottom_center(box):
    x1, y1, x2, y2 = box
    return (
        int((x1 + x2) / 2),
        int(y2),
    )


def point_distance(p1, p2):
    return math.hypot(
        p1[0] - p2[0],
        p1[1] - p2[1],
    )


def draw_label(frame, text, origin, color, scale=0.60):
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def point_inside_expanded_box(point, box, expand_x=0.15, expand_y=0.20):
    px, py = point
    x1, y1, x2, y2 = box

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    ex = int(width * expand_x)
    ey = int(height * expand_y)

    return (
        x1 - ex <= px <= x2 + ex
        and
        y1 - ey <= py <= y2 + ey
    )


def hand_is_near_sink(hand_box, sink_box):
    hx1, hy1, hx2, hy2 = hand_box
    sx1, sy1, sx2, sy2 = sink_box

    hand_x, hand_y = box_center(hand_box)

    sink_width = max(1, sx2 - sx1)
    sink_height = max(1, sy2 - sy1)

    horizontal_margin = int(
        sink_width * SINK_HORIZONTAL_MARGIN
    )

    allowed_x1 = sx1 - horizontal_margin
    allowed_x2 = sx2 + horizontal_margin

    horizontal_match = (
        allowed_x1 <= hand_x <= allowed_x2
    )

    top_margin = int(
        sink_height * SINK_TOP_MARGIN
    )

    bottom_margin = int(
        sink_height * SINK_BOTTOM_MARGIN
    )

    allowed_y1 = sy1 - top_margin
    allowed_y2 = sy2 + bottom_margin

    vertical_match = (
        allowed_y1 <= hand_y <= allowed_y2
    )

    return horizontal_match and vertical_match


def assign_person_to_sink(person_box, sink_boxes):
    """
    Assign ONE sink to a person.

    We use the person's lower-center point because it is
    generally more useful for locating the workstation/sink
    than the person's head.

    Returns:
        sink_index or None
    """
    if not sink_boxes:
        return None

    person_point = bottom_center(person_box)

    best_index = None
    best_distance = float("inf")

    for sink_index, sink_data in enumerate(sink_boxes):
        sink_box, _confidence = sink_data

        sx1, sy1, sx2, sy2 = sink_box
        sink_center = (
            int((sx1 + sx2) / 2),
            int((sy1 + sy2) / 2),
        )

        distance = point_distance(
            person_point,
            sink_center,
        )

        sink_width = max(1, sx2 - sx1)
        sink_height = max(1, sy2 - sy1)

        max_distance = (
            max(sink_width, sink_height)
            * MAX_PERSON_SINK_DISTANCE
        )

        if distance <= max_distance and distance < best_distance:
            best_distance = distance
            best_index = sink_index

    return best_index


def get_current_step(step_result, step_model):
    """
    Return the highest-confidence step in the frame.

    The step model is a frame-level model in the original
    script. We later associate its box with the nearest person.
    """
    current_step = None
    current_confidence = 0.0
    current_box = None

    if step_result.boxes is None:
        return current_step, current_confidence, current_box

    if len(step_result.boxes) == 0:
        return current_step, current_confidence, current_box

    for box, class_id, confidence in zip(
        step_result.boxes.xyxy,
        step_result.boxes.cls,
        step_result.boxes.conf,
    ):
        confidence = float(confidence)

        if confidence < STEP_CONFIDENCE:
            continue

        class_id = int(class_id)

        step_name = step_model.names.get(
            class_id,
            f"class_{class_id}",
        )

        if confidence > current_confidence:
            current_step = step_name
            current_confidence = confidence
            current_box = tuple(box.int().tolist())

    return current_step, current_confidence, current_box


# ============================================================
# VIDEO-TIME ACTION LOGGING
# ============================================================

def format_video_time(frame_number, fps):
    """Convert video frame number to MM:SS using video time."""
    if fps <= 0:
        fps = 25.0

    total_seconds = int(
        max(0, frame_number - 1) / fps
    )

    minutes = total_seconds // 60
    seconds = total_seconds % 60

    return f"{minutes:02d}:{seconds:02d}"


def write_action_log(
    person_id,
    action,
    frame_number,
    fps,
    extra=""
):
    """Write an event using the VIDEO timeline, not wall-clock time."""

    video_time = format_video_time(
        frame_number,
        fps
    )

    line = (
        f"Person ID: {person_id} | "
        f"Action: {action} | "
        f"Video Time: {video_time}"
    )

    if extra:
        line += f" | {extra}"

    with open(
        LOG_FILE_PATH,
        "a",
        encoding="utf-8"
    ) as log_file:
        log_file.write(line + "\n")

    print(f"\n[LOG] {line}")


def change_action_state(
    person_id,
    state,
    new_state,
    action,
    frame_number,
    fps
):
    """Change state and log only when the state changes."""

    if state["action_state"] == new_state:
        return False

    state["action_state"] = new_state
    state["last_action_frame"] = frame_number

    write_action_log(
        person_id,
        action,
        frame_number,
        fps
    )

    return True


# ============================================================
# CHECK FILES
# ============================================================

for file_path in (
    HAND_MODEL_PATH,
    SINK_MODEL_PATH,
    HAND_STEP_MODEL_PATH,
    VIDEO_PATH,
):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"\nFile not found:\n{file_path}"
        )


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("Loading models...")
print("Device:", "GPU" if DEVICE == 0 else "CPU")

person_model = YOLO(PERSON_MODEL_PATH)
hand_model = YOLO(HAND_MODEL_PATH)
sink_model = YOLO(SINK_MODEL_PATH)
step_model = YOLO(HAND_STEP_MODEL_PATH)

print("Person model:", person_model.names)
print("Hand model  :", hand_model.names)
print("Sink model  :", sink_model.names)
print("Step model  :", step_model.names)
print("=" * 70)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

source_fps = cap.get(cv2.CAP_PROP_FPS)

if source_fps <= 0:
    source_fps = 25.0

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

print()
print(f"Source FPS   : {source_fps:.2f}")
print(f"Video size   : {width} x {height}")
print(f"Frame count  : {total_frames}")
print()


# ============================================================
# OUTPUT VIDEO
#
# Keep the ORIGINAL source FPS so playback speed is not changed.
# ============================================================

out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    source_fps,
    (width, height),
)

if not out.isOpened():
    cap.release()
    raise RuntimeError(
        f"Could not create output video:\n{OUTPUT_PATH}"
    )


# ============================================================
# PERSON STATE
#
# One independent state object per ByteTrack ID.
# ============================================================

person_states = defaultdict(
    lambda: {
        "sink_index": None,

        # Current hand centers belonging to this person.
        "hand_centers": [],

        # Historical hand center for movement calculation.
        "movement_history": deque(
            maxlen=MOVEMENT_HISTORY_SIZE
        ),

        "movement_score": 0.0,

        "hand_near_sink": False,
        "step_detected": False,

        "washing": False,

        # Event state and VIDEO-FRAME timing.
        "action_state": "NOT_NEAR_SINK",
        "last_action_frame": 0,
        "entered_sink_frame": None,
        "washing_start_frame": None,
        "washing_duration_logged": 0,

        "positive_frames": 0,
        "no_detection_frames": 0,

        "event_count": 0,

        "last_seen_frame": 0,
    }
)


# ============================================================
# VARIABLES
# ============================================================

frame_number = 0

# Number of frames corresponding to the reset timeout.
reset_after_no_detection_frames = max(
    1,
    int(
        source_fps *
        RESET_AFTER_NO_DETECTION_SECONDS
    )
)

# Adaptive processing.
#
# At source 30 FPS:
#   no person  -> process approximately every 3rd frame = 10 FPS
#   person     -> process every frame = 30 FPS
#
# At another source FPS, the same idea is scaled.
no_person_stride = max(
    1,
    int(round(source_fps / NO_PERSON_PROCESS_FPS))
)

person_stride = max(
    1,
    int(round(source_fps / PERSON_PROCESS_FPS))
)

active_person_seen = False

last_annotated_frame = None

processing_times = deque(maxlen=30)
processing_fps = 0.0

hand_washing_count = 0

# Start a fresh action log for this video.
# All timestamps written later are VIDEO TIME (MM:SS).
with open(LOG_FILE_PATH, "w", encoding="utf-8") as log_file:
    log_file.write("=" * 80 + "\n")
    log_file.write("HANDWASHING ACTION LOG\n")
    log_file.write("=" * 80 + "\n")
    log_file.write(f"Video: {VIDEO_PATH}\n")
    log_file.write("Time format: VIDEO TIME (MM:SS)\n")
    log_file.write("=" * 80 + "\n\n")


# ============================================================
# PROCESS VIDEO
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    frame_start_time = time.time()

    # ========================================================
    # ADAPTIVE FRAME SKIP
    # ========================================================
    #
    # If no person is currently known:
    #   run detection approximately at 10 FPS.
    #
    # If a person is active:
    #   run detection approximately at 30 FPS.
    #
    # IMPORTANT:
    # We still WRITE every source frame to the output video.
    # ========================================================

    current_stride = (
        person_stride
        if active_person_seen
        else no_person_stride
    )

    should_process = (
        (frame_number - 1) % current_stride == 0
    )

    if not should_process:

        if last_annotated_frame is not None:
            out.write(last_annotated_frame)

        else:
            out.write(frame)

        continue


    # ========================================================
    # PERSON DETECTION + BYTETRACK
    # ========================================================

    person_result = person_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[PERSON_CLASS_ID],
        conf=PERSON_CONFIDENCE,
        imgsz=640,
        device=DEVICE,
        verbose=False,
    )[0]


    # ========================================================
    # COLLECT PERSONS
    # ========================================================

    persons = []

    if (
        person_result.boxes is not None
        and
        len(person_result.boxes) > 0
        and
        person_result.boxes.id is not None
    ):

        person_ids = (
            person_result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        person_boxes = (
            person_result.boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )

        person_confidences = (
            person_result.boxes.conf
            .cpu()
            .tolist()
        )

        for person_id, person_box, confidence in zip(
            person_ids,
            person_boxes,
            person_confidences,
        ):

            person_id = int(person_id)

            person_box = tuple(
                map(int, person_box)
            )

            persons.append(
                {
                    "id": person_id,
                    "box": person_box,
                    "confidence": float(confidence),
                }
            )

    active_person_seen = len(persons) > 0


    # ========================================================
    # MODEL 2 - SINK DETECTION
    # ========================================================

    sink_result = sink_model.predict(
        frame,
        imgsz=640,
        conf=SINK_CONFIDENCE,
        verbose=False,
        device=DEVICE,
    )[0]

    sink_boxes = []

    if sink_result.boxes is not None:

        for box, class_id, confidence in zip(
            sink_result.boxes.xyxy,
            sink_result.boxes.cls,
            sink_result.boxes.conf,
        ):

            detected_class_id = int(class_id)

            if detected_class_id != SINK_CLASS_ID:
                continue

            sink_box = tuple(
                box.int().tolist()
            )

            sink_boxes.append(
                (
                    sink_box,
                    float(confidence),
                )
            )


    # ========================================================
    # ASSIGN EACH PERSON TO ONE SINK
    # ========================================================

    for person in persons:

        person_id = person["id"]
        person_box = person["box"]

        sink_index = assign_person_to_sink(
            person_box,
            sink_boxes,
        )

        person_states[person_id]["sink_index"] = sink_index
        person_states[person_id]["last_seen_frame"] = frame_number


    # ========================================================
    # MODEL 3 - HAND DETECTION
    # ========================================================

    hand_result = hand_model.predict(
        frame,
        imgsz=640,
        conf=HAND_CONFIDENCE,
        verbose=False,
        device=DEVICE,
    )[0]

    hand_boxes = []

    if hand_result.boxes is not None:

        for box, class_id, confidence in zip(
            hand_result.boxes.xyxy,
            hand_result.boxes.cls,
            hand_result.boxes.conf,
        ):

            detected_class_id = int(class_id)

            if detected_class_id != HAND_CLASS_ID:
                continue

            hand_box = tuple(
                box.int().tolist()
            )

            hand_boxes.append(
                (
                    hand_box,
                    float(confidence),
                )
            )


    # ========================================================
    # MODEL 4 - HANDWASHING STEP MODEL
    # ========================================================

    step_result = step_model.predict(
        frame,
        imgsz=640,
        conf=STEP_CONFIDENCE,
        verbose=False,
        device=DEVICE,
    )[0]

    (
        detected_step,
        detected_step_confidence,
        detected_step_box,
    ) = get_current_step(
        step_result,
        step_model,
    )


    # ========================================================
    # RESET CURRENT PER-PERSON FRAME STATE
    # ========================================================

    for person in persons:

        person_id = person["id"]
        state = person_states[person_id]

        state["hand_centers"] = []
        state["hand_near_sink"] = False
        state["step_detected"] = False


    # ========================================================
    # ASSIGN HANDS -> PERSON
    # ========================================================
    #
    # This is the critical change.
    #
    # A hand is NOT simply considered valid because it is
    # near ANY sink.
    #
    # It must first belong to a specific person.
    # ========================================================

    hand_assignments = []

    for hand_box, hand_confidence in hand_boxes:

        hand_center = box_center(hand_box)

        best_person_id = None
        best_distance = float("inf")

        for person in persons:

            person_id = person["id"]
            person_box = person["box"]

            if not point_inside_expanded_box(
                hand_center,
                person_box,
                PERSON_BOX_EXPAND_X,
                PERSON_BOX_EXPAND_Y,
            ):
                continue

            person_center = box_center(person_box)

            distance = point_distance(
                hand_center,
                person_center,
            )

            if distance < best_distance:
                best_distance = distance
                best_person_id = person_id

        hand_assignments.append(
            {
                "box": hand_box,
                "confidence": hand_confidence,
                "center": hand_center,
                "person_id": best_person_id,
            }
        )


    # ========================================================
    # PERSON -> HAND -> PERSON'S SINK
    # ========================================================

    for assignment in hand_assignments:

        person_id = assignment["person_id"]

        if person_id is None:
            continue

        hand_box = assignment["box"]
        hand_center = assignment["center"]

        state = person_states[person_id]

        state["hand_centers"].append(
            hand_center
        )

        sink_index = state["sink_index"]

        if sink_index is not None:

            sink_box, _sink_confidence = (
                sink_boxes[sink_index]
            )

            if hand_is_near_sink(
                hand_box,
                sink_box,
            ):
                state["hand_near_sink"] = True


    # ========================================================
    # HAND MOVEMENT PER PERSON
    # ========================================================

    for person in persons:

        person_id = person["id"]
        state = person_states[person_id]

        centers = state["hand_centers"]

        if centers:

            # Average center of all hands belonging to this person.
            avg_x = sum(
                p[0] for p in centers
            ) / len(centers)

            avg_y = sum(
                p[1] for p in centers
            ) / len(centers)

            current_center = (
                int(avg_x),
                int(avg_y),
            )

            state["movement_history"].append(
                current_center
            )

            history = state["movement_history"]

            if len(history) >= 2:

                movement = point_distance(
                    history[-1],
                    history[-2],
                )

                state["movement_score"] = (
                    0.8 * state["movement_score"]
                    +
                    0.2 * movement
                )

        else:

            state["movement_score"] *= 0.8


    # ========================================================
    # ASSIGN STEP DETECTION -> NEAREST PERSON
    # ========================================================

    if (
        USE_STEP_MODEL_AS_SUPPORT
        and
        detected_step_box is not None
        and
        persons
    ):

        step_center = box_center(
            detected_step_box
        )

        best_person_id = None
        best_distance = float("inf")

        for person in persons:

            person_id = person["id"]
            person_box = person["box"]

            person_center = box_center(
                person_box
            )

            distance = point_distance(
                step_center,
                person_center,
            )

            if distance < best_distance:
                best_distance = distance
                best_person_id = person_id

        if best_person_id is not None:

            person_states[
                best_person_id
            ]["step_detected"] = True


    # ========================================================
    # FINAL DECISION FOR EACH PERSON
    # ========================================================

    for person in persons:

        person_id = person["id"]
        state = person_states[person_id]

        has_hand = (
            len(state["hand_centers"])
            >= MIN_HANDS_FOR_WASHING
        )

        near_sink = state["hand_near_sink"]

        moving = (
            state["movement_score"]
            >= MIN_HAND_MOVEMENT
        )

        # ----------------------------------------------------
        # MAIN WASHING CONDITION
        # ----------------------------------------------------

        if REQUIRE_HAND_MOVEMENT:

            washing_evidence = (
                has_hand
                and
                near_sink
                and
                moving
            )

        else:

            washing_evidence = (
                has_hand
                and
                near_sink
            )

        # ----------------------------------------------------
        # TEMPORAL CONFIRMATION
        # ----------------------------------------------------

        if washing_evidence:

            state["positive_frames"] += 1
            state["no_detection_frames"] = 0

        else:

            state["positive_frames"] = 0
            state["no_detection_frames"] += 1

        # ----------------------------------------------------
        # PERSON IS NEAR THE SINK
        # ----------------------------------------------------

        if near_sink:

            if state["entered_sink_frame"] is None:

                state["entered_sink_frame"] = frame_number

                change_action_state(
                    person_id,
                    state,
                    "NEAR_SINK_NOT_WASHING",
                    "ENTERED SINK AREA",
                    frame_number,
                    source_fps
                )

            # -----------------------------------------------
            # CONFIRM WASHING
            # -----------------------------------------------

            if (
                not state["washing"]
                and
                state["positive_frames"]
                >= DETECTION_CONFIRMATION_FRAMES
            ):

                state["washing"] = True
                state["event_count"] += 1
                hand_washing_count += 1

                # Store VIDEO FRAME when washing starts.
                state["washing_start_frame"] = frame_number
                state["washing_duration_logged"] = 0

                change_action_state(
                    person_id,
                    state,
                    "WASHING",
                    "WASHING STARTED",
                    frame_number,
                    source_fps
                )

            elif not state["washing"]:

                change_action_state(
                    person_id,
                    state,
                    "NEAR_SINK_NOT_WASHING",
                    "NEAR TO SINK - NOT WASHING",
                    frame_number,
                    source_fps
                )

        # ----------------------------------------------------
        # PERSON HAS LEFT THE SINK AREA
        # ----------------------------------------------------

        if (
            not near_sink
            and
            state["entered_sink_frame"] is not None
            and
            state["no_detection_frames"]
            >= reset_after_no_detection_frames
        ):

            if state["washing"]:

                washing_duration = 0.0

                if state["washing_start_frame"] is not None:

                    washing_duration = (
                        frame_number
                        -
                        state["washing_start_frame"]
                    ) / max(source_fps, 1.0)

                if washing_duration >= MIN_WASHING_DURATION:

                    write_action_log(
                        person_id,
                        "WASHED",
                        frame_number,
                        source_fps,
                        f"Duration: {washing_duration:.2f} seconds"
                    )

                    state["action_state"] = "WASHED"

                else:

                    write_action_log(
                        person_id,
                        "NOT WASHED",
                        frame_number,
                        source_fps,
                        f"Duration: {washing_duration:.2f} seconds"
                    )

                    state["action_state"] = "NOT_WASHED"

                state["washing"] = False
                state["washing_start_frame"] = None
                state["washing_duration_logged"] = 0

            else:

                write_action_log(
                    person_id,
                    "NOT WASHED",
                    frame_number,
                    source_fps
                )

                state["action_state"] = "NOT_WASHED"

            write_action_log(
                person_id,
                "LEFT SINK AREA",
                frame_number,
                source_fps
            )

            state["entered_sink_frame"] = None
            state["positive_frames"] = 0
            state["no_detection_frames"] = 0


    # ========================================================
    # WASHING DURATION MILESTONES
    # ========================================================

    for person in persons:

        person_id = person["id"]
        state = person_states[person_id]

        if (
            state["washing"]
            and
            state["washing_start_frame"] is not None
        ):

            duration = (
                frame_number
                -
                state["washing_start_frame"]
            ) / max(source_fps, 1.0)

            milestone = int(duration // 5) * 5

            if (
                milestone >= 5
                and
                milestone >
                state["washing_duration_logged"]
            ):

                write_action_log(
                    person_id,
                    f"WASHING - {milestone} SECONDS",
                    frame_number,
                    source_fps,
                    f"Current duration: {duration:.2f} seconds"
                )

                state["washing_duration_logged"] = milestone



    # ========================================================
    # REMOVE OLD PERSON STATES
    # ========================================================

    old_ids = []

    for person_id, state in person_states.items():

        if (
            frame_number
            -
            state["last_seen_frame"]
            >
            int(source_fps * 3)
        ):
            old_ids.append(person_id)

    for person_id in old_ids:
        del person_states[person_id]


    # ========================================================
    # CREATE ANNOTATED FRAME
    # ========================================================

    annotated_frame = frame.copy()


    # ========================================================
    # DRAW SINKS
    # ========================================================

    for sink_index, (
        sink_box,
        confidence,
    ) in enumerate(sink_boxes):

        x1, y1, x2, y2 = sink_box

        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            SINK_COLOR,
            2,
        )

        sink_center = box_center(
            sink_box
        )

        cv2.circle(
            annotated_frame,
            sink_center,
            5,
            YELLOW,
            -1,
        )

        draw_label(
            annotated_frame,
            f"SINK {sink_index + 1}",
            (
                x1,
                max(25, y1 - 8),
            ),
            SINK_COLOR,
            0.55,
        )


    # ========================================================
    # DRAW PERSONS
    # ========================================================

    for person in persons:

        person_id = person["id"]
        person_box = person["box"]

        x1, y1, x2, y2 = person_box

        state = person_states[person_id]

        if state["washing"]:
            person_color = WASHING_COLOR
            status_text = "WASHING"
        elif state["action_state"] == "NEAR_SINK_NOT_WASHING":
            person_color = NOT_WASHING_COLOR
            status_text = "NEAR SINK - NOT WASHING"
        elif state["action_state"] == "WASHED":
            person_color = WASHING_COLOR
            status_text = "WASHED"
        elif state["action_state"] == "NOT_WASHED":
            person_color = NOT_WASHING_COLOR
            status_text = "NOT WASHED"
        else:
            person_color = NOT_WASHING_COLOR
            status_text = "NOT NEAR SINK"


        # Person box
        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            person_color,
            2,
        )


        # Person ID
        draw_label(
            annotated_frame,
            (
                f"ID {person_id} | "
                f"{status_text}"
            ),
            (
                x1,
                max(25, y1 - 10),
            ),
            person_color,
            0.65,
        )


        # ----------------------------------------------------
        # Assigned sink
        # ----------------------------------------------------

        sink_index = state["sink_index"]

        if sink_index is not None:

             draw_label(
                 annotated_frame,
                 f"SINK {sink_index + 1}",
                 (
                     x1,
                     min(
                         height - 10,
                         y2 + 22,
                     ),
                 ),
                 PERSON_COLOR,
                 0.50,
             )

        # ----------------------------------------------------
        # Person state information
        # ----------------------------------------------------

        washing_duration = 0.0

        if (
            state["washing"]
            and
            state["washing_start_frame"] is not None
        ):
            washing_duration = (
                frame_number
                -
                state["washing_start_frame"]
            ) / max(source_fps, 1.0)

        state_text = (
            f"Hand:{len(state['hand_centers'])} "
            f"NearSink:{'YES' if state['hand_near_sink'] else 'NO'} "
            f"Move:{state['movement_score']:.1f} "
            f"Action:{state['action_state']} "
            f"WashTime:{washing_duration:.1f}s"
        )

        draw_label(
            annotated_frame,
            state_text,
            (
                x1,
                min(
                    height - 10,
                    y2 + 43,
                ),
            ),
            WHITE,
            0.45,
        )


    # ========================================================
    # DRAW HANDS
    # ========================================================

    for assignment in hand_assignments:

        hand_box = assignment["box"]
        confidence = assignment["confidence"]
        person_id = assignment["person_id"]

        x1, y1, x2, y2 = hand_box

        is_associated = person_id is not None

        is_near_person_sink = False

        if is_associated:

            state = person_states[person_id]

            sink_index = state["sink_index"]

            if sink_index is not None:

                sink_box, _sink_confidence = (
                    sink_boxes[sink_index]
                )

                is_near_person_sink = (
                    hand_is_near_sink(
                        hand_box,
                        sink_box,
                    )
                )


        if (
            is_associated
            and
            is_near_person_sink
        ):

            hand_color = VALID_HAND_COLOR

            label = (
                f"HAND -> ID {person_id}"
            )

        elif is_associated:

            hand_color = INVALID_HAND_COLOR

            label = (
                f"HAND -> ID {person_id}"
            )

        else:

            hand_color = INVALID_HAND_COLOR

            label = "HAND -> NO PERSON"


        cv2.rectangle(
            annotated_frame,
            (x1, y1),
            (x2, y2),
            hand_color,
            2,
        )

        draw_label(
            annotated_frame,
            (
                f"{label} "
                f"{confidence:.2f}"
            ),
            (
                x1,
                max(25, y1 - 8),
            ),
            hand_color,
            0.48,
        )


    # ========================================================
    # DRAW STEP
    # ========================================================

    if (
        detected_step is not None
        and
        detected_step_box is not None
    ):

        sx1, sy1, sx2, sy2 = (
            detected_step_box
        )

        cv2.rectangle(
            annotated_frame,
            (sx1, sy1),
            (sx2, sy2),
            STEP_COLOR,
            2,
        )

        draw_label(
            annotated_frame,
            (
                f"{detected_step} "
                f"{detected_step_confidence:.2f}"
            ),
            (
                sx1,
                max(25, sy1 - 8),
            ),
            STEP_COLOR,
            0.55,
        )


    # ========================================================
    # GLOBAL STATUS
    # ========================================================

    washing_persons = [
        person["id"]
        for person in persons
        if person_states[
            person["id"]
        ]["washing"]
    ]

    if washing_persons:

        global_status = (
            "HANDWASHING: YES | "
            f"IDs: {washing_persons}"
        )

        global_color = WASHING_COLOR

    else:

        global_status = "HANDWASHING: NO"
        global_color = NOT_WASHING_COLOR


    draw_label(
        annotated_frame,
        global_status,
        (20, 35),
        global_color,
        0.75,
    )


    # ========================================================
    # INFO PANEL
    # ========================================================

    processing_elapsed = (
        time.time() - frame_start_time
    )

    if processing_elapsed > 0:

        processing_times.append(
            processing_elapsed
        )

        processing_fps = (
            1.0 /
            (
                sum(processing_times)
                /
                len(processing_times)
            )
        )


    target_fps = (
        PERSON_PROCESS_FPS
        if active_person_seen
        else NO_PERSON_PROCESS_FPS
    )

    draw_label(
        annotated_frame,
        (
            f"Persons: {len(persons)} | "
            f"Sinks: {len(sink_boxes)} | "
            f"Hands: {len(hand_boxes)}"
        ),
        (20, 65),
        WHITE,
        0.55,
    )

    draw_label(
        annotated_frame,
        (
            f"Target inference FPS: "
            f"{target_fps:.0f} | "
            f"Actual: {processing_fps:.1f}"
        ),
        (20, 92),
        YELLOW,
        0.55,
    )

    draw_label(
        annotated_frame,
        (
            f"Total washing events: "
            f"{hand_washing_count}"
        ),
        (20, 119),
        WHITE,
        0.55,
    )


    # ========================================================
    # SAVE
    # ========================================================

    last_annotated_frame = annotated_frame.copy()

    out.write(
        annotated_frame
    )


    # ========================================================
    # CONSOLE DEBUG
    # ========================================================

    person_debug = []

    for person in persons:

        person_id = person["id"]
        state = person_states[person_id]

        sink_text = (
            str(state["sink_index"] + 1)
            if state["sink_index"] is not None
            else "NONE"
        )

        person_debug.append(
            (
                f"ID={person_id} "
                f"SINK={sink_text} "
                f"HAND={len(state['hand_centers'])} "
                f"NEAR={state['hand_near_sink']} "
                f"MOVE={state['movement_score']:.1f} "
                f"WASH={state['washing']}"
            )
        )

    print(
        f"\rFrame {frame_number}/{total_frames} | "
        + " || ".join(person_debug),
        end="",
    )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

print()
print()
print("=" * 70)
print("VIDEO PROCESSING COMPLETE")
print("=" * 70)
print("Frames processed       :", frame_number)
print("Source FPS             :", f"{source_fps:.2f}")
print("No-person target FPS   :", NO_PERSON_PROCESS_FPS)
print("Person-present target  :", PERSON_PROCESS_FPS)
print("Total washing events   :", hand_washing_count)
print("Output video            :", OUTPUT_PATH)
print("Action log              :", LOG_FILE_PATH)
print("Log time format         : VIDEO TIME (MM:SS)")
print("=" * 70)
