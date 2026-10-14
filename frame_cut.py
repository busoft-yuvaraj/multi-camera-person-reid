import os
import cv2
import argparse
from ultralytics import YOLO


# =========================
# CONFIG
# =========================

INPUT_PATH = "cctv_samples/frame"  # Can be a folder of videos or a single video file
OUTPUT_DIR = "reid_crops"

MODEL_PATH = "yolo11n.pt"
TRACKER_CONFIG = "botsort.yaml"

# Supported video extensions
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")

# Save approximately 1 crop per second per track
SAVE_INTERVAL_SECONDS = 1.0

# YOLO settings
CONFIDENCE = 0.4
IMGSZ = 960

# COCO class ID for person
PERSON_CLASS_ID = 0

# Min detection size to ignore false positives / tiny boxes
MIN_WIDTH = 40
MIN_HEIGHT = 80


def reset_tracker(model):
    """
    Reset tracker state in Ultralytics YOLO to ensure clean tracking across different videos.
    """
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers") and model.predictor.trackers:
            for trk in model.predictor.trackers:
                if hasattr(trk, "reset"):
                    trk.reset()
                if hasattr(trk, "reset_id"):
                    trk.reset_id()


def process_video(
    video_path: str,
    output_dir: str,
    model: YOLO,
    tracker_config: str = TRACKER_CONFIG,
    save_interval_seconds: float = SAVE_INTERVAL_SECONDS,
    confidence: float = CONFIDENCE,
    imgsz: int = IMGSZ,
    person_class_id: int = PERSON_CLASS_ID,
    min_width: int = MIN_WIDTH,
    min_height: int = MIN_HEIGHT,
):
    """
    Process a single video file, track people, and save crops grouped by track ID.
    Output structure: output_dir/<video_name>/track_<track_id>/frame_XXXXXX_t_YY.YY.jpg
    """
    video_name = os.path.basename(video_path)
    video_stem = os.path.splitext(video_name)[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"[WARN] Invalid FPS ({fps}) detected for {video_name}, defaulting to 30.0")
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if total_frames > 0 else 0

    print("-" * 50, flush=True)
    print(f"Video    : {video_name}", flush=True)
    print(f"FPS      : {fps:.2f}", flush=True)
    print(f"Frames   : {total_frames}", flush=True)
    print(f"Duration : {duration:.2f} seconds", flush=True)
    print("-" * 50, flush=True)

    # Reset tracker state between videos
    reset_tracker(model)

    # Stores the last interval index at which each track was saved
    last_saved_interval = {}
    saved_crops_count = 0
    frame_number = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_number += 1
        current_time = frame_number / fps
        current_interval = int(current_time / save_interval_seconds)

        # ---------------------------------
        # YOLO + BoT-SORT tracking
        # ---------------------------------
        results = model.track(
            frame,
            persist=True,
            tracker=tracker_config,
            classes=[person_class_id],
            conf=confidence,
            imgsz=imgsz,
            verbose=False,
        )

        result = results[0]

        # No detections or tracks
        if result.boxes is None or result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()
        track_ids = result.boxes.id.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().numpy()

        # ---------------------------------
        # Process every tracked person
        # ---------------------------------
        for box, track_id, conf in zip(boxes, track_ids, confidences):
            track_id = int(track_id)
            x1, y1, x2, y2 = map(int, box)

            # Clamp coordinates to frame boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            width = x2 - x1
            height = y2 - y1

            # Ignore extremely small detections
            if width < min_width or height < min_height:
                continue

            # ---------------------------------
            # Save only once per interval for each track ID
            # ---------------------------------
            previous_interval = last_saved_interval.get(track_id, -1)
            if current_interval <= previous_interval:
                continue

            # ---------------------------------
            # Crop person
            # ---------------------------------
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue

            # ---------------------------------
            # Create track directory: reid_crops/<video_name>/track_<track_id>
            # ---------------------------------
            track_dir = os.path.join(output_dir, video_stem, f"track_{track_id}")
            os.makedirs(track_dir, exist_ok=True)

            # ---------------------------------
            # Save image
            # ---------------------------------
            filename = f"frame_{frame_number:06d}_t_{current_time:.2f}.jpg"
            output_path = os.path.join(track_dir, filename)

            cv2.imwrite(output_path, person_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            last_saved_interval[track_id] = current_interval
            saved_crops_count += 1

            print(
                f"Saved | "
                f"Video={video_stem} | "
                f"Track={track_id:3d} | "
                f"Time={current_time:7.2f}s | "
                f"Conf={conf:.2f} | "
                f"{output_path}",
                flush=True,
            )

    cap.release()
    print(f"--> Finished '{video_name}': {saved_crops_count} crops saved across {len(last_saved_interval)} tracks.", flush=True)
    return saved_crops_count, len(last_saved_interval)


def main():
    parser = argparse.ArgumentParser(description="Extract person ReID crops from videos grouped by tracks.")
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_PATH,
        help=f"Path to a video file or folder of videos (default: {INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_DIR,
        help=f"Directory to save crops (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_PATH,
        help=f"YOLO model path (default: {MODEL_PATH})",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default=TRACKER_CONFIG,
        help=f"Tracker config (default: {TRACKER_CONFIG})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=SAVE_INTERVAL_SECONDS,
        help=f"Interval in seconds between saving crops for each track (default: {SAVE_INTERVAL_SECONDS})",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
        help=f"Confidence threshold (default: {CONFIDENCE})",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMGSZ,
        help=f"Inference image size (default: {IMGSZ})",
    )

    args = parser.parse_args()

    # Collect video files to process
    if os.path.isfile(args.input):
        video_files = [args.input]
    elif os.path.isdir(args.input):
        video_files = [
            os.path.join(args.input, f)
            for f in sorted(os.listdir(args.input))
            if f.lower().endswith(VIDEO_EXTENSIONS)
        ]
    else:
        raise FileNotFoundError(f"Input path '{args.input}' does not exist.")

    if not video_files:
        print(f"[WARN] No video files found in '{args.input}' matching {VIDEO_EXTENSIONS}")
        return

    # Ensure output directory exists
    os.makedirs(args.output, exist_ok=True)

    print("==================================================", flush=True)
    print("STARTING BATCH PERSON CROP EXTRACTION", flush=True)
    print(f"Input path       : {args.input}", flush=True)
    print(f"Output directory : {args.output}", flush=True)
    print(f"Total videos     : {len(video_files)}", flush=True)
    print(f"Model            : {args.model}", flush=True)
    print(f"Tracker          : {args.tracker}", flush=True)
    print(f"Save interval    : {args.interval}s per track", flush=True)
    print("==================================================", flush=True)

    # Load YOLO model once
    model = YOLO(args.model)

    total_crops = 0
    total_tracks = 0

    for idx, video_path in enumerate(video_files, 1):
        print(f"\n[{idx}/{len(video_files)}] Processing: {os.path.basename(video_path)}", flush=True)
        crops_count, tracks_count = process_video(
            video_path=video_path,
            output_dir=args.output,
            model=model,
            tracker_config=args.tracker,
            save_interval_seconds=args.interval,
            confidence=args.conf,
            imgsz=args.imgsz,
            person_class_id=PERSON_CLASS_ID,
            min_width=MIN_WIDTH,
            min_height=MIN_HEIGHT,
        )
        total_crops += crops_count
        total_tracks += tracks_count

    print("\n" + "=" * 50, flush=True)
    print("EXTRACTION COMPLETED", flush=True)
    print("=" * 50, flush=True)
    print(f"Total videos processed : {len(video_files)}", flush=True)
    print(f"Total tracks found     : {total_tracks}", flush=True)
    print(f"Total crops saved      : {total_crops}", flush=True)
    print(f"Output directory       : {os.path.abspath(args.output)}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()