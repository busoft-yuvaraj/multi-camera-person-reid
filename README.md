# Formal Attire Detection

Real-time formal-attire detection and tracking, built on a custom-trained
YOLO model (`best.pt`) with **ByteTrack** for multi-object tracking, a live
OpenCV window, and automatic saving of the annotated output video.

## Project structure

```
formal_attire_detection/
├── config/
│   ├── config.yaml       # app-level settings (model, video, output, display)
│   └── bytetrack.yaml    # ByteTrack tracker parameters
├── models/
│   └── best.pt            # <- put your trained weights here
├── outputs/                # annotated video (and optional frames) saved here
├── logs/                   # run logs
├── src/
│   ├── detector.py         # YOLO + ByteTrack wrapper
│   ├── video_stream.py     # camera/file/RTSP reader + video writer
│   ├── utils.py             # config loading + logging setup
│   └── main.py               # pipeline orchestration
├── run.py                   # entry point
└── requirements.txt
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Copy your trained model into `models/best.pt` (or edit
`config/config.yaml -> model.weights_path` to point elsewhere).

## Run

```bash
# Default webcam, live window + saved video in outputs/
python run.py

# A specific video file
python run.py --source path/to/video.mp4

# An RTSP / network camera stream
python run.py --source rtsp://user:pass@camera-ip/stream

# Headless (no live window) — e.g. running on a server
python run.py --no-display
```

Press **q** in the live window to stop early. Output is written to
`outputs/formal_attire_detection.mp4` (filename/dir configurable in
`config/config.yaml`).

## How it works

1. `VideoStream` opens the configured source (webcam index, file path, or
   RTSP URL) and reports the real frame size/fps.
2. `FormalAttireDetector` loads `best.pt` and, per frame, calls
   ultralytics' `model.track(..., tracker="config/bytetrack.yaml")`, which
   runs detection and feeds the boxes into ByteTrack so each detected
   person/item keeps a stable ID across frames.
3. `OutputWriter` streams every annotated frame to
   `outputs/formal_attire_detection.mp4` via `cv2.VideoWriter`.
4. If `display.show_live` is true, each annotated frame is also shown live
   in a `cv2.imshow` window with an FPS overlay.

## Configuration highlights (`config/config.yaml`)

| Key | Purpose |
|---|---|
| `model.weights_path` | Path to `best.pt` |
| `model.confidence_threshold` / `iou_threshold` | Detection filtering |
| `model.device` | `cpu`, `cuda:0`, etc. |
| `tracker.config_path` | Points to `config/bytetrack.yaml` |
| `video.source` | `0` for webcam, or a file path / RTSP URL |
| `output.save_dir` / `video_filename` | Where the annotated video is saved |
| `display.show_live` | Toggle the live `cv2.imshow` window |

## Extending

- Swap in your own class-specific logic (e.g. flag "non-formal" detections)
  inside `FormalAttireDetector` / `src/main.py`'s loop, using
  `result.boxes.cls`, `.conf`, and `.id` from the tracked result.
- For multiple camera feeds, run multiple instances of `run.py` with
  different `--source` values and distinct `output.video_filename`s.
