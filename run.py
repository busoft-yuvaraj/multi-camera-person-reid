#!/usr/bin/env python
"""
Project entry point.

Usage:
    python run.py
    python run.py --config config/config.yaml
    python run.py --source 0                 # webcam
    python run.py --source path/to/video.mp4  # video file
    python run.py --source rtsp://...         # network stream
    python run.py --no-display                # headless run (server/CI)
"""

from src.main import main

if __name__ == "__main__":
    main()
