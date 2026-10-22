import json

print("=== TRACKS IN PANTRY (Frame 1675 - 1710) ===")
# Let's inspect logs/run_20261016_162825.log for pantry track detections
with open("logs/run_20261016_162825.log", "r") as f:
    for line in f:
        if "pantry" in line and any(f"frm{i}" in line or f"frame {i}" in line or f"Frame {i}" in line or f"frame_idx {i}" in line for i in range(1680, 1700)):
            print(line.strip())
