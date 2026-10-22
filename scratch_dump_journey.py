import json

print("=== JOURNEY EVENTS (Frame 2150 - 2330) ===")
with open("logs/journey.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        f_idx = d.get("frame_idx", 0)
        if 2150 <= f_idx <= 2330:
            print(f"Frame {f_idx:4d} | Evt: {d.get('event'):22s} | GID: {d.get('global_id')} | Cam: {d.get('camera_id'):12s} | Trk: {d.get('local_track_id')} | Time: {d.get('timestamp'):.2f} | Dets: {d.get('details')}")
