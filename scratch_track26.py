import json

print("=== Check Track 26 in pantry between 2210 and 2305 ===")
# Let's inspect where Track 26 was seen
# In association_decisions or let's inspect the video / tracking
with open("logs/association_decisions.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        f_idx = d.get("frame_idx", 0)
        if 2210 <= f_idx <= 2310 and d.get("local_track_id") == 26:
            print(f"Frm {f_idx} | Cam: {d.get('camera_id')} | Trk 26 | Dec: {d.get('decision')}")
