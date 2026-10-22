import json

print("=== Person_005 in association decisions around 2290-2310 ===")
with open("logs/association_decisions.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        f_idx = d.get("frame_idx", 0)
        if 2290 <= f_idx <= 2310:
            if "Person_005" in str(d):
                print(f"Frame {f_idx} | Cam: {d.get('camera_id')} | Trk: {d.get('local_track_id')} | Rejects: {d.get('rejected_candidates')}")
