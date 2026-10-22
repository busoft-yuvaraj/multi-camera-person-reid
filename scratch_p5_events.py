import json

print("=== ALL JOURNEY EVENTS FOR Person_005 ===")
with open("logs/journey.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        if d.get("global_id") == "Person_005":
            print(f"Frm {d.get('frame_idx')} | Evt: {d.get('event')} | Cam: {d.get('camera_id')} | Trk: {d.get('local_track_id')} | Time: {d.get('timestamp')} | Dets: {d.get('details')}")
