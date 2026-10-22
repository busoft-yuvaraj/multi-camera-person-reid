import json

def analyze_journey(path):
    print("=== Analyzing", path, "===")
    with open(path, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f if line.strip()]

    print(f"Total events: {len(events)}")
    gids = set()
    cam_tracks = {}
    for e in events:
        gid = e.get("global_id")
        cam = e.get("camera_id")
        tid = e.get("local_track_id")
        evt = e.get("event")
        ts = e.get("timestamp")
        gids.add(gid)
        if gid in ["Person_001", "Person_004", "Person_002", "Person_003"]:
            # Print significant events
            if evt in ["TRACK_STARTED", "GLOBAL_ID_CONFIRMED", "GLOBAL_ID_REACQUIRED", "GLOBAL_ID_CONFLICT", "TRANSITION_CROSSED"]:
                reason = e.get("reason", e.get("details", {}).get("reason", ""))
                print(f"[{ts:6.2f}s | f{e.get('frame_idx', 0):4d}] {evt:22s} | {cam:13s} trk{tid:2d} -> {gid} | {reason}")

if __name__ == "__main__":
    analyze_journey("logs/journey.jsonl")
