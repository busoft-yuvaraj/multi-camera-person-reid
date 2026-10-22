import json

with open("logs/association_decisions.jsonl", "r") as f:
    for line in f:
        d = json.loads(line)
        f_idx = d.get("frame_idx", 0)
        if (1680 <= f_idx <= 1745) or (2290 <= f_idx <= 2340):
            print(f"Frame {f_idx:4d} | Cam {d.get('camera_id'):12s} | Trk {d.get('local_track_id')} | Dec: {d.get('decision')} | Sel: {d.get('selected_global_id')} | Top: {d.get('top_candidate')} ({d.get('top_score')}) | 2nd: {d.get('second_score')} | Mgn: {d.get('margin')} | Valids: {d.get('valid_candidates')} | Rejects: {d.get('rejected_candidates')}")
