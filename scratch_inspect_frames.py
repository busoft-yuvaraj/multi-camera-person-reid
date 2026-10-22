import json

with open("logs/association_decisions.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        d = json.loads(line)
        f_idx = d.get("frame_idx", 0)
        cid = d.get("camera_id")
        tid = d.get("local_track_id")
        if 825 <= f_idx <= 880:
            print(f"[f{f_idx:4d} | {d.get('timestamp', 0):6.2f}s] {cid:13s} trk{tid:2d}: "
                  f"decision={d.get('decision')} "
                  f"sel_gid={d.get('selected_global_id')} "
                  f"top_cand={d.get('top_candidate')} "
                  f"top_score={d.get('top_score')} "
                  f"conflict_rej={d.get('conflict_rejections')} "
                  f"scores={d.get('qdrant_scores')}")
