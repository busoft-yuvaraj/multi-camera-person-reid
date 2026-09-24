import yaml
from boxmot import BoTSORT

class TrackerManager:
    def __init__(self, config_path, frame_rate=30):
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
            
        self.tracker = BoTSORT(
            model_weights=None, # Not using internal ReID
            device='cpu',
            fp16=False,
            track_high_thresh=cfg.get("track_high_thresh", 0.4),
            track_low_thresh=cfg.get("track_low_thresh", 0.1),
            new_track_thresh=cfg.get("new_track_thresh", 0.55),
            track_buffer=cfg.get("track_buffer", 200),
            match_thresh=cfg.get("match_thresh", 0.8),
            cmc_method=cfg.get("gmc_method", "none"),
            frame_rate=frame_rate
        )
        
    def update(self, dets, frame):
        # dets: array of [x1, y1, x2, y2, conf, cls]
        # boxmot expects (N, 6) tensor or array
        return self.tracker.update(dets, frame)
