import yaml
import logging
import os
from datetime import datetime
from app.reid.extractor import OSNetExtractor
from app.reid.extractors.par_extractor import ParExtractor
from app.reid.extractors.pose_extractor import PoseExtractor
from app.reid.extractors.attire_extractor import AttireExtractor
from app.reid.global_gallery import GlobalGallery
from app.reid.qdrant_gallery import QdrantGallery
from app.reid.matcher import GlobalMatcher
from app.multi_camera_processor import MultiCameraProcessor

def setup_logging(config):
    os.makedirs("logs", exist_ok=True)
    log_filename = datetime.now().strftime("logs/run_%Y%m%d_%H%M%S.log")
    
    level = logging.DEBUG if config.get("enable_debug_logging") else logging.INFO
    
    # Remove any existing handlers (Ultralytics sets up its own, which causes double-printing)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized. Writing to {log_filename}")

def main():
    with open("config/app_config.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    setup_logging(config)
        
    logging.info("Loading OSNet Feature Extractor...")
    extractor = OSNetExtractor(model_name=config["reid_model_path"], device="cpu")
    
    logging.info("Loading PAR Extractor...")
    par_extractor = ParExtractor(model_path=config.get("par_model_path", "models/mobilenet_par.pt"), device="cpu")
    
    logging.info("Loading Pose Extractor...")
    pose_extractor = PoseExtractor(model_path=config.get("pose_model_path", "models/yolo11n-pose.pt"), device="cpu")
    
    attire_extractor = None
    if config.get("attire_detection_enabled", True):
        logging.info("Loading Formal Attire Extractor...")
        attire_extractor = AttireExtractor(
            model_path=config.get("attire_model_path", "models/formal_attire_best.pt"),
            conf=config.get("attire_conf_threshold", 0.25),
            device="cpu"
        )
    
    gallery = None
    if config.get("qdrant_url"):
        try:
            logging.info("Initializing Qdrant Gallery...")
            gallery = QdrantGallery(
                url=config["qdrant_url"],
                api_key=config.get("qdrant_api_key"),
                collection_name=config.get("qdrant_collection", "cctv-poc"),
                clear_on_start=config.get("clear_gallery_on_start", True),
                max_embeddings=config.get("reid_bank_size", 15)
            )
        except Exception as e:
            logging.warning(f"Could not connect to Qdrant ({e}). Falling back to local GlobalGallery.")
            gallery = None

    if gallery is None:
        logging.info("Initializing Global Gallery (Local Pickle)...")
        gallery = GlobalGallery(
            storage_path=config.get("global_gallery_path", "global_gallery.pkl"),
            clear_on_start=config.get("clear_gallery_on_start", True)
        )
    matcher = GlobalMatcher(
        gallery=gallery,
        config=config
    )
    
    logging.info("Starting Multi-Camera processing...")
    v_cfg = config.get("violation_detection", {})
    if v_cfg.get("enabled", False) and "cameras" in v_cfg:
        cameras_info = []
        for cam_id, cam_item in v_cfg["cameras"].items():
            v_path = cam_item.get("video_path")
            if not v_path:
                v_path = config.get(f"{cam_id}_video", "")
            cameras_info.append({"camera_id": cam_id, "video_path": v_path})
    else:
        cameras_info = [
            {"camera_id": "cctv1", "video_path": config["cctv1_video"]},
            {"camera_id": "cctv2", "video_path": config["cctv2_video"]}
        ]
    
    processor = MultiCameraProcessor(
        cameras_info=cameras_info,
        output_path=config.get("combined_output", "output/combined_output.mp4"),
        config=config,
        extractor=extractor,
        par_extractor=par_extractor,
        pose_extractor=pose_extractor,
        attire_extractor=attire_extractor,
        gallery=gallery,
        matcher=matcher
    )
    
    try:
        processor.run()
    except KeyboardInterrupt:
        logging.info("Interrupt received. Stopping processing...")
        processor.stop()
        
    logging.info("Processing complete.")
    if hasattr(gallery, "save"):
        gallery.save()

if __name__ == "__main__":
    main()
