import yaml
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

from app.reid.extractor import OSNetExtractor
from app.reid.extractors.par_extractor import ParExtractor
from app.reid.extractors.pose_extractor import PoseExtractor
from app.reid.extractors.attire_extractor import AttireExtractor
from app.reid.global_gallery import GlobalGallery
from app.reid.qdrant_gallery import QdrantGallery
from app.reid.matcher import GlobalMatcher
from app.multi_camera_processor import MultiCameraProcessor

# Phase 1 Floor-Plan-Aware Components
from app.journey.journey_store import JourneyStore
from app.journey.journey_manager import JourneyManager
from app.association.topology import TopologyGate
from app.association.temporal_gate import TemporalGate
from app.association.spatial_gate import SpatialGate
from app.association.candidate_filter import CandidateFilter
from app.association.association_manager import AssociationManager

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
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Camera Person Tracker (Phase 1)")
    parser.add_argument("--config", type=str, default="config/app_config.yaml", help="Path to config file")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit frames to process")
    args = parser.parse_args()

    load_dotenv()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.max_frames is not None:
        config["max_frames"] = args.max_frames
        
    # Read Qdrant credentials securely from environment variables
    qdrant_url = os.getenv("QDRANT_URL") or config.get("qdrant_url")
    qdrant_api_key = os.getenv("QDRANT_API_KEY") or config.get("qdrant_api_key")
    qdrant_collection = os.getenv("QDRANT_COLLECTION") or config.get("qdrant_collection", "cctv-poc")
    if qdrant_url:
        config["qdrant_url"] = qdrant_url
    if qdrant_api_key:
        config["qdrant_api_key"] = qdrant_api_key
    if qdrant_collection:
        config["qdrant_collection"] = qdrant_collection

    setup_logging(config)
        
    logging.info("Loading OSNet Feature Extractor...")
    extractor = OSNetExtractor(model_name=config["reid_model_path"], device="cpu")
    
    logging.info("Loading PAR Extractor...")
    par_extractor = ParExtractor(model_path=config.get("par_model_path", "models/mobilenet_par.pt"), device="cpu")
    
    logging.info("Loading Pose Extractor...")
    pose_extractor = PoseExtractor(model_path=config.get("pose_model_path", "models/yolo11n-pose.pt"), device="cpu")
    
    attire_extractor = None
    if config.get("attire_detection_enabled", False):
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
    
    # -----------------------------------------------------------------
    # Phase 1 Pipeline Components Setup
    # -----------------------------------------------------------------
    logging.info("Initializing Phase 1 Journey and Association Managers...")
    store = JourneyStore(
        journey_log_path=config.get("journey_log_path", "logs/journey.jsonl"),
        association_log_path=config.get("association_log_path", "logs/association_decisions.jsonl")
    )
    journey_manager = JourneyManager(
        store=store,
        reacquisition_timeout_seconds=config.get("temporal_constraints", {}).get("same_camera_reacquisition_max_seconds", 20.0)
    )
    topology_gate = TopologyGate(
        topology_config=config.get("topology"),
        transition_rules=config.get("transition_rules")
    )
    temporal_gate = TemporalGate(
        temporal_config=config.get("temporal_constraints")
    )
    spatial_gate = SpatialGate(
        transition_rules=config.get("transition_rules")
    )
    candidate_filter = CandidateFilter(
        topology_gate=topology_gate,
        temporal_gate=temporal_gate,
        spatial_gate=spatial_gate
    )
    association_manager = AssociationManager(
        gallery=gallery,
        matcher=matcher,
        candidate_filter=candidate_filter,
        journey_manager=journey_manager,
        store=store,
        config=config
    )

    logging.info("Starting Multi-Camera processing...")
    cameras_info = []
    if "cameras" in config:
        for cam_id, cam_item in config["cameras"].items():
            cameras_info.append({
                "camera_id": cam_id,
                "video_path": cam_item.get("video_path", "")
            })
    else:
        cameras_info = [
            {"camera_id": "cctv1", "video_path": config.get("cctv1_video", "")},
            {"camera_id": "cctv2", "video_path": config.get("cctv2_video", "")}
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
        matcher=matcher,
        association_manager=association_manager,
        journey_manager=journey_manager,
        store=store
    )
    
    try:
        processor.run(max_frames=args.max_frames)
    except KeyboardInterrupt:
        logging.info("Interrupt received. Stopping processing...")
        processor.stop()
        
    logging.info("Processing complete.")
    if hasattr(gallery, "save"):
        gallery.save()

    logging.info("=== Phase 1 Journey Summary ===")
    for gid, id_state in journey_manager.identities.items():
        logging.info(
            f"Global ID {gid}: status={id_state.status.value}, "
            f"last_camera={id_state.last_camera}, last_zone={id_state.current_zone}, "
            f"events_count={len(id_state.journey_history)}"
        )

if __name__ == "__main__":
    main()
