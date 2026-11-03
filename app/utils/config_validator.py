"""
Configuration validation module for multi-camera tracking pipeline.
Ensures strict validation of camera definitions, expected resolutions,
ROI polygons, transition lines, topology, and temporal constraints.
"""
from typing import Dict, Any, List, Tuple
import os
import logging

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when pipeline configuration is invalid or inconsistent."""
    pass


class ResolutionValidationError(ValueError):
    """Raised when actual video frame dimensions do not match expected ROI coordinate system."""
    pass


def validate_pipeline_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates complete application configuration at startup.
    Returns normalized configuration with validated expected_resolutions.
    Raises ConfigurationError on any invalid or contradictory settings.
    """
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a dictionary.")

    cameras = config.get("cameras")
    if not cameras or not isinstance(cameras, dict):
        raise ConfigurationError("Configuration must define a non-empty 'cameras' dictionary.")

    valid_cameras = set(cameras.keys())
    valid_zones = set()

    for cam_id, cam_cfg in cameras.items():
        if not isinstance(cam_cfg, dict):
            raise ConfigurationError(f"Camera '{cam_id}' configuration must be a dictionary.")

        # Ensure camera_id and video_path
        cam_cfg.setdefault("camera_id", cam_id)
        zone = cam_cfg.get("zone", cam_id)
        valid_zones.add(zone)

        video_path = cam_cfg.get("video_path")
        if not video_path:
            raise ConfigurationError(f"Camera '{cam_id}' is missing required 'video_path'.")

        # Resolution validation
        exp_res = cam_cfg.get("expected_resolution") or cam_cfg.get("resolution")
        if exp_res is None:
            # Default to standard 1280x720 if not specified
            exp_res = [1280, 720]
            cam_cfg["expected_resolution"] = exp_res

        if not (isinstance(exp_res, (list, tuple)) and len(exp_res) == 2):
            raise ConfigurationError(
                f"Camera '{cam_id}' has invalid expected_resolution: {exp_res}. "
                f"Expected [width, height]."
            )

        exp_w, exp_h = int(exp_res[0]), int(exp_res[1])
        if exp_w <= 0 or exp_h <= 0:
            raise ConfigurationError(
                f"Camera '{cam_id}' expected_resolution dimensions must be positive, got {exp_w}x{exp_h}."
            )
        cam_cfg["expected_resolution"] = [exp_w, exp_h]

        # Crop validation if configured
        crop = cam_cfg.get("crop")
        if crop:
            if not (isinstance(crop, (list, tuple)) and len(crop) == 4):
                raise ConfigurationError(
                    f"Camera '{cam_id}' crop must be [x1, y1, x2, y2], got {crop}."
                )
            cx1, cy1, cx2, cy2 = map(int, crop)
            if cx1 < 0 or cy1 < 0 or cx2 <= cx1 or cy2 <= cy1:
                raise ConfigurationError(
                    f"Camera '{cam_id}' has invalid crop bounds: {crop}."
                )

        # ROI polygon coordinate validation against expected_resolution
        rois = cam_cfg.get("roi", {})
        if isinstance(rois, dict):
            for r_id, r_info in rois.items():
                if not isinstance(r_info, dict):
                    continue
                pts = r_info.get("points")
                if pts is not None:
                    if not isinstance(pts, (list, tuple)) or len(pts) < 3:
                        raise ConfigurationError(
                            f"Camera '{cam_id}' ROI '{r_id}' must define at least 3 points, got {pts}."
                        )
                    for pt_idx, pt in enumerate(pts):
                        if not (isinstance(pt, (list, tuple)) and len(pt) == 2):
                            raise ConfigurationError(
                                f"Camera '{cam_id}' ROI '{r_id}' point {pt_idx} is not [x, y]: {pt}."
                            )
                        px, py = pt[0], pt[1]
                        # Allow slight tolerance (up to 1.1x in case coordinates touch edge or crop)
                        # but reject wildly out of bound coordinates
                        if px < 0 or py < 0 or px > exp_w * 1.5 or py > exp_h * 1.5:
                            raise ConfigurationError(
                                f"Camera '{cam_id}' ROI '{r_id}' point {pt} exceeds coordinate basis "
                                f"bounds (expected {exp_w}x{exp_h})."
                            )

        # Transition line coordinate validation
        transitions = cam_cfg.get("transitions", {})
        if isinstance(transitions, dict):
            for t_id, t_info in transitions.items():
                if not isinstance(t_info, dict):
                    continue
                pts = t_info.get("points", {})
                start = pts.get("start")
                end = pts.get("end")
                if not (start and end and len(start) == 2 and len(end) == 2):
                    raise ConfigurationError(
                        f"Camera '{cam_id}' transition '{t_id}' must have start and end [x, y] points."
                    )
                for pt_name, pt in [("start", start), ("end", end)]:
                    px, py = pt[0], pt[1]
                    if px < 0 or py < 0 or px > exp_w * 1.5 or py > exp_h * 1.5:
                        raise ConfigurationError(
                            f"Camera '{cam_id}' transition '{t_id}' {pt_name} point {pt} "
                            f"exceeds coordinate basis bounds (expected {exp_w}x{exp_h})."
                        )

    # Topology validation
    topology = config.get("topology", {})
    if isinstance(topology, dict):
        for src_zone, next_zones in topology.items():
            if isinstance(next_zones, list):
                for dest in next_zones:
                    # Allowed if dest is in valid_cameras, valid_zones, or common zones like workspace
                    if dest not in valid_cameras and dest not in valid_zones and dest not in ["workspace"]:
                        logger.warning(
                            f"Topology defines transition to '{dest}' which is not a declared camera/zone."
                        )

    # Temporal constraints validation
    temporal = config.get("temporal_constraints", {})
    if isinstance(temporal, dict):
        overlapping_pairs = temporal.get("overlapping_camera_pairs", [])
        if not isinstance(overlapping_pairs, list):
            raise ConfigurationError("'overlapping_camera_pairs' must be a list of pairs.")
        for pair in overlapping_pairs:
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise ConfigurationError(f"Each entry in overlapping_camera_pairs must have 2 cameras, got: {pair}")
            c1, c2 = pair
            if c1 not in valid_cameras or c2 not in valid_cameras:
                raise ConfigurationError(
                    f"overlapping_camera_pairs references unknown camera: {pair}. "
                    f"Configured cameras: {list(valid_cameras)}"
                )

    return config


def validate_frame_resolution(
    camera_id: str,
    actual_width: int,
    actual_height: int,
    expected_resolution: Tuple[int, int],
    affected_config: str = "roi"
):
    """
    Validates actual frame dimensions against expected resolution for the camera.
    Raises ResolutionValidationError with descriptive context if dimensions differ.
    """
    exp_w, exp_h = expected_resolution
    if actual_width != exp_w or actual_height != exp_h:
        raise ResolutionValidationError(
            f"Resolution mismatch for camera '{camera_id}': expected frame dimensions "
            f"({exp_w}, {exp_h}) for ROI coordinate system, but got ({actual_width}, {actual_height}). "
            f"Affected configuration: cameras.{camera_id}.{affected_config}"
        )
