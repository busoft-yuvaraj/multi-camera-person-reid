"""
World Coordinate Mapper Engine.
Transforms 2D image footpoints into local ground-plane coordinates and global building coordinates,
performing obstacle collision checks, walkable boundary tests, and cross-camera metric handoffs.
"""

import logging
import math
from typing import Dict, Any, Tuple, Optional, List
import yaml
import numpy as np

from app.calibration.calibration_models import (
    Point2D,
    CameraCalibrationData,
    CalibrationResult
)
from app.calibration.homography_estimator import HomographyEstimator
from app.calibration.obstacle_manager import ObstacleManager

logger = logging.getLogger(__name__)


class WorldCoordinateMapper:
    """
    Central coordinator for physical camera calibration and coordinate mapping.
    Manages camera-specific homographies, local-to-global building transformations,
    and ground-plane spatial validity gating.
    """

    def __init__(self, config_path: str = "config/camera_calibration.yaml"):
        self.config_path = config_path
        self.cameras: Dict[str, CameraCalibrationData] = {}
        self.calibration_results: Dict[str, CalibrationResult] = {}
        self.obstacle_managers: Dict[str, ObstacleManager] = {}
        self.common_building_frame: Dict[str, Any] = {}
        self.topology: Dict[str, Any] = {}

        self.load_configuration(config_path)

    def load_configuration(self, config_path: str) -> None:
        """Loads calibration configuration and computes homographies for all cameras."""
        logger.info(f"Loading camera calibration configuration from '{config_path}'...")
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f)

        self.common_building_frame = raw_cfg.get("common_building_frame", {})
        self.topology = self.common_building_frame.get("topology", {})

        cams_dict = raw_cfg.get("cameras", {})
        for cam_id, cam_data in cams_dict.items():
            calib_data = CameraCalibrationData.from_dict(cam_id, cam_data)
            self.cameras[cam_id] = calib_data

            # Initialize Obstacle Manager
            self.obstacle_managers[cam_id] = ObstacleManager(
                walkable_regions=calib_data.walkable_regions,
                obstacles=calib_data.obstacles,
                boundaries=calib_data.boundaries
            )

            # Compute Homography Matrix using DLT
            res = HomographyEstimator.calibrate_camera(
                camera_id=cam_id,
                landmarks=calib_data.landmarks
            )
            self.calibration_results[cam_id] = res

        logger.info(f"Successfully initialized calibration for {len(self.cameras)} cameras: {list(self.cameras.keys())}")

    def get_calibration_result(self, camera_id: str) -> Optional[CalibrationResult]:
        """Returns the CalibrationResult for a given camera."""
        return self.calibration_results.get(camera_id)

    def pixel_to_local_ground(
        self,
        camera_id: str,
        u: float,
        v: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Projects 2D image pixel footpoint (u, v) to 2D local ground coordinates (X, Y).
        
        Returns:
            (x_ground, y_ground, success_flag)
        """
        res = self.calibration_results.get(camera_id)
        if not res or not res.is_valid:
            logger.warning(f"No valid calibration for camera '{camera_id}'")
            return None, None, False

        H_inv = res.H_img2ground
        pt_homog = np.array([float(u), float(v), 1.0], dtype=np.float64)
        gnd_homog = H_inv @ pt_homog

        denom = gnd_homog[2]
        # Guard against points at or behind camera horizon (denom near zero or negative)
        if abs(denom) < 1e-7:
            return None, None, False

        x_gnd = float(gnd_homog[0] / denom)
        y_gnd = float(gnd_homog[1] / denom)

        return x_gnd, y_gnd, True

    def local_ground_to_pixel(
        self,
        camera_id: str,
        x_ground: float,
        y_ground: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        Projects 2D local ground coordinate (X, Y) back to image pixel space (u, v).
        
        Returns:
            (u, v, success_flag)
        """
        res = self.calibration_results.get(camera_id)
        if not res or not res.is_valid:
            return None, None, False

        H = res.H_ground2img
        pt_homog = np.array([float(x_ground), float(y_ground), 1.0], dtype=np.float64)
        img_homog = H @ pt_homog

        denom = img_homog[2]
        if abs(denom) < 1e-7:
            return None, None, False

        u = float(img_homog[0] / denom)
        v = float(img_homog[1] / denom)

        return u, v, True

    def local_to_global_building(
        self,
        camera_id: str,
        x_local: float,
        y_local: float
    ) -> Tuple[float, float]:
        """
        Transforms local ground-plane coordinates into the common building coordinate frame.
        Applies 2D Euclidean rigid body rotation, scaling, and translation.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return x_local, y_local

        gt = cam.global_transform
        rad = math.radians(gt.rotation_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # Scale and rotate
        xs = x_local * gt.scale
        ys = y_local * gt.scale
        x_rot = xs * cos_t - ys * sin_t
        y_rot = xs * sin_t + ys * cos_t

        # Translate
        x_global = x_rot + gt.translation.x
        y_global = y_rot + gt.translation.y

        return float(x_global), float(y_global)

    def global_building_to_local(
        self,
        camera_id: str,
        x_global: float,
        y_global: float
    ) -> Tuple[float, float]:
        """
        Transforms common building coordinates back into a camera's local ground-plane frame.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return x_global, y_global

        gt = cam.global_transform
        # Inverse translation
        dx = x_global - gt.translation.x
        dy = y_global - gt.translation.y

        # Inverse rotation
        rad = math.radians(-gt.rotation_deg)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)
        x_rot = dx * cos_t - dy * sin_t
        y_rot = dx * sin_t + dy * cos_t

        # Inverse scale
        scale = max(1e-6, gt.scale)
        x_local = x_rot / scale
        y_local = y_rot / scale

        return float(x_local), float(y_local)

    def pixel_to_global_building(
        self,
        camera_id: str,
        u: float,
        v: float
    ) -> Tuple[Optional[float], Optional[float], bool]:
        """
        End-to-end projection from image pixel coordinates directly to common building coordinates.
        """
        x_loc, y_loc, ok = self.pixel_to_local_ground(camera_id, u, v)
        if not ok or x_loc is None or y_loc is None:
            return None, None, False

        x_glob, y_glob = self.local_to_global_building(camera_id, x_loc, y_loc)
        return x_glob, y_glob, True

    def validate_footpoint(
        self,
        camera_id: str,
        u: float,
        v: float
    ) -> Dict[str, Any]:
        """
        Evaluates a detected pedestrian footpoint (u, v):
        1. Checks image boundary constraints.
        2. Projects into local ground plane.
        3. Tests obstacle collision and walkable region containment.
        4. Projects into global building coordinates.
        
        Returns:
            Dict containing full spatial audit result.
        """
        cam = self.cameras.get(camera_id)
        if not cam:
            return {
                "is_valid": False,
                "status": "UNKNOWN_CAMERA",
                "message": f"Camera '{camera_id}' is not configured",
                "local_ground": None,
                "global_building": None
            }

        w, h = cam.image_resolution
        if not (0 <= u <= w and 0 <= v <= h):
            return {
                "is_valid": False,
                "status": "OUT_OF_FRAME",
                "message": f"Footpoint ({u:.1f}, {v:.1f}) is outside image resolution ({w}x{h})",
                "local_ground": None,
                "global_building": None
            }

        # Project to local ground
        x_loc, y_loc, ok = self.pixel_to_local_ground(camera_id, u, v)
        if not ok or x_loc is None or y_loc is None:
            return {
                "is_valid": False,
                "status": "HORIZON_PROJECTION_ERROR",
                "message": f"Footpoint ({u:.1f}, {v:.1f}) could not be projected to ground plane",
                "local_ground": None,
                "global_building": None
            }

        # Validate against walkable regions and obstacles
        obs_mgr = self.obstacle_managers[camera_id]
        ground_status = obs_mgr.validate_ground_position(x_loc, y_loc)

        # Global building position
        x_glob, y_glob = self.local_to_global_building(camera_id, x_loc, y_loc)

        return {
            "is_valid": ground_status["is_valid"],
            "status": ground_status["status"],
            "message": ground_status["message"],
            "region_name": ground_status["region_name"],
            "obstacle_name": ground_status["obstacle_name"],
            "local_ground": [round(x_loc, 2), round(y_loc, 2)],
            "global_building": [round(x_glob, 2), round(y_glob, 2)]
        }

    def compute_ground_distance(
        self,
        cam_a: str,
        pos_a: Tuple[float, float],
        cam_b: str,
        pos_b: Tuple[float, float]
    ) -> float:
        """
        Computes the Euclidean physical distance between two observations in the common building frame.
        """
        if cam_a == cam_b:
            return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

        g_ax, g_ay = self.local_to_global_building(cam_a, pos_a[0], pos_a[1])
        g_bx, g_by = self.local_to_global_building(cam_b, pos_b[0], pos_b[1])

        return math.sqrt((g_ax - g_bx)**2 + (g_ay - g_by)**2)
