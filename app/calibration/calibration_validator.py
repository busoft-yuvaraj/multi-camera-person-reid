"""
Calibration Validation Suite.
Evaluates homography matrix numerical stability, condition numbers,
reprojection error residuals, and cross-camera topological consistency.
"""

from typing import Dict, Any, List
import numpy as np

from app.calibration.calibration_models import CalibrationResult
from app.calibration.world_coordinate_mapper import WorldCoordinateMapper


class CalibrationValidator:
    """
    Performs rigorous verification of camera calibration matrices and ground configurations.
    """

    def __init__(self, mapper: WorldCoordinateMapper):
        self.mapper = mapper

    def validate_all_cameras(self) -> Dict[str, Any]:
        """
        Runs comprehensive validation across all configured cameras and cross-camera handoffs.
        """
        results = {}
        all_valid = True

        for cam_id, calib_res in self.mapper.calibration_results.items():
            cam_data = self.mapper.cameras[cam_id]
            cam_eval = self.evaluate_single_camera(cam_id, calib_res, cam_data)
            results[cam_id] = cam_eval
            if not cam_eval["is_healthy"]:
                all_valid = False

        # Evaluate Cross-Camera Topological Handoff Consistency
        handoff_eval = self.evaluate_topological_handoffs()

        return {
            "overall_status": "HEALTHY" if all_valid and handoff_eval["all_handoffs_valid"] else "WARNING",
            "cameras": results,
            "topology_validation": handoff_eval
        }

    def evaluate_single_camera(
        self,
        cam_id: str,
        res: CalibrationResult,
        cam_data: Any
    ) -> Dict[str, Any]:
        """
        Audits single camera homography stability and accuracy.
        """
        num_landmarks = len(cam_data.landmarks)
        is_healthy = True
        warnings = []

        # 1. Landmark count check
        if num_landmarks < 4:
            is_healthy = False
            warnings.append(f"Insufficient landmarks: {num_landmarks} (min 4 required)")

        # 2. RMS Error Check
        if res.rms_reprojection_error > 5.0:
            warnings.append(f"High RMS reprojection error: {res.rms_reprojection_error:.2f} px (> 5.0 px)")
            if res.rms_reprojection_error > 12.0:
                is_healthy = False

        # 3. Condition Number Check
        if res.condition_number > 1e5:
            warnings.append(f"High matrix condition number: {res.condition_number:.1e} (near singular)")
            if res.condition_number > 1e7:
                is_healthy = False

        # 4. Round-Trip Inversion Error Check
        # Project landmarks ground -> img -> ground
        gnd_pts = np.array([[lm.ground_point.x, lm.ground_point.y] for lm in cam_data.landmarks], dtype=np.float64)
        img_proj = self.mapper.calibration_results[cam_id].H_ground2img
        H_inv = self.mapper.calibration_results[cam_id].H_img2ground

        roundtrip_errs = []
        for g_pt in gnd_pts:
            # forward
            u, v, ok1 = self.mapper.local_ground_to_pixel(cam_id, g_pt[0], g_pt[1])
            if ok1 and u is not None and v is not None:
                # inverse
                x_rec, y_rec, ok2 = self.mapper.pixel_to_local_ground(cam_id, u, v)
                if ok2 and x_rec is not None and y_rec is not None:
                    err = np.sqrt((x_rec - g_pt[0])**2 + (y_rec - g_pt[1])**2)
                    roundtrip_errs.append(err)

        mean_roundtrip = float(np.mean(roundtrip_errs)) if roundtrip_errs else float("inf")
        if mean_roundtrip > 0.05:
            warnings.append(f"Noticeable round-trip projection discrepancy: {mean_roundtrip:.4f} ground units")

        return {
            "camera_id": cam_id,
            "num_landmarks": num_landmarks,
            "rms_reprojection_error_px": round(res.rms_reprojection_error, 2),
            "max_reprojection_error_px": round(res.max_reprojection_error, 2),
            "condition_number": float(f"{res.condition_number:.2e}"),
            "mean_roundtrip_error_units": round(mean_roundtrip, 6),
            "is_healthy": is_healthy,
            "warnings": warnings,
            "per_point_errors": res.per_point_errors
        }

    def evaluate_topological_handoffs(self) -> Dict[str, Any]:
        """
        Evaluates cross-camera boundary continuity for connected pairs in the topology.
        """
        handoffs = {}
        all_valid = True

        # Check Passage 1 <-> Pantry Overlap Threshold
        # In Passage 1: P1 (0, 0) is the pantry doorway left threshold
        # In Pantry: P5 (85, 90) is the passage doorway threshold
        if "passage1" in self.mapper.cameras and "pantry" in self.mapper.cameras:
            # Global coordinates of Passage far threshold
            g_pass_x, g_pass_y = self.mapper.local_to_global_building("passage1", 0.0, 10.0)
            # Global coordinates of Pantry carpet threshold
            g_pantry_x, g_pantry_y = self.mapper.local_to_global_building("pantry", 85.0, 90.0)

            dist_overlap = np.sqrt((g_pass_x - g_pantry_x)**2 + (g_pass_y - g_pantry_y)**2)
            is_overlap_valid = dist_overlap < 15.0  # within overlap tolerance

            handoffs["passage_to_pantry"] = {
                "connection_type": "overlap",
                "passage_threshold_global": [round(g_pass_x, 1), round(g_pass_y, 1)],
                "pantry_threshold_global": [round(g_pantry_x, 1), round(g_pantry_y, 1)],
                "spatial_discrepancy_units": round(dist_overlap, 2),
                "is_consistent": is_overlap_valid
            }
            if not is_overlap_valid:
                all_valid = False

        # Check Passage 1 <-> Waiting Lobby Straight Handoff
        # Passage far end (P1/P2 at X=0) continues straight into Lobby entrance (P3/P4 at Y=100)
        if "passage1" in self.mapper.cameras and "waiting_lobby" in self.mapper.cameras:
            g_pass_end_x, g_pass_end_y = self.mapper.local_to_global_building("passage1", 0.0, 10.0)
            g_lobby_ent_x, g_lobby_ent_y = self.mapper.local_to_global_building("waiting_lobby", 50.0, 100.0)

            dist_straight = np.sqrt((g_pass_end_x - g_lobby_ent_x)**2 + (g_pass_end_y - g_lobby_ent_y)**2)
            is_straight_valid = dist_straight < 25.0

            handoffs["passage_to_lobby"] = {
                "connection_type": "adjacent_straight",
                "passage_exit_global": [round(g_pass_end_x, 1), round(g_pass_end_y, 1)],
                "lobby_entrance_global": [round(g_lobby_ent_x, 1), round(g_lobby_ent_y, 1)],
                "spatial_discrepancy_units": round(dist_straight, 2),
                "is_consistent": is_straight_valid
            }
            if not is_straight_valid:
                all_valid = False

        return {
            "all_handoffs_valid": all_valid,
            "handoff_checks": handoffs
        }
