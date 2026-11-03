"""
Homography Estimation Engine implementing Normalized Direct Linear Transformation (DLT)
with SVD decomposition, condition number auditing, and reprojection error validation.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from app.calibration.calibration_models import (
    Point2D,
    CalibrationLandmark,
    CalibrationResult
)

logger = logging.getLogger(__name__)


class HomographyEstimator:
    """
    Computes planar homographies mapping between 2D camera image coordinates (u, v)
    and 2D ground-plane coordinates (X, Y).
    """

    @staticmethod
    def compute_hartley_normalization(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Hartley isotropic normalization matrix T such that the centroid
        of points is at the origin and the average Euclidean distance to the origin is sqrt(2).
        
        Args:
            points: N x 2 array of coordinates [x, y]
            
        Returns:
            normalized_points: N x 2 array
            T: 3 x 3 normalization transformation matrix
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[0] == 0:
            return pts, np.eye(3, dtype=np.float64)

        mean_x = np.mean(pts[:, 0])
        mean_y = np.mean(pts[:, 1])

        # Centered points
        centered_x = pts[:, 0] - mean_x
        centered_y = pts[:, 1] - mean_y
        dist = np.sqrt(centered_x**2 + centered_y**2)
        mean_dist = np.mean(dist)

        if mean_dist < 1e-8:
            scale = 1.0
        else:
            scale = np.sqrt(2.0) / mean_dist

        T = np.array([
            [scale, 0.0, -scale * mean_x],
            [0.0, scale, -scale * mean_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # Apply T to homogeneous coordinates
        n = pts.shape[0]
        homog = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])  # N x 3
        norm_homog = (T @ homog.T).T  # N x 3
        norm_pts = norm_homog[:, :2] / norm_homog[:, 2:3]

        return norm_pts, T

    @classmethod
    def estimate_homography_dlt(
        cls,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> np.ndarray:
        """
        Estimates the 3x3 homography matrix H mapping src_points -> dst_points
        using Normalized Direct Linear Transformation (DLT) solved via SVD.
        
        Equation: s * [x_dst, y_dst, 1]^T = H * [x_src, y_src, 1]^T
        
        Args:
            src_points: N x 2 array of source coordinates
            dst_points: N x 2 array of destination coordinates (N >= 4)
            
        Returns:
            H: 3 x 3 normalized homography matrix
        """
        src = np.asarray(src_points, dtype=np.float64)
        dst = np.asarray(dst_points, dtype=np.float64)

        n = src.shape[0]
        if n < 4:
            raise ValueError(f"DLT homography estimation requires >= 4 point correspondences, got {n}")

        # 1. Hartley Normalization
        norm_src, T_src = cls.compute_hartley_normalization(src)
        norm_dst, T_dst = cls.compute_hartley_normalization(dst)

        # 2. Build 2N x 9 constraint matrix A
        A = np.zeros((2 * n, 9), dtype=np.float64)
        for i in range(n):
            xs, ys = norm_src[i, 0], norm_src[i, 1]
            xd, yd = norm_dst[i, 0], norm_dst[i, 1]

            # Row 2i: -xs, -ys, -1, 0, 0, 0, xd*xs, xd*ys, xd
            A[2 * i] = [-xs, -ys, -1.0, 0.0, 0.0, 0.0, xd * xs, xd * ys, xd]
            # Row 2i+1: 0, 0, 0, -xs, -ys, -1, yd*xs, yd*ys, yd
            A[2 * i + 1] = [0.0, 0.0, 0.0, -xs, -ys, -1.0, yd * xs, yd * ys, yd]

        # 3. Solve A * h = 0 via Singular Value Decomposition (SVD)
        _, _, Vt = np.linalg.svd(A)
        H_norm = Vt[-1].reshape((3, 3))

        # 4. Denormalize: H = T_dst^{-1} * H_norm * T_src
        T_dst_inv = np.linalg.inv(T_dst)
        H = T_dst_inv @ H_norm @ T_src

        # 5. Normalize matrix so H[2, 2] == 1.0 (or unit Frobenius norm if H[2,2] is near zero)
        if abs(H[2, 2]) > 1e-8:
            H = H / H[2, 2]
        else:
            H = H / np.linalg.norm(H)

        return H

    @staticmethod
    def project_points(H: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Projects 2D points using 3x3 homography matrix H.
        
        Args:
            H: 3 x 3 homography matrix
            points: N x 2 array of coordinates [x, y]
            
        Returns:
            projected: N x 2 array of transformed coordinates
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.size == 0:
            return np.empty((0, 2), dtype=np.float64)

        n = pts.shape[0]
        homog = np.hstack([pts, np.ones((n, 1), dtype=np.float64)])  # N x 3
        proj_homog = (H @ homog.T).T  # N x 3

        # Guard against division by zero (points at optical horizon)
        denom = proj_homog[:, 2:3]
        denom = np.where(np.abs(denom) < 1e-9, 1e-9 * np.sign(denom + 1e-12), denom)

        proj = proj_homog[:, :2] / denom
        return proj

    @classmethod
    def calculate_reprojection_errors(
        cls,
        H: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray
    ) -> Tuple[float, float, np.ndarray]:
        """
        Computes forward Euclidean reprojection errors between projected source points and destination points.
        
        Returns:
            rms_error: Root-Mean-Square reprojection error
            max_error: Maximum individual point error
            per_point_errors: 1D array of individual point Euclidean errors
        """
        projected = cls.project_points(H, src_points)
        diff = projected - dst_points
        euclidean_errors = np.sqrt(np.sum(diff**2, axis=1))

        rms_error = float(np.sqrt(np.mean(euclidean_errors**2)))
        max_error = float(np.max(euclidean_errors))

        return rms_error, max_error, euclidean_errors

    @classmethod
    def calibrate_camera(
        cls,
        camera_id: str,
        landmarks: List[CalibrationLandmark]
    ) -> CalibrationResult:
        """
        Computes both forward (Ground -> Image) and inverse (Image -> Ground) homographies
        from a list of CalibrationLandmarks and evaluates calibration health.
        
        Args:
            camera_id: Identifier for camera (e.g. 'waiting_lobby')
            landmarks: List of CalibrationLandmark objects
            
        Returns:
            CalibrationResult dataclass
        """
        if len(landmarks) < 4:
            err_msg = f"Camera '{camera_id}' has only {len(landmarks)} landmarks. Minimum 4 required."
            logger.error(err_msg)
            return CalibrationResult(
                camera_id=camera_id,
                H_img2ground=np.eye(3, dtype=np.float64),
                H_ground2img=np.eye(3, dtype=np.float64),
                rms_reprojection_error=float("inf"),
                max_reprojection_error=float("inf"),
                per_point_errors=[],
                condition_number=float("inf"),
                is_valid=False,
                error_message=err_msg
            )

        img_pts = np.array([[lm.image_point.x, lm.image_point.y] for lm in landmarks], dtype=np.float64)
        gnd_pts = np.array([[lm.ground_point.x, lm.ground_point.y] for lm in landmarks], dtype=np.float64)

        try:
            # 1. H_ground2img: Ground [X, Y] -> Image [u, v]
            H_g2i = cls.estimate_homography_dlt(src_points=gnd_pts, dst_points=img_pts)
            # 2. H_img2ground: Image [u, v] -> Ground [X, Y]
            H_i2g = np.linalg.inv(H_g2i)
            if abs(H_i2g[2, 2]) > 1e-8:
                H_i2g = H_i2g / H_i2g[2, 2]

            # 3. Audit Matrix Health: Condition Number & Determinant
            cond_num = float(np.linalg.cond(H_g2i))
            det = float(np.linalg.det(H_g2i))

            # 4. Calculate Reprojection Errors in Image Pixel Space
            rms_err, max_err, pt_errs = cls.calculate_reprojection_errors(
                H=H_g2i,
                src_points=gnd_pts,
                dst_points=img_pts
            )

            per_point_list = []
            for i, lm in enumerate(landmarks):
                per_point_list.append({
                    "id": lm.id,
                    "name": lm.name,
                    "image_point": [lm.image_point.x, lm.image_point.y],
                    "ground_point": [lm.ground_point.x, lm.ground_point.y],
                    "reprojection_error_px": float(pt_errs[i])
                })

            # Validity threshold: RMS error < 10.0 pixels and condition number < 1e6
            is_valid = (rms_err < 10.0) and (cond_num < 1e6) and (abs(det) > 1e-9)
            err_msg = None if is_valid else f"Poor calibration: RMS={rms_err:.2f}px, cond={cond_num:.1e}"

            logger.info(
                f"Calibrated '{camera_id}': RMS Error = {rms_err:.2f} px, "
                f"Max Error = {max_err:.2f} px, Condition = {cond_num:.1f}, Valid = {is_valid}"
            )

            return CalibrationResult(
                camera_id=camera_id,
                H_img2ground=H_i2g,
                H_ground2img=H_g2i,
                rms_reprojection_error=rms_err,
                max_reprojection_error=max_err,
                per_point_errors=per_point_list,
                condition_number=cond_num,
                is_valid=is_valid,
                error_message=err_msg
            )

        except Exception as e:
            err_msg = f"Failed to compute homography for '{camera_id}': {e}"
            logger.exception(err_msg)
            return CalibrationResult(
                camera_id=camera_id,
                H_img2ground=np.eye(3, dtype=np.float64),
                H_ground2img=np.eye(3, dtype=np.float64),
                rms_reprojection_error=float("inf"),
                max_reprojection_error=float("inf"),
                per_point_errors=[],
                condition_number=float("inf"),
                is_valid=False,
                error_message=err_msg
            )
