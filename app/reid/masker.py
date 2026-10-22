import os
import cv2
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

logger = logging.getLogger("reid.masker")

class PersonMasker:
    """
    Lightweight person segmentation and outline masking for ReID embeddings.
    Suppresses background pixels (walls, floors, furniture) while preserving 
    person clothing and body features for OSNet feature extraction.
    
    Designed with CPU performance optimizations:
      - Runs small input resolution (imgsz=160) for fast CPU execution (~35-45ms)
      - Caches normalized masks per track to reuse across consecutive frames
      - Immediate zero-overhead fallback to original bounding box crop if disabled or unavailable
    """
    def __init__(
        self,
        model_path: str = "models/yolo11n-seg.pt",
        enabled: bool = True,
        imgsz: int = 160,
        bg_fill: Tuple[int, int, int] = (128, 128, 128),
        mask_reuse_frames: int = 6,
        device: str = "cpu"
    ):
        self.enabled = enabled
        self.imgsz = imgsz
        self.bg_fill = np.array(bg_fill, dtype=np.uint8)
        self.mask_reuse_frames = mask_reuse_frames
        self.device = device
        self.model = None
        self._mask_cache: Dict[Tuple[str, int], Dict[str, Any]] = {}

        if not self.enabled:
            logger.info("PersonMasker is disabled by configuration.")
            return

        if not os.path.exists(model_path):
            alt_path = model_path.replace('.pt', '.onnx') if '.pt' in model_path else model_path.replace('.onnx', '.pt')
            if os.path.exists(alt_path):
                model_path = alt_path

        if os.path.exists(model_path) and YOLO is not None:
            try:
                self.model = YOLO(model_path)
                logger.info(f"Loaded lightweight person segmentation model from {model_path} on {device}")
            except Exception as e:
                logger.warning(f"Failed to load segmentation model from {model_path}: {e}. Operating in fallback mode.")
                self.model = None
        else:
            logger.warning(f"Segmentation model not found at {model_path}. Operating in fallback mode (unmasked crops).")

    def is_enabled(self) -> bool:
        return self.enabled and self.model is not None

    def get_masked_crop(
        self, 
        crop: np.ndarray, 
        camera_id: str = "", 
        track_id: int = -1, 
        frame_idx: int = 0
    ) -> Tuple[np.ndarray, bool]:
        """
        Suppresses background pixels outside the person silhouette.
        
        Returns:
            (masked_crop: np.ndarray, was_masked: bool)
            If masking is disabled, fails, or is unavailable, returns (crop, False).
        """
        if not self.is_enabled() or crop is None or crop.size == 0:
            return crop, False

        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return crop, False

        track_key = (camera_id, track_id) if track_id >= 0 else None

        # -------------------------------------------------------------
        # Optimization: Safe Mask Reuse across adjacent ReID extractions
        # -------------------------------------------------------------
        if track_key and track_key in self._mask_cache:
            cache_entry = self._mask_cache[track_key]
            last_frame = cache_entry.get("frame_idx", -999)
            if (frame_idx - last_frame) <= self.mask_reuse_frames:
                cached_mask_norm = cache_entry.get("mask_norm")
                last_ratio = cache_entry.get("aspect_ratio", 1.0)
                curr_ratio = h / max(1.0, float(w))
                
                # Check aspect ratio stability (< 15% change)
                if cached_mask_norm is not None and abs(curr_ratio - last_ratio) / last_ratio < 0.15:
                    try:
                        mask_resized = cv2.resize(cached_mask_norm, (w, h), interpolation=cv2.INTER_NEAREST)
                        mask_bool = (mask_resized > 0)[:, :, None]
                        bg = np.full_like(crop, self.bg_fill)
                        masked_crop = np.where(mask_bool, crop, bg)
                        return masked_crop, True
                    except Exception as e:
                        logger.debug(f"Mask reuse resize failed: {e}")

        # -------------------------------------------------------------
        # Run Lightweight Segmentation Inference on Crop
        # -------------------------------------------------------------
        try:
            results = self.model.predict(
                source=crop,
                imgsz=self.imgsz,
                classes=[0],  # Person class only
                conf=0.25,
                device=self.device,
                verbose=False
            )

            if not results or not results[0].masks or len(results[0].masks) == 0:
                return crop, False

            # Select the mask with largest coverage area
            masks_data = results[0].masks.data.cpu().numpy()
            best_mask = None
            best_area = 0

            for m in masks_data:
                area = np.count_nonzero(m > 0.5)
                if area > best_area:
                    best_area = area
                    best_mask = m

            if best_mask is None:
                return crop, False

            # Resize binary mask to exact crop dimensions
            mask_resized = cv2.resize(best_mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_bin = (mask_resized > 0.5).astype(np.uint8)

            coverage = float(mask_bin.mean())
            # Sanity check: valid person mask must cover at least 10% and not exceed 98%
            if coverage < 0.10 or coverage > 0.98:
                return crop, False

            # Cache normalized mask (at 128x64 or imgsz resolution) to save RAM
            if track_key:
                mask_norm = cv2.resize(mask_bin, (64, 128), interpolation=cv2.INTER_NEAREST)
                self._mask_cache[track_key] = {
                    "mask_norm": mask_norm,
                    "frame_idx": frame_idx,
                    "aspect_ratio": h / max(1.0, float(w))
                }
                # Limit cache size to avoid unbounded memory growth
                if len(self._mask_cache) > 100:
                    oldest_key = min(self._mask_cache.keys(), key=lambda k: self._mask_cache[k]["frame_idx"])
                    self._mask_cache.pop(oldest_key, None)

            # Apply background neutralization to neutral gray (preserves edge features for OSNet)
            bg = np.full_like(crop, self.bg_fill)
            mask_bool = (mask_bin > 0)[:, :, None]
            masked_crop = np.where(mask_bool, crop, bg)
            return masked_crop, True

        except Exception as e:
            logger.debug(f"PersonMasker error: {e}. Falling back to original crop.")
            return crop, False

    def get_contours(
        self,
        crop: np.ndarray,
        camera_id: str = "",
        track_id: int = -1,
        x1: int = 0,
        y1: int = 0,
        frame_idx: int = 0
    ) -> list:
        """
        Extracts person polygon outline contours mapped to full-frame coordinates.
        Reuses cached segmentation masks for fast zero-overhead rendering.
        Returns a list of contour coordinate arrays ready for cv2.polylines.
        """
        if not self.is_enabled() or crop is None or crop.size == 0:
            return []

        h, w = crop.shape[:2]
        if h < 20 or w < 10:
            return []

        track_key = (camera_id, track_id) if track_id >= 0 else None
        mask_bin = None

        # Check if we have a cached normalized mask
        if track_key and track_key in self._mask_cache:
            cache_entry = self._mask_cache[track_key]
            cached_mask_norm = cache_entry.get("mask_norm")
            if cached_mask_norm is not None:
                try:
                    mask_bin = cv2.resize(cached_mask_norm, (w, h), interpolation=cv2.INTER_NEAREST)
                except Exception:
                    mask_bin = None

        # If not cached, compute fresh mask
        if mask_bin is None:
            _, was_masked = self.get_masked_crop(crop, camera_id, track_id, frame_idx)
            if track_key and track_key in self._mask_cache:
                cached_mask_norm = self._mask_cache[track_key].get("mask_norm")
                if cached_mask_norm is not None:
                    try:
                        mask_bin = cv2.resize(cached_mask_norm, (w, h), interpolation=cv2.INTER_NEAREST)
                    except Exception:
                        mask_bin = None

        if mask_bin is None or np.count_nonzero(mask_bin) < 50:
            return []

        # Find external contours
        try:
            contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            frame_contours = []
            for cnt in contours:
                if len(cnt) >= 3 and cv2.contourArea(cnt) > 80:
                    cnt_offset = cnt.copy()
                    cnt_offset[:, :, 0] += x1
                    cnt_offset[:, :, 1] += y1
                    frame_contours.append(cnt_offset)
            return frame_contours
        except Exception as e:
            logger.debug(f"Error computing contours: {e}")
            return []

    def reset_track(self, camera_id: str, track_id: int):
        """Cleans up cache when track terminates."""
        self._mask_cache.pop((camera_id, track_id), None)
