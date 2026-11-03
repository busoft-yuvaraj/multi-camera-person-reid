"""
Data models and type definitions for physical calibration and ground-plane mapping.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class Point2D:
    x: float
    y: float

    def to_list(self) -> List[float]:
        return [self.x, self.y]

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)

    def to_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=np.float64)


@dataclass
class CalibrationLandmark:
    id: str
    name: str
    image_point: Point2D       # [u, v] in pixel space
    ground_point: Point2D      # [X, Y] in local ground space

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CalibrationLandmark":
        img_pt = data["image_point"]
        gnd_pt = data["ground_point"]
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            image_point=Point2D(x=float(img_pt[0]), y=float(img_pt[1])),
            ground_point=Point2D(x=float(gnd_pt[0]), y=float(gnd_pt[1])),
        )


@dataclass
class Obstacle:
    id: str
    name: str
    obstacle_type: str        # 'circle' or 'polygon'
    center: Optional[Point2D] = None
    radius: Optional[float] = None
    polygon: List[Point2D] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Obstacle":
        obs_type = data.get("type", "polygon").lower()
        center = None
        radius = None
        polygon = []

        if obs_type == "circle":
            c_raw = data.get("center", [0.0, 0.0])
            center = Point2D(x=float(c_raw[0]), y=float(c_raw[1]))
            radius = float(data.get("radius", 0.0))
        elif obs_type == "polygon":
            poly_raw = data.get("polygon", [])
            polygon = [Point2D(x=float(p[0]), y=float(p[1])) for p in poly_raw]

        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            obstacle_type=obs_type,
            center=center,
            radius=radius,
            polygon=polygon,
        )


@dataclass
class WalkableRegion:
    id: str
    name: str
    surface_type: str         # 'tile', 'wood', 'carpet'
    polygon: List[Point2D]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WalkableRegion":
        poly_raw = data.get("polygon", [])
        polygon = [Point2D(x=float(p[0]), y=float(p[1])) for p in poly_raw]
        return cls(
            id=str(data["id"]),
            name=str(data.get("name", "")),
            surface_type=str(data.get("surface_type", "tile")),
            polygon=polygon,
        )


@dataclass
class GlobalTransform:
    translation: Point2D
    rotation_deg: float
    scale: float

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalTransform":
        t_raw = data.get("translation", [0.0, 0.0])
        return cls(
            translation=Point2D(x=float(t_raw[0]), y=float(t_raw[1])),
            rotation_deg=float(data.get("rotation_deg", 0.0)),
            scale=float(data.get("scale", 1.0)),
        )


@dataclass
class CameraCalibrationData:
    camera_id: str
    zone: str
    floor_plan_image: str
    coverage_map_reference: str
    image_resolution: Tuple[int, int]
    camera_orientation: str
    coordinate_convention: Dict[str, Any]
    landmarks: List[CalibrationLandmark]
    walkable_regions: List[WalkableRegion]
    obstacles: List[Obstacle]
    boundaries: Dict[str, float]
    global_transform: GlobalTransform

    @classmethod
    def from_dict(cls, cam_id: str, data: Dict[str, Any]) -> "CameraCalibrationData":
        res = data.get("image_resolution", [1280, 720])
        landmarks = [CalibrationLandmark.from_dict(l) for l in data.get("landmarks", [])]
        walkable = [WalkableRegion.from_dict(w) for w in data.get("walkable_regions", [])]
        obstacles = [Obstacle.from_dict(o) for o in data.get("obstacles", [])]
        transform = GlobalTransform.from_dict(data.get("global_transform", {}))
        
        return cls(
            camera_id=cam_id,
            zone=data.get("zone", cam_id),
            floor_plan_image=data.get("floor_plan_image", ""),
            coverage_map_reference=data.get("coverage_map_reference", ""),
            image_resolution=(int(res[0]), int(res[1])),
            camera_orientation=data.get("camera_orientation", ""),
            coordinate_convention=data.get("coordinate_convention", {}),
            landmarks=landmarks,
            walkable_regions=walkable,
            obstacles=obstacles,
            boundaries=data.get("boundaries", {}),
            global_transform=transform,
        )


@dataclass
class CalibrationResult:
    camera_id: str
    H_img2ground: np.ndarray       # 3x3 matrix: Image [u, v, 1] -> Ground [X, Y, 1]
    H_ground2img: np.ndarray       # 3x3 matrix: Ground [X, Y, 1] -> Image [u, v, 1]
    rms_reprojection_error: float  # RMS error in pixels
    max_reprojection_error: float  # Maximum error in pixels
    per_point_errors: List[Dict[str, Any]]
    condition_number: float
    is_valid: bool
    error_message: Optional[str] = None
