"""
Obstacle and walkable ground-plane validation manager.
Provides fast geometric containment tests for walkable surfaces and physical obstacles.
"""

from typing import List, Tuple, Dict, Any, Optional
import math
import numpy as np
import cv2

from app.calibration.calibration_models import (
    Point2D,
    Obstacle,
    WalkableRegion
)


class ObstacleManager:
    """
    Manages walkable regions and obstacle geometry on the local ground plane.
    Performs point containment, collision detection, and distance-to-boundary calculations.
    """

    def __init__(
        self,
        walkable_regions: List[WalkableRegion],
        obstacles: List[Obstacle],
        boundaries: Optional[Dict[str, float]] = None
    ):
        self.walkable_regions = walkable_regions
        self.obstacles = obstacles
        self.boundaries = boundaries or {"x_min": -100.0, "x_max": 200.0, "y_min": -100.0, "y_max": 200.0}

        # Cache OpenCV polygon contours for sub-millisecond pointPolygonTest
        self._walkable_contours = []
        for w in self.walkable_regions:
            pts = np.array([[p.x, p.y] for p in w.polygon], dtype=np.float32)
            self._walkable_contours.append((w, pts))

        self._obstacle_contours = []
        self._obstacle_circles = []
        for obs in self.obstacles:
            if obs.obstacle_type == "circle" and obs.center and obs.radius:
                self._obstacle_circles.append((obs, obs.center.x, obs.center.y, obs.radius))
            elif obs.obstacle_type == "polygon" and obs.polygon:
                pts = np.array([[p.x, p.y] for p in obs.polygon], dtype=np.float32)
                self._obstacle_contours.append((obs, pts))

    def is_in_boundaries(self, x: float, y: float) -> bool:
        """Checks if coordinate falls within outer room bounding box."""
        return (
            self.boundaries.get("x_min", -1e5) <= x <= self.boundaries.get("x_max", 1e5) and
            self.boundaries.get("y_min", -1e5) <= y <= self.boundaries.get("y_max", 1e5)
        )

    def is_in_walkable_area(self, x: float, y: float) -> Tuple[bool, Optional[str]]:
        """
        Determines whether (x, y) falls inside any defined walkable polygon.
        
        Returns:
            is_walkable: bool
            region_name: str or None
        """
        pt = (float(x), float(y))
        for region, contour in self._walkable_contours:
            # pointPolygonTest: >= 0 means inside or on edge
            dist = cv2.pointPolygonTest(contour, pt, False)
            if dist >= 0:
                return True, region.name
        return False, None

    def is_in_obstacle(self, x: float, y: float) -> Tuple[bool, Optional[str]]:
        """
        Determines whether (x, y) falls inside any physical obstacle (circle or polygon).
        
        Returns:
            is_obstacle: bool
            obstacle_name: str or None
        """
        # 1. Test circular obstacles (e.g. Planter C1)
        for obs, cx, cy, radius in self._obstacle_circles:
            dist_sq = (x - cx)**2 + (y - cy)**2
            if dist_sq <= radius**2:
                return True, obs.name

        # 2. Test polygonal obstacles (e.g. dining table, kitchen counter)
        pt = (float(x), float(y))
        for obs, contour in self._obstacle_contours:
            dist = cv2.pointPolygonTest(contour, pt, False)
            if dist >= 0:
                return True, obs.name

        return False, None

    def validate_ground_position(self, x: float, y: float) -> Dict[str, Any]:
        """
        Comprehensive spatial validity test for a ground-plane coordinate.
        
        Returns dictionary with:
            is_valid: bool (True only if inside walkable area AND outside all obstacles)
            status: str ("VALID", "OUTSIDE_BOUNDS", "NON_WALKABLE", "OBSTACLE_COLLISION")
            message: human-readable explanation
            region_name: surface/region name if valid
            obstacle_name: collided obstacle if any
        """
        # Check boundary bounds
        if not self.is_in_boundaries(x, y):
            return {
                "is_valid": False,
                "status": "OUTSIDE_BOUNDS",
                "message": f"Point ({x:.1f}, {y:.1f}) is outside room boundaries",
                "region_name": None,
                "obstacle_name": None
            }

        # Check obstacle collision
        in_obs, obs_name = self.is_in_obstacle(x, y)
        if in_obs:
            return {
                "is_valid": False,
                "status": "OBSTACLE_COLLISION",
                "message": f"Point ({x:.1f}, {y:.1f}) collides with obstacle '{obs_name}'",
                "region_name": None,
                "obstacle_name": obs_name
            }

        # Check walkable area
        in_walk, reg_name = self.is_in_walkable_area(x, y)
        if not in_walk:
            return {
                "is_valid": False,
                "status": "NON_WALKABLE",
                "message": f"Point ({x:.1f}, {y:.1f}) falls outside designated walkable regions",
                "region_name": None,
                "obstacle_name": None
            }

        return {
            "is_valid": True,
            "status": "VALID",
            "message": f"Valid pedestrian position on '{reg_name}'",
            "region_name": reg_name,
            "obstacle_name": None
        }

    def get_distance_to_nearest_obstacle(self, x: float, y: float) -> float:
        """Computes Euclidean distance in ground units from (x, y) to the nearest obstacle boundary."""
        min_dist = float("inf")

        # Circles
        for _, cx, cy, radius in self._obstacle_circles:
            center_dist = math.sqrt((x - cx)**2 + (y - cy)**2)
            edge_dist = max(0.0, center_dist - radius)
            if edge_dist < min_dist:
                min_dist = edge_dist

        # Polygons
        pt = (float(x), float(y))
        for _, contour in self._obstacle_contours:
            dist = cv2.pointPolygonTest(contour, pt, True)
            edge_dist = 0.0 if dist >= 0 else abs(dist)
            if edge_dist < min_dist:
                min_dist = edge_dist

        return min_dist
