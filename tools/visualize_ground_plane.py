"""
Ground-Plane Visualizer Tool.
Renders 2D top-down ground plans with walkable surfaces, obstacles, calibration landmarks,
and synthetic pedestrian footpoints independently of live CCTV video feeds.
"""

import os
import argparse
import numpy as np
import cv2

from app.calibration.world_coordinate_mapper import WorldCoordinateMapper


def render_camera_ground_plan(
    mapper: WorldCoordinateMapper,
    camera_id: str,
    canvas_size: tuple = (800, 800),
    out_path: str = None
) -> np.ndarray:
    """
    Renders the local top-down ground plan for a specific camera.
    """
    cam_data = mapper.cameras.get(camera_id)
    if not cam_data:
        raise ValueError(f"Unknown camera '{camera_id}'")

    w_canvas, h_canvas = canvas_size
    canvas = np.ones((h_canvas, w_canvas, 3), dtype=np.uint8) * 245

    # Determine coordinate bounds
    bounds = cam_data.boundaries
    x_min = bounds.get("x_min", -10.0)
    x_max = bounds.get("x_max", 110.0)
    y_min = bounds.get("y_min", -10.0)
    y_max = bounds.get("y_max", 110.0)

    # Padding
    pad = 60
    scale_x = (w_canvas - 2 * pad) / (x_max - x_min)
    scale_y = (h_canvas - 2 * pad) / (y_max - y_min)

    def to_screen(gx, gy):
        sx = int(pad + (gx - x_min) * scale_x)
        sy = int(h_canvas - pad - (gy - y_min) * scale_y)
        return (sx, sy)

    # Draw Grid Lines
    for gx in np.arange(0, x_max + 1, 10):
        p1 = to_screen(gx, y_min)
        p2 = to_screen(gx, y_max)
        cv2.line(canvas, p1, p2, (225, 230, 235), 1)
        cv2.putText(canvas, f"{gx:.0f}", (p1[0] - 8, h_canvas - pad + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 140), 1)

    for gy in np.arange(0, y_max + 1, 10):
        p1 = to_screen(x_min, gy)
        p2 = to_screen(x_max, gy)
        cv2.line(canvas, p1, p2, (225, 230, 235), 1)
        cv2.putText(canvas, f"{gy:.0f}", (pad - 30, p1[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 130, 140), 1)

    # 1. Draw Walkable Regions
    surface_colors = {
        "tile": (220, 225, 230),     # light grey
        "wood": (185, 215, 235),     # warm wood tan (BGR)
        "carpet": (210, 205, 215)    # subtle purple carpet (BGR)
    }

    for region in cam_data.walkable_regions:
        pts = np.array([to_screen(p.x, p.y) for p in region.polygon], dtype=np.int32)
        col = surface_colors.get(region.surface_type, (220, 225, 230))
        cv2.fillPoly(canvas, [pts], col)
        cv2.polylines(canvas, [pts], True, (160, 170, 180), 2)

    # 2. Draw Obstacles
    for obs in cam_data.obstacles:
        if obs.obstacle_type == "circle" and obs.center and obs.radius:
            sc = to_screen(obs.center.x, obs.center.y)
            sr = int(obs.radius * scale_x)
            # Planter green fill
            cv2.circle(canvas, sc, sr, (180, 220, 180), -1)
            cv2.circle(canvas, sc, sr, (80, 160, 80), 2)
            cv2.circle(canvas, sc, max(2, sr // 3), (60, 120, 60), -1)
            cv2.putText(canvas, obs.name.split()[0], (sc[0] - 15, sc[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (30, 60, 30), 1)

        elif obs.obstacle_type == "polygon" and obs.polygon:
            pts = np.array([to_screen(p.x, p.y) for p in obs.polygon], dtype=np.int32)
            # Furniture / counter grey fill
            cv2.fillPoly(canvas, [pts], (180, 180, 190))
            cv2.polylines(canvas, [pts], True, (100, 100, 120), 2)
            # Label
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            cv2.putText(canvas, obs.name.split()[0], (cx - 20, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 50), 1)

    # 3. Draw Calibration Landmarks
    for lm in cam_data.landmarks:
        sp = to_screen(lm.ground_point.x, lm.ground_point.y)
        cv2.circle(canvas, sp, 6, (0, 0, 255), -1)
        cv2.circle(canvas, sp, 8, (0, 0, 0), 2)
        tag = f"{lm.id} ({lm.ground_point.x:.0f},{lm.ground_point.y:.0f})"
        cv2.putText(canvas, tag, (sp[0] + 8, sp[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 150), 1)

    # 4. Header and Legend
    cv2.rectangle(canvas, (0, 0), (w_canvas, 48), (28, 35, 45), -1)
    title = f"{cam_data.camera_id.upper()} - Predefined Ground Plan (Local Coordinates)"
    cv2.putText(canvas, title, (20, 31), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1)

    # Status summary
    res = mapper.calibration_results.get(camera_id)
    if res:
        stat_txt = f"RMS Error: {res.rms_reprojection_error:.2f}px | Cond: {res.condition_number:.1e} | Landmarks: {len(cam_data.landmarks)}"
        cv2.putText(canvas, stat_txt, (w_canvas - 450, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, canvas)
        print(f"Saved local ground plan: {out_path}")

    return canvas


def render_all_ground_plans(config_path: str = "config/camera_calibration.yaml", out_dir: str = "outputs/calibration_visualizations"):
    """
    Renders top-down ground plans for all three cameras and saves them to out_dir.
    """
    os.makedirs(out_dir, exist_ok=True)
    mapper = WorldCoordinateMapper(config_path=config_path)

    for cam_id in mapper.cameras.keys():
        out_file = os.path.join(out_dir, f"{cam_id}_top_down_ground_plan.png")
        render_camera_ground_plan(mapper, cam_id, out_path=out_file)

    # Render a 3-Panel Side-by-Side Overview
    imgs = []
    for cam_id in ["passage1", "pantry", "waiting_lobby"]:
        if cam_id in mapper.cameras:
            im = render_camera_ground_plan(mapper, cam_id, canvas_size=(600, 600))
            imgs.append(im)

    if imgs:
        composite = np.hstack(imgs)
        comp_path = os.path.join(out_dir, "multi_camera_ground_plans_composite.png")
        cv2.imwrite(comp_path, composite)
        print(f"Saved multi-camera composite: {comp_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render preconfigured 2D ground plans")
    parser.add_argument("--config", default="config/camera_calibration.yaml", help="Path to calibration config")
    parser.add_argument("--out-dir", default="outputs/calibration_visualizations", help="Output directory")
    args = parser.parse_args()

    render_all_ground_plans(args.config, args.out_dir)
