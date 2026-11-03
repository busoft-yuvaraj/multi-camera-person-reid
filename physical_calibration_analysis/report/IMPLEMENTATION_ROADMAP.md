# Proposed Physical Calibration Implementation Roadmap

> **Document Classification:** Engineering Implementation Blueprint & Architecture Specification  
> **Status:** PROPOSED ARCHITECTURE ONLY – STRICTLY NON-DESTRUCTIVE (No Code Created)  
> **Target Subsystem:** `app/calibration/` within `busoft-yuvaraj/multi-camera-person-reid`  
> **Prerequisite:** Explicit User Review, Approval, and Provision of Survey Measurements  

---

## 1. Architectural Blueprint for `app/calibration/`

The proposed calibration subsystem is designed as an isolated, modular package residing under `app/calibration/`. It encapsulates all geometric transformations, coordinate projections, and metric validation routines without coupling directly to tracking internals.

```
app/
  calibration/
    __init__.py
    frame_extractor.py             # Phase 1: Interactive & automated keyframe capture
    point_annotator.py             # Phase 1: GUI/CLI ground-plane reference point digitizer
    homography_estimator.py        # Phase 2: DLT/RANSAC planar homography solver & refiner
    world_coordinate_mapper.py     # Phase 3: Forward/Inverse projection & multi-camera stitching
    calibration_validator.py       # Phase 4: Reprojection error audit & metric distance checker
    camera_calibration_manager.py  # Phase 5: High-level runtime interface for tracking pipeline
```

---

## 2. Phased Implementation Roadmap

---

### Phase 1: Reference Frame Extraction & Ground-Truth Digitization Tooling

#### Objective:
Provide robust, reproducible tooling to extract pristine vacant-floor keyframes from video streams and enable users to accurately digitize ground-truth pixel landmarks $(u, v)$ and assign corresponding metric world coordinates $(X_w, Y_w)$.

#### Proposed Modules:
- `app/calibration/frame_extractor.py`
- `app/calibration/point_annotator.py`

#### Required Inputs:
- Raw video feeds (`waiting_lobby.mp4`, `pantry.mp4`, `passage1.mp4`).
- Candidate reference points identified in [`REFERENCE_POINT_ANALYSIS.md`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/REFERENCE_POINT_ANALYSIS.md).
- Manual metric measurements from user (checklist in [`MANUAL_INPUT_CHECKLIST.md`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/MANUAL_INPUT_CHECKLIST.md)).

#### Dependencies:
- `opencv-python`, `numpy`, `pyyaml`.

#### Expected Outputs:
- Calibration point configuration file: `config/calibration_points.yaml` storing:
  ```yaml
  cameras:
    waiting_lobby:
      points:
        - id: "C1_P1"
          pixel: [282, 581]
          world_metric: [0.00, 1.20]  # meters
        - id: "C1_P5"
          pixel: [590, 650]
          world_metric: [2.10, 0.00]
  ```

#### Validation Criteria:
- Digitizer tool displays real-time zoom loupe ($4\times$) for sub-pixel crosshair alignment.
- Validates that no fewer than four non-collinear points are saved per camera view.

#### Risks & Limitations:
- Human error during clicking; mitigated by interactive zoom loupe and residual reprojection preview.

---

### Phase 2: Homography Estimation & Planar Metric Calibration Engine

#### Objective:
Compute optimal, robust planar homography matrices $\mathbf{H}_i$ mapping 2D pixel coordinates to metric world coordinates using normalized Direct Linear Transformation (DLT) with Levenberg-Marquardt non-linear refinement.

#### Proposed Modules:
- `app/calibration/homography_estimator.py`
- `app/calibration/calibration_validator.py`

#### Required Inputs:
- `config/calibration_points.yaml` (from Phase 1).

#### Dependencies:
- `opencv-python` (`cv2.findHomography`), `scipy.optimize`.

#### Expected Outputs:
- Calculated calibration matrices saved to `config/camera_homographies.yaml`:
  ```yaml
  homographies:
    waiting_lobby:
      H: [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]]
      reprojection_error_rms_px: 1.84
      metric_error_rms_m: 0.038
  ```

#### Validation Criteria:
- Root-Mean-Square (RMS) reprojection error must be $\le 3.0\text{ pixels}$.
- Metric distance validation: Distance between pairs of known test points must match ground truth within $\le 5.0\text{ cm}$.

#### Risks & Limitations:
- Matrix ill-conditioning caused by near-collinear points; mitigated by Hartley pre-normalization and condition number monitoring ($\kappa(\mathbf{H}) < 10^4$).

---

### Phase 3: World Coordinate Mapping & Cross-Camera Global Alignment

#### Objective:
Implement forward and inverse projection engines that map 2D pedestrian footpoints into a unified global facility coordinate frame, fusing CCTV 2 and CCTV 3 via direct visual tie-points, and registering CCTV 1 via the measured corridor offset.

#### Proposed Modules:
- `app/calibration/world_coordinate_mapper.py`

#### Required Inputs:
- `config/camera_homographies.yaml` (from Phase 2).
- Tie-point correspondences between CCTV 2 and CCTV 3.
- Measured metric translation vector $\mathbf{t}_{12} = [\Delta X_{12}, \Delta Y_{12}]^T$ and rotation angle $\theta_{12}$ for the blind corridor between CCTV 1 and CCTV 2.

#### Dependencies:
- `numpy`.

#### Expected Outputs:
- `WorldCoordinateMapper` class exposing clean, high-performance vectorized APIs:
  - `map_pixel_to_local_metric(camera_id, u, v) -> (x_m, y_m)`
  - `map_pixel_to_global_metric(camera_id, u, v) -> (X_g, Y_g)`
  - `map_global_to_pixel(camera_id, X_g, Y_g) -> (u, v)`

#### Validation Criteria:
- Co-visible stationary landmarks (e.g., tripod plant base) projected from CCTV 2 and CCTV 3 must align in global coordinates with Euclidean discrepancy $\le 0.15\text{ meters}$.

#### Risks & Limitations:
- Accumulation of alignment error across the blind corridor; mitigated by constraining $\mathbf{t}_{12}$ against architectural CAD drawings.

---

### Phase 4: Integration into Tracking & Association Subsystems

#### Objective:
Integrate `WorldCoordinateMapper` into existing tracking modules (`SpatialGate`, `TemporalGate`, `CandidateFilter`, and `AssociationManager`) as an additive validation layer.

#### Proposed Modules to Augment:
- [app/association/spatial_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/spatial_gate.py)
- [app/association/temporal_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/temporal_gate.py)
- [app/association/association_manager.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/association_manager.py)
- [app/journey/journey_manager.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/journey_manager.py)

#### Required Inputs:
- Verified `WorldCoordinateMapper` from Phase 3.
- Metric transition polygon definitions.

#### Dependencies:
- Existing tracking architecture.

#### Expected Outputs:
- Dynamic velocity filtering: Automatically discards cross-camera association candidates with implied velocities $v > 2.5\text{ m/s}$ or $v < 0.3\text{ m/s}$.
- Spatial distance penalty: Fuses exponential spatial distance weight $w_{\text{spatial}} = 0.20$ into `GlobalMatcher`.
- Real-world metric trajectory logging in `logs/journey.jsonl`.

#### Validation Criteria:
- Zero regression in local BoT-SORT track stability.
- False-positive cross-camera association rate reduced by $\ge 40\%$ on test footage containing subjects with identical clothing.

#### Risks & Limitations:
- Bounding-box footpoint jitter at far depth; mitigated by fusing YOLO-Pose ankle keypoints and Kalman smoothing.

---

### Phase 5: Pipeline Benchmarking, Visual Dashboard & Acceptance Testing

#### Objective:
Perform end-to-end multi-camera verification, produce side-by-side metric tracking dashboards, and validate accuracy across full test footage runs.

#### Proposed Modules:
- `app/calibration/camera_calibration_manager.py` (Runtime Coordinator)
- Top-Down 2D Metric Dashboard Visualizer (`tools/visualize_world_tracks.py`)

#### Required Inputs:
- Synchronized multi-camera test videos (`waiting_lobby.mp4`, `pantry.mp4`, `passage1.mp4`).

#### Expected Outputs:
- Comprehensive validation report detailing:
  - Global identity switch count before vs. after calibration.
  - Mean trajectory position variance.
  - Video render displaying live synchronized top-down 2D floor plan with animated subject trajectories.

#### Validation Criteria:
- Pipeline executes in real-time ($> 24\text{ fps}$ on CPU/GPU without thread contention).
- Seamless user acceptance of identity continuity across all transition boundaries.
