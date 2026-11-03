# Physical Camera Calibration Assessment & Feasibility Report
## Multi-Camera Person Re-Identification & Tracking System

> **Document Classification:** Engineering Assessment, Feasibility Study & Implementation Blueprint  
> **Facility Location:** 3rd Floor Corporate Office (Waiting Lobby, Pantry, Passage 1)  
> **Target System:** Multi-Camera Person Tracking & ReID Pipeline (`cctv-tracker`)  
> **Analysis Phase:** STRICTLY NON-DESTRUCTIVE INVESTIGATION & PLANNING (No Code Modified)  
> **Author:** Antigravity AI Engineering Team  
> **Date:** November 2026  

---

## 1. Executive Summary

This report delivers a comprehensive technical feasibility assessment and architectural planning analysis for integrating **Homography-based Physical Ground-Plane Calibration** into the existing multi-camera person tracking and re-identification system.

The investigated facility comprises three ceiling-mounted fixed CCTV cameras:
1. **CCTV 1 (`waiting_lobby`):** Elevated perspective facing north down the primary entrance lobby toward elevator doors ("LIFT-1", "LIFT-2").
2. **CCTV 2 (`pantry`):** High-angle oblique perspective facing northwest across the office pantry and dining area toward an acoustically partitioned conference room.
3. **CCTV 3 (`passage`):** Long-throw, shallow-angle axial perspective looking north down a narrow corridor ("Passage 1") directly through the southern doorway of the pantry.

### Key Technical Verdicts:
1. **Single-Camera Ground-Plane Homography is Highly Feasible for CCTV 1 and CCTV 2, and Feasible with Depth Foreshortening Constraints for CCTV 3.**
   - The floors across all three environments are rigid, flat, horizontal planes (commercial ceramic tile in Lobby and Pantry, wooden parquet under the pantry dining table, and commercial acoustic carpet in Passage 1).
   - Sufficient non-collinear physical landmarks lying directly on the floor plane ($Z = 0$) exist in all cameras ($\ge 10$ candidates per camera).
2. **Cross-Camera Metric Alignment into a Unified World Coordinate System is Achievable via a Hybrid Homography-CAD Stitching Approach:**
   - **CCTV 2 (Pantry) and CCTV 3 (Passage 1)** share a **direct physical overlap** at the southern pantry entrance. Shared landmarks (such as the base of the white tripod planter, circular planter rim, and carpet-to-tile floor transition) permit direct homographic tie-point registration.
   - **CCTV 1 (Waiting Lobby) and CCTV 2 (Pantry)** are separated by an enclosed connecting corridor that forms an unmonitored **blind transit zone** ($\approx 2.5\text{ m} - 4.0\text{ m}$ transit distance; $2 - 6\text{ s}$ transit duration). Because their fields of view do not overlap, homographies cannot be co-registered purely from image correspondences; **manual metric ground measurements or architectural CAD floor plan anchors are mandatory** to couple CCTV 1 to the global world frame.
   - **CCTV 1 (Waiting Lobby) and CCTV 3 (Passage 1)** do not overlap and are separated by the Pantry.
3. **Severe Visual Ambiguity Detected and Resolved:**
   - Visual inspection confirmed that both the Waiting Lobby and the Pantry contain circular white planters with large broad-leafed indoor plants. **These are two distinct physical objects** situated in separate architectural zones. Any automated calibration algorithm assuming them to be the same landmark would induce catastrophic coordinate collapse.
4. **Impact on Multi-Camera Architecture:**
   - Physical ground-plane calibration will transform camera-local 2D bounding box footpoints into metric Cartesian coordinates $(X_w, Y_w)$ in meters.
   - This provides deterministic physical velocity gating ($v = \Delta d / \Delta t$) in `TemporalCandidateManager`, prevents perspective-induced false crossings in `SpatialROIGate`, introduces spatial distance penalties in `CameraAssociationManager`, and significantly disambiguates cross-camera identity switches between persons wearing similar attire.
5. **Strict Non-Destructive Adherence:**
   - No tracking code, configuration files, thresholds, or models have been altered. This assessment establishes the rigorous mathematical foundation and data requirements needed before implementation commences upon user approval.

---

## 2. Synthesis of Primary Objectives

| Objective | Feasibility & Technical Finding | Key Dependency / Constraint |
|---|---|---|
| **1. Feasibility using available footage** | **FEASIBLE.** Footage provides static camera mounts, high resolution ($1280 \times 720$), clean lighting, and rigid planar floors. | Dependent on obtaining real-world metric anchor measurements for scale. |
| **2. Physical information extractable from videos** | Perspective vanishing lines, tile aspect ratios, floor material transition lines, door threshold alignments, and relative transit timelines. | Absolute metric scale ($s$) and absolute inter-camera baseline distances cannot be extracted from video alone. |
| **3. Suitable ground-plane regions** | **Lobby:** Central tile corridor ($319 \le x \le 708$).<br>**Pantry:** Entire central tile floor, wood parquet section, and entrance mat ($0 \le x \le 860$).<br>**Passage:** Central carpet corridor between door frames ($430 \le x \le 740$). | Floor regions occluded by tables, chairs, planters, and glass partitions must be gated out. |
| **4. Reference points for homography** | Identified 12 candidates in CCTV 1, 15 candidates in CCTV 2, and 10 candidates in CCTV 3. All lie strictly on the floor ($Z = 0$). | High bar table surface and planter top rims must NOT be used as ground points. |
| **5. Common coordinate system mapping** | **FEASIBLE via 2-stage stitching:** CCTV 2 & 3 align via visual overlap; CCTV 1 aligns via CAD/manual corridor offset. | Requires 4 ground tie-points between CCTV 2 and 3, plus metric corridor measurement between CCTV 1 and 2. |
| **6. Additional manual measurements needed** | Standard floor tile width/length, bar table dimensions, circular planter diameter, and distance along the Lobby-Pantry connecting hallway. | See [`MANUAL_INPUT_CHECKLIST.md`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/MANUAL_INPUT_CHECKLIST.md) for full protocol. |
| **7. Tracking architecture enhancements** | Converts ad-hoc pixel ROIs into world polygons; replaces empirical min/max temporal buffers with physics-based speed bounds; adds spatial distance to ReID. | Bounding-box footpoint detection error ($\pm 15\text{ px}$) propagates non-linearly at far depth. |
| **8. Proposed implementation approach** | A 5-phase modular architecture under `app/calibration/` including interactive annotator, homography engine, and spatial world mapper. | Implementation strictly paused pending user review and provision of metric measurements. |

---

## 3. High-Level Camera Suitability Summary

```
+----------------------------------------------------------------------------------------------------+
|                                    FACILITY GROUND-PLANE TOPOLOGY                                  |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|   +--------------------------+         [Blind Corridor]         +------------------------------+   |
|   |         CCTV 2           | <==============================> |            CCTV 1            |   |
|   |         PANTRY           |     dt: 2 - 6s, dist: ~3.5m      |        WAITING LOBBY         |   |
|   | Suitability: HIGH        |     (Manual Anchor Required)     | Suitability: HIGH            |   |
|   +--------------------------+                                  +------------------------------+   |
|                ||                                                                                  |
|                || [Direct Visual Overlap]                                                          |
|                || Tie-Points: Tripod Plant, Planter Base, Doorway                                  |
|                \/                                                                                  |
|   +--------------------------+                                                                     |
|   |         CCTV 3           |                                                                     |
|   |        PASSAGE 1         |                                                                     |
|   | Suitability: MEDIUM-HIGH |                                                                     |
|   +--------------------------+                                                                     |
|                ||                                                                                  |
|                \/                                                                                  |
|       [Open Workspace]                                                                             |
+----------------------------------------------------------------------------------------------------+
```

- **CCTV 1 (`waiting_lobby`): HIGH SUITABILITY.** High camera mounting angle, unobstructed central tile corridor, sharp tile seams, clear door thresholds.
- **CCTV 2 (`pantry`): HIGH SUITABILITY.** Richest calibration environment in the facility. Multi-material floor boundaries (wood parquet, tile grid, carpet mat) and prominent floor-level circular features.
- **CCTV 3 (`passage`): MEDIUM-HIGH SUITABILITY.** Excellent longitudinal coverage; direct visual line-of-sight into Pantry. High foreshortening at far depths ($y < 450\text{ px}$) necessitates non-linear uncertainty weighting.

---

## 4. Structure of Detailed Report Deliverables

To provide complete technical clarity, this assessment is broken down into modular technical reports:

1. [**`CAMERA_PERSPECTIVE_ANALYSIS.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/CAMERA_PERSPECTIVE_ANALYSIS.md): Footage inventory, video metadata, lens distortion, viewing geometries, and representative frame catalog.
2. [**`REFERENCE_POINT_ANALYSIS.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/REFERENCE_POINT_ANALYSIS.md): Comprehensive inventory of candidate ground-plane landmarks, pixel coordinates, ground validity criteria, and confidence classifications.
3. [**`CAMERA_OVERLAP_ANALYSIS.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/CAMERA_OVERLAP_ANALYSIS.md): Physical spatial layout, overlap zones, blind corridors, synchronized transit verification, and coverage diagram analysis.
4. [**`HOMOGRAPHY_FEASIBILITY.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/HOMOGRAPHY_FEASIBILITY.md): Mathematical formulations, planar homography feasibility per camera, relative vs metric calibration trade-offs, and cross-camera registration proofs.
5. [**`TRACKING_ARCHITECTURE_IMPACT.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/TRACKING_ARCHITECTURE_IMPACT.md): Architectural analysis detailing how spatial calibration augments `SpatialROIGate`, `TemporalCandidateManager`, `CameraAssociationManager`, and Global ID assignment without altering existing ReID/PAR/Pose models.
6. [**`IMPLEMENTATION_ROADMAP.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/IMPLEMENTATION_ROADMAP.md): A 5-phase engineering implementation roadmap detailing proposed modules (`app/calibration/`), validation gates, dependencies, and risk mitigation strategies.
7. [**`MANUAL_INPUT_CHECKLIST.md`**](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/MANUAL_INPUT_CHECKLIST.md): Actionable, prioritized physical measurement checklist required from on-site facility personnel before implementation.

### Accompanying Visual Artifacts:
- **Representative Frames:** `physical_calibration_analysis/extracted_frames/` (`cctv1/`, `cctv2/`, `cctv3/`)
- **Contact Sheets:** `physical_calibration_analysis/contact_sheets/` (`cctv1_contact_sheet.jpg`, `cctv2_contact_sheet.jpg`, `cctv3_contact_sheet.jpg`)
- **Annotated Reference Points:** `physical_calibration_analysis/annotated_reference_points/` (`cctv1_annotated_reference_points.jpg`, `cctv2_annotated_reference_points.jpg`, `cctv3_annotated_reference_points.jpg`)
- **Camera Coverage Diagram:** `physical_calibration_analysis/camera_coverage_diagram/camera_coverage_diagram.png`

---

## 5. Final Recommendation & Next Steps

1. **Do not begin coding calibration algorithms yet.** Proceeding without real-world metric measurements would limit the system to arbitrary coordinate spaces, preventing metric velocity filtering.
2. **Review Candidate Reference Points:** Review the annotated images in `physical_calibration_analysis/annotated_reference_points/` to confirm that the chosen physical landmarks match facility maintenance records.
3. **Execute Physical Measurements:** Use a laser rangefinder or standard tape measure to record the items listed in [`MANUAL_INPUT_CHECKLIST.md`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/MANUAL_INPUT_CHECKLIST.md).
4. **Authorize Phase 1 Implementation:** Upon receipt of manual measurements, authorization can be granted to implement the dedicated calibration module `app/calibration/`.
