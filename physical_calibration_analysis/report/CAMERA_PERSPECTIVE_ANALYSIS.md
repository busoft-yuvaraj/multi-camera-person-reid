# Camera Perspective, Environmental & Ground-Plane Analysis

> **Target Repository:** `busoft-yuvaraj/multi-camera-person-reid` (`c:/Users/Agira/Desktop/cctv-tracker`)  
> **Input Video Feeds:** `cctv_samples/waiting_lobby.mp4`, `cctv_samples/pantry.mp4`, `cctv_samples/passage1.mp4`  
> **Raw NVR Archives Checked:** `cctv_samples/test_1/NVR_ch10_*.dav`, `ch3_*.dav`, `ch4_*.dav`  

---

## 1. Project Video Footage & Configuration Inventory

A systematic audit was conducted on all video assets, directory paths, existing configuration files, and video processing utilities in the project.

### 1.1 Video File Audit Table

| Property | CCTV 1 (Waiting Lobby) | CCTV 2 (Pantry) | CCTV 3 (Passage 1) |
|---|---|---|---|
| **Camera ID** | `waiting_lobby` | `pantry` | `passage` |
| **Zone Name** | `waiting_lobby` | `pantry` | `passage` |
| **Video Filename** | `cctv_samples/waiting_lobby.mp4` | `cctv_samples/pantry.mp4` | `cctv_samples/passage1.mp4` |
| **Raw NVR Archive** | `cctv_samples/test_1/NVR_ch4_extra1_20260918_132933.dav` | `cctv_samples/test_1/NVR_ch3_extra1_20260918_132932.dav` | `cctv_samples/test_1/NVR_ch10_extra1_20260918_132930.dav` |
| **Native Resolution** | $1280 \times 720$ px | $1280 \times 720$ px | $1280 \times 720$ px |
| **Processing Crop** | None (Full Frame $1280 \times 720$) | Configured crop `[0, 0, 720, 720]` in `app_config.yaml` | None (Full Frame $1280 \times 720$) |
| **Frame Rate (FPS)** | $24.88$ fps | $24.10$ fps | $23.91$ fps |
| **Duration** | $66.96$ seconds | $66.76$ seconds | $67.72$ seconds |
| **Total Frame Count** | $1666$ frames | $1609$ frames | $1619$ frames |
| **Start OSD Timestamp**| `2026-09-29 12:42:11` | `29/09/2026 12:42:06` | `29-09-2026 12:42:07` |
| **Camera Perspective** | High-angle elevated, facing North toward elevator bank | High-angle corner mount, facing Northwest across room | Shallow-angle axial mount, facing North down corridor |
| **Floor Visibility** | High in central corridor ($40\%$ of frame); partially occluded by planter and booth | Very High ($65\%$ of frame); wood parquet, tile, and carpet visible | High along length ($45\%$ of frame); narrow tunnel geometry |
| **Physical Overlap** | **No overlap** with Pantry or Passage (separated by blind corridor) | **Direct visual overlap** with Passage 1 at southern doorway | **Direct visual overlap** with Pantry looking north through doorway |

### 1.2 Existing Project Infrastructure Identified
- **Configuration Files:** [config/app_config.yaml](file:///c:/Users/Agira/Desktop/cctv-tracker/config/app_config.yaml) and [config/app_config_new.yaml](file:///c:/Users/Agira/Desktop/cctv-tracker/config/app_config_new.yaml) define camera resolutions, coordinate crops, monitored polygon ROIs, tripwires, topology adjacency matrices, and temporal transit constraints.
- **Video Ingestion Engine:** [app/multi_camera_processor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py) executes synchronized multi-stream ingestion with coordinate transforms for cropped views.
- **Event Detection Engines:** [app/events/roi_detector.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/roi_detector.py) performs OpenCV `pointPolygonTest` on footpoint coordinates $( (x_1+x_2)/2, y_2 )$, and [app/events/transition_detector.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/transition_detector.py) detects line crossings using signed 2D cross-products.
- **Gating Framework:** [app/association/spatial_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/spatial_gate.py) and [app/association/temporal_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/temporal_gate.py) filter association candidates based on graph topology and empirical time windows.

---

## 2. Camera-by-Camera Perspective & Ground-Plane Analysis

---

### 2.1 CCTV 1 – Waiting Lobby (`waiting_lobby.mp4`)

#### A. Camera Perspective & Optical Geometry
- **Viewing Direction:** Facing magnetic North toward the elevator lobby doors ("LIFT-1", "LIFT-2").
- **Mounting Position & Elevation:** Ceiling-mounted at approximately $2.8\text{ m} - 3.2\text{ m}$ height above the floor, near the southern boundary of the lobby. Downward tilt angle is approximately $25^\circ - 30^\circ$ from horizontal.
- **Perspective Distortion:** Moderate foreshortening. A standing person in the near foreground ($y \approx 650\text{ px}$) occupies approximately $450\text{ px}$ in height; the same person at the rear elevator doors ($y \approx 280\text{ px}$) occupies approximately $180\text{ px}$ in height (height ratio $\approx 2.5:1$).
- **Lens Distortion:** Moderate barrel distortion along the outer frame margins. Vertical glass mullions on the far left ($x < 150$) and far right ($x > 1150$) exhibit minor radial bowing.
- **Challenging Visibility Factors:**
  1. *Acoustic Glass Partitions:* Large floor-to-ceiling curved glass booth on the right ($x: 520 - 750$) and frosted honeycomb partition on the left ($x: 0 - 320$) produce severe specular reflections of moving subjects. A person walking in the center corridor appears as a ghost reflection inside the glass booth (observed clearly at frame 450).
  2. *Floor Specularity:* High-gloss polished grey porcelain tiles reflect overhead linear LED troffers, creating high-intensity specular blooms that can confound edge detectors.

#### B. Ground-Plane Visibility & Physical Landmarks
- **Visible Floor Boundaries:** The central walking corridor between $x = 319$ and $x = 708$ has completely unobstructed floor visibility from $y = 265$ (elevator door threshold) to $y = 720$ (bottom edge of frame).
- **Floor Corners & Seams:**
  - Standard square tile grid with crisp $2\text{ mm}$ grout joints visible throughout the central corridor.
  - Distinct longitudinal grout line at $x \approx 420\text{ px}$ extending from the foreground to the rear doors.
  - Transverse grout seams at $y \approx 410\text{ px}$, $y \approx 530\text{ px}$, and $y \approx 608\text{ px}$.
- **Fixed Environmental Objects at Ground Level ($Z = 0$):**
  - **Circular Planter 1 Base:** White cylindrical base anchored to the tile floor. Front rim touches floor at $(282, 581)$; rightmost rim touches floor at $(362, 515)$; rear rim at $(319, 472)$.
  - **Curved Glass Booth Base Track:** Aluminum floor track of the meeting booth. Tangent contact point at $(530, 560)$; foreground curve at $(590, 650)$.
  - **Elevator Lobby Door Threshold:** Transverse floor seam where lobby tiles terminate at the stainless steel lift threshold ($y = 265$, $x: 338 - 466$).

#### C. Calibration Suitability Rating: HIGH
- **Justification:** The central walking corridor is entirely planar, horizontally rigid, and contains well-distributed, non-collinear floor landmarks spanning the foreground, midground, and background. Glass reflections and furniture occlusions are isolated outside the primary monitored corridor.

---

### 2.2 CCTV 2 – Pantry (`pantry.mp4`)

#### A. Camera Perspective & Optical Geometry
- **Viewing Direction:** Mounted in the southeast corner of the pantry, looking northwest across the dining area toward the glass conference room wall and kitchen counter.
- **Mounting Position & Elevation:** Ceiling corner mount at approximately $2.6\text{ m} - 2.9\text{ m}$ height. Downward tilt angle is approximately $35^\circ - 40^\circ$.
- **Perspective Distortion:** Moderate-to-high oblique perspective. Excellent downward pitch provides strong ground separation between foreground and background objects.
- **Lens Distortion:** Mild barrel distortion noticeable on ceiling linear luminaires and perimeter cabinet edges.
- **Challenging Visibility Factors:**
  1. *Furniture Occlusion:* High bar-height dining table ($4$ bar stools with tall backs) occupies the left quadrant ($x: 100 - 460$, $y: 350 - 680$). Floor under the table is partially occluded by chair legs and table pedestals.
  2. *Sink Counter Zone:* Kitchen counter on far left ($x < 100$) is elevated and non-planar.
  3. *Glass Partitions:* North conference room wall reflects lighting and movement.

#### B. Ground-Plane Visibility & Physical Landmarks
- **Visible Floor Boundaries:** Over $65\%$ of the total canvas exposes the rigid ground plane.
- **Multi-Material Ground Boundaries (Extremely High Calibration Value):**
  - **Carpet-Wood-Tile Junction (`C2_P1`):** Point $(530, 490)$ where the dark grey entrance carpet, warm wooden parquet section, and grey ceramic tiles meet. This is a crisp, sub-millimeter physical vertex on the floor plane.
  - **Wood Parquet Diagonal Border:** Straight transition line running from $(49, 532)$ down to $(530, 490)$.
  - **Carpet Mat Transverse Threshold:** Straight edge running from $(530, 490)$ to $(860, 650)$, separating the pantry tile floor from the carpeted entrance of Passage 1.
- **Fixed Environmental Objects at Ground Level ($Z = 0$):**
  - **Circular Planter 2 Base:** White cylindrical base in the center of the room. Front floor contact at $(580, 595)$; left at $(490, 535)$; right at $(675, 535)$; rear at $(580, 465)$.
  - **High Bar Table Legs:** Four square metal legs firmly planted on the parquet floor. Front-right leg at $(465, 675)$; rear-right leg at $(535, 580)$.
  - **Tripod Potted Plant Stand:** Three wooden legs resting on tiles in the foreground right. Left leg at $(748, 650)$; center leg at $(765, 660)$; right leg at $(790, 640)$.

#### C. Calibration Suitability Rating: HIGH (Highest in the Project)
- **Justification:** CCTV 2 contains the richest collection of ground-plane geometric primitives across the facility. The sharp multi-material floor intersections (wood/tile/carpet) provide ideal ground-truth reference lines that can be directly surveyed.

---

### 2.3 CCTV 3 – Passage 1 (`passage1.mp4`)

#### A. Camera Perspective & Optical Geometry
- **Viewing Direction:** Axial alignment facing North down the central axis of "Passage 1" toward the southern entrance of the Pantry.
- **Mounting Position & Elevation:** Ceiling-mounted at approximately $2.5\text{ m} - 2.7\text{ m}$ height at the south end of the hallway. Downward tilt angle is shallow ($\approx 12^\circ - 18^\circ$).
- **Perspective Distortion: SEVERE FORESHORTENING (Tunnel Geometry).**
  - A person standing at the near hallway threshold ($y \approx 700\text{ px}$) occupies $\approx 550\text{ px}$ in height.
  - A person standing at the far end near the pantry threshold ($y \approx 480\text{ px}$) occupies $\approx 160\text{ px}$ in height.
  - Depth foreshortening ratio exceeds $7:1$ from foreground to background. A $1\text{ meter}$ distance on the ground plane corresponds to $\approx 180\text{ px}$ at $y = 650$, but drops to only $\approx 18\text{ px}$ at $y = 480$.
- **Lens Distortion:** Mild wide-angle barrel distortion along the outer vertical door frames.
- **Challenging Visibility Factors:**
  1. *Extreme Aspect Ratio:* Corridor width is narrow ($\approx 1.5\text{ m} - 1.8\text{ m}$), while length is long ($\approx 12\text{ m} - 15\text{ m}$).
  2. *Low Ground Texture:* The entire corridor floor is covered by uniform, low-contrast dark acoustic carpet without distinct grid lines.
  3. *Hanging Obstacles:* Suspended illuminated "FIRE EXIT" sign on the right ceiling edge ($x \approx 800$, $y \approx 500 - 800$) partially occludes upper body heights of tall subjects, though floor remains visible.

#### B. Ground-Plane Visibility & Physical Landmarks
- **Visible Floor Boundaries:** The full length of the carpet corridor is visible between the left cabin partitions ($x \approx 430 - 500$) and the right wall baseboard ($x \approx 635 - 730$).
- **Baseboard & Architectural Intersections ($Z = 0$):**
  - Near-left door jamb base (Cabin 3G) at $(430, 700)$.
  - Near-right fire exit wall baseboard at $(635, 700)$.
  - Mid-hallway door jamb base (Cabin 3F) at $(460, 580)$.
  - Mid-hallway right service door base at $(675, 580)$.
- **Doorway Threshold & Shared Overlap Region ($Z = 0$):**
  - Hallway-to-pantry floor transition line running from $(575, 485)$ to $(740, 485)$.
  - Through the doorway opening, CCTV 3 directly observes landmarks inside the Pantry:
    * Base of the Pantry tripod plant stand at $(915, 455)$.
    * Base of the Pantry circular planter rim at $(635, 430)$.

#### C. Calibration Suitability Rating: MEDIUM-HIGH
- **Justification:** CCTV 3 is highly suitable for 1D longitudinal tracking and direct tie-point registration with CCTV 2. However, due to severe depth foreshortening, lateral 2D homography estimation will suffer high variance at depths past $y < 460\text{ px}$. Uncertainty ellipses must be modeled along the depth axis.

---

## 3. Representative Frame Extraction Catalog

Twelve representative frames were sampled per camera (36 frames total) and preserved under `physical_calibration_analysis/extracted_frames/cctv1/`, `cctv2/`, and `cctv3/`. Contact sheets summarizing each camera feed are stored under `physical_calibration_analysis/contact_sheets/`.

### 3.1 CCTV 1 (Waiting Lobby) Frame Catalog

| Frame # | Timestamp (s) | Scene Content & Calibration Utility |
|---|---|---|
| `frame_0000` | $0.00\text{ s}$ | **Optimal Calibration Baseline:** Completely vacant floor; full view of tile seams, planter base, curved booth base track, and elevator door threshold without human occlusion. |
| `frame_0150` | $6.03\text{ s}$ | Baseline lighting verification; identical static geometry confirming absence of camera mount sway or vibration. |
| `frame_0250` | $10.05\text{ s}$ | Subject entering elevator lobby doors in far background ($y \approx 280$); validates maximum visible depth limit. |
| `frame_0350` | $14.07\text{ s}$ | Subject in white shirt entering foreground right; confirms foot-ground contact resolution at near distances. |
| `frame_0450` | $18.09\text{ s}$ | Two subjects walking down central corridor; illustrates ghost reflection on curved glass booth on the right. |
| `frame_0550` | $22.11\text{ s}$ | Subjects exiting through rear door; shows foot-ground contact at transition threshold. |
| `frame_0750` | $30.14\text{ s}$ | Vacant floor recovery; ideal frame for background modeling and static landmark coordinate verification. |
| `frame_0900` | $36.17\text{ s}$ | Subjects entering from rear lift area; demonstrates foot-point tracking under high ambient illumination. |
| `frame_1050` | $42.20\text{ s}$ | Single subject walking in center lane; foot-ground contact points align directly with longitudinal tile grout line $x = 420$. |
| `frame_1200` | $48.23\text{ s}$ | Clean midground floor; verifies stability of tile grid landmarks. |
| `frame_1400` | $56.27\text{ s}$ | Clear view of left planter base and right glass booth curve. |
| `frame_1600` | $64.31\text{ s}$ | Final scene state; confirms zero drift in camera mounting position over the recording period. |

### 3.2 CCTV 2 (Pantry) Frame Catalog

| Frame # | Timestamp (s) | Scene Content & Calibration Utility |
|---|---|---|
| `frame_0000` | $0.00\text{ s}$ | **Optimal Calibration Baseline:** Full view of parquet-tile-carpet junction, bar table legs, circular planter base, and tripod plant legs. |
| `frame_0120` | $4.98\text{ s}$ | Staff member in uniform at kitchen counter; establishes spatial boundary of kitchen sink non-ground region. |
| `frame_0240` | $9.96\text{ s}$ | Staff member moving near left boundary; confirms clearance of monitored corridor. |
| `frame_0360` | $14.94\text{ s}$ | Unobstructed view of central circular planter and rear conference room doors. |
| `frame_0520` | $21.58\text{ s}$ | Subject walking past central planter; illustrates footpoint elevation relative to planter rim. |
| `frame_0600` | $24.90\text{ s}$ | Subject traversing rear hallway; validates ground-plane continuity behind planter. |
| `frame_0750` | $31.12\text{ s}$ | Vacant room state; provides pristine reference for manual landmark digitization. |
| `frame_0950` | $39.42\text{ s}$ | Multiple subjects near rear glass wall; evaluates occlusion handling against fixed furniture. |
| `frame_1050` | $43.57\text{ s}$ | Subjects transitioning toward Passage 1 threshold; critical frame for cross-camera handoff calibration. |
| `frame_1150` | $47.72\text{ s}$ | Subject stepping across carpet mat into Passage 1; direct visual tie-in with CCTV 3 frame 1050. |
| `frame_1350` | $56.02\text{ s}$ | Static floor validation; shows stability of tripod plant legs on tile floor. |
| `frame_1550` | $64.32\text{ s}$ | Final clean floor view; confirms constant lighting and camera pose stability. |

### 3.3 CCTV 3 (Passage 1) Frame Catalog

| Frame # | Timestamp (s) | Scene Content & Calibration Utility |
|---|---|---|
| `frame_0000` | $0.00\text{ s}$ | **Optimal Calibration Baseline:** Clean view down corridor; clear baseboard lines on left and right; direct line of sight into Pantry. |
| `frame_0150` | $6.27\text{ s}$ | Static baseline frame; confirms hallway baseboard corner coordinates. |
| `frame_0300` | $12.55\text{ s}$ | Clear view of far-end pantry threshold; tripod plant and circular planter clearly visible in background. |
| `frame_0450` | $18.82\text{ s}$ | Evaluates lighting consistency in corridor ceiling LED troffers. |
| `frame_0600` | $25.10\text{ s}$ | Verifies door frame positions along left cabin row (Cabin 3G, 3F). |
| `frame_0750` | $31.37\text{ s}$ | Vacant corridor; ideal for measuring corridor width pixel ratios across depth. |
| `frame_0900` | $37.64\text{ s}$ | Static scene verification; confirms stability of fire exit hanging sign. |
| `frame_1050` | $43.91\text{ s}$ | Subjects appearing at far pantry doorway; directly corresponds with CCTV 2 frame 1050/1150 handoff. |
| `frame_1200` | $50.19\text{ s}$ | Two subjects walking down corridor toward camera; provides clear foot-carpet contact points across mid-depth. |
| `frame_1350` | $56.46\text{ s}$ | Subjects exiting foreground south; demonstrates footpoint tracking under extreme near-field projection. |
| `frame_1450` | $60.64\text{ s}$ | Corridor clear; provides clean ground-plane verification after subject transit. |
| `frame_1580` | $66.08\text{ s}$ | Final frame; confirms zero camera mount movement or focal length change. |
