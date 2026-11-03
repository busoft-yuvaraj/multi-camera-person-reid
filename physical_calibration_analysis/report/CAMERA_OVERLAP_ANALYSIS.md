# Camera Overlap, Physical Topology & Spatial Layout Analysis

> **Document Type:** Spatial Layout Analysis, Inter-Camera Coverage Audit & Boundary Mapping  
> **Facility Floor:** 3rd Floor Corporate Suite  
> **Cameras Analyzed:** CCTV 1 (Waiting Lobby), CCTV 2 (Pantry), CCTV 3 (Passage 1)  
> **Visual Reference Artifact:** [`camera_coverage_diagram/camera_coverage_diagram.png`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/camera_coverage_diagram/camera_coverage_diagram.png)  

---

## 1. Physical Spatial Layout Overview

The audited 3rd floor environment comprises four interconnected functional zones arranged along an East-West and North-South architectural axis:
1. **Waiting Lobby (East Zone):** Primary entrance lobby accessed via an elevator bank (LIFT 1, LIFT 2) on its north wall. Contains an enclosed glass acoustic booth along its east boundary and an exit corridor opening on its west wall.
2. **Connecting Corridor (Blind Transit Zone):** An enclosed, unmonitored transition hallway ($\approx 2.5\text{ m} - 4.0\text{ m}$ long) connecting the west wall of the Waiting Lobby to the east entrance of the Pantry.
3. **Pantry (Central Zone):** An open dining and refreshment area with a kitchenette on the west, an enclosed glass conference room on the north, an entrance from the lobby corridor on the east, and an open doorway leading south into Passage 1.
4. **Passage 1 (South Corridor):** A long, narrow carpeted hallway extending south from the Pantry doorway. Private executive cabins (Cabin 3G, Cabin 3F, Cabin 3E) border the east side, while utility doors and a Fire Exit border the west side. At its southern termination, Passage 1 opens into the main open-plan workspace.

```
+----------------------------------------------------------------------------------------------------+
|                                    ARCHITECTURAL ZONE CONNECTIVITY                                 |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|    [North Conference Room]                                         [Elevator Bank: Lift 1, Lift 2] |
|              ||                                                                  ||                |
|    +--------------------+       [Blind Corridor]                       +--------------------+      |
|    |    CCTV 2: PANTRY  | <==========================================> |   CCTV 1: LOBBY    |      |
|    +--------------------+            (2.5m - 4.0m)                     +--------------------+      |
|              ||                                                                                    |
|         [Doorway Overlap]                                                                          |
|              ||                                                                                    |
|    +--------------------+                                                                          |
|    | CCTV 3: PASSAGE 1  |                                                                          |
|    +--------------------+                                                                          |
|              ||                                                                                    |
|              \/                                                                                    |
|       [Open Workspace]                                                                             |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Pairwise Camera Relationship & Overlap Analysis

---

### 2.1 Waiting Lobby (CCTV 1) $\longleftrightarrow$ Pantry (CCTV 2)

- **Physical Relationship:** Adjacent architectural rooms separated by an intervening corridor.
- **Direct Visual Overlap:** **NONE ($0\%$ Overlap).**
  - CCTV 1 faces north and cannot see through the west corridor opening.
  - CCTV 2 faces northwest and cannot see past its eastern door jamb.
- **Shared Visible Landmarks:** **NONE.**
  - As established in [`REFERENCE_POINT_ANALYSIS.md`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/report/REFERENCE_POINT_ANALYSIS.md), the circular planter in the Lobby and the circular planter in the Pantry are separate physical objects.
- **Simultaneous Visibility:** **IMPOSSIBLE.** A subject cannot be simultaneously visible in CCTV 1 and CCTV 2 because solid drywall separates the fields of view.
- **Transition Handoff & Blind Zone:**
  - Configured tripwire `lobby_yellow` ($x: 338 - 357, y: 52 - 546$) marks departure from CCTV 1.
  - Configured tripwire `pantry_yellow` ($x: 523 - 526, y: 152 - 417$) marks arrival in CCTV 2.
  - **Blind Transit Corridor:** When walking from Lobby to Pantry, a person disappears from CCTV 1, spends approximately $2.0 - 5.5\text{ seconds}$ inside the unmonitored blind corridor, and then emerges into CCTV 2.
- **Ground-Plane Alignment Feasibility:**
  - **Cannot be aligned purely from image correspondences.**
  - **Alignment Method:** Requires manual metric offset $[\Delta X_{12}, \Delta Y_{12}]$ and relative room orientation angle $\theta_{12}$ derived from architectural CAD floor plans or laser rangefinder survey.

---

### 2.2 Pantry (CCTV 2) $\longleftrightarrow$ Passage 1 (CCTV 3)

- **Physical Relationship:** Directly contiguous along the southern doorway of the Pantry.
- **Direct Visual Overlap:** **YES ($\approx 10\% - 15\%$ Overlap).**
  - CCTV 3 is positioned at the southern end of Passage 1 looking directly north through the open doorway into the southern half of the Pantry.
  - CCTV 2 is mounted high on the southeast ceiling of the Pantry looking northwest, capturing the foreground floor leading directly into the Passage 1 doorway.
- **Shared Visible Landmarks:**
  1. *Tripod Plant Stand Base:* Leg floor contact points (`C2_P10`/`C2_P11`/`C2_P12` in CCTV 2) are directly visible in CCTV 3 at $(915, 455)$ (`C3_P9`).
  2. *Circular Planter 2 Base:* The white cylindrical planter ring in CCTV 2 is partially visible through the doorway opening in CCTV 3 at $(635, 430)$ (`C3_P10`).
  3. *Doorway Floor Transition Seam:* The transverse threshold line where dark carpet meets wood parquet and ceramic tile is co-visible in both cameras:
     - In CCTV 2: Lines up with `pantry_green` ($x: 65 - 530, y: 502 - 490$).
     - In CCTV 3: Lines up with `passage_green` ($x: 576 - 743, y: 386$).
- **Simultaneous Visibility:** **YES.** A person stepping across the doorway threshold is simultaneously visible in CCTV 2 (stepping out of foreground) and CCTV 3 (stepping into far background).
  - *Observed in Footage:* At synchronized time `12:42:55` (CCTV 2 frame 1150, CCTV 3 frame 1050), a subject in a light shirt is simultaneously detected in both feeds.
- **Ground-Plane Alignment Feasibility:**
  - **Direct Image-Based Homography Registration is FEASIBLE.**
  - The shared floor landmarks and common doorway threshold permit calculation of a relative ground-plane homography matrix $\mathbf{H}_{3 \to 2}$ that rigidly maps CCTV 3 ground coordinates directly onto CCTV 2 ground coordinates.

---

### 2.3 Waiting Lobby (CCTV 1) $\longleftrightarrow$ Passage 1 (CCTV 3)

- **Physical Relationship:** Completely separated rooms without direct spatial adjacency.
- **Direct Visual Overlap:** **NONE ($0\%$ Overlap).**
- **Transit Path:** Any subject moving between Waiting Lobby and Passage 1 MUST traverse the Pantry (or bypass through an unmonitored back corridor).
- **Temporal Constraint:** Direct transits between Lobby and Passage require traversing two transition boundaries:
  $$\text{Lobby} \xrightarrow{\text{lobby\_yellow}} \text{Blind Corridor} \xrightarrow{\text{pantry\_yellow}} \text{Pantry} \xrightarrow{\text{pantry\_green}} \text{Passage 1}$$
  Minimum physically plausible transit time: $t_{\text{min}} \approx 8.0\text{ seconds}$ at normal walking speed ($1.2\text{ m/s}$). Any re-identification match with $\Delta t < 4\text{ s}$ is physically impossible and should be rejected.

---

## 3. Preliminary Camera Coverage Diagram Analysis

The visual layout generated in [`camera_coverage_diagram/camera_coverage_diagram.png`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/camera_coverage_diagram/camera_coverage_diagram.png) provides the structural blueprint for physical tracking:

```
        3rd Floor Multi-Camera Coverage & Field of View Topology
        =========================================================

                 North Conference Room (Glass Wall)
    +--------------------------------------------------------+
    |                                                        |
    |      +------------+                                    |
    |      |  Kitchen   |       (Pantry Planter)             |
    |      |  Counter   |             (O)                    |
    |      +------------+                                    |
    |                                            [pantry_y]  |
    |      [Bar Table]                        +==============+  <--- Blind Corridor (2.5m - 4.0m)
    |      +---------+                        |              |        No Camera Coverage
    |      |   Wood  |                        |  BLIND ZONE  |        Transit Time: 2 - 6s
    |      | Parquet |                        |              |
    |      +---------+                        +==============+  <--- [lobby_y]
    |                                            [Lobby Door]|
    |                     CCTV 2 [FOV Cone]                  |   +--------------------------+
    |                 \                      /               |   |                          |
    |                  \                    /                |   |        (O) Planter 1     |
    |                   \  Shared Overlap  /                 |   |                          |
    |  +=================+================+                  |   |  CCTV 1 [FOV Cone]       |
    |  |  pantry_green   |  passage_green |                  |   |    \                /    |
    |  +=================+================+                  |   |     \              /     |
    |          |                                             |   |      \            /      |
    |          |                                             |   |       \          /       |
    |          |                                             |   |        \ Meeting/        |
    |  Cabin 3G|   PASSAGE 1                                 |   |         \ Booth/         |
    |  Cabin 3F|   (Carpet Corridor)       Fire Exit         |   |          \    /          |
    |  Cabin 3E|                                             |   |           \  /           |
    |          |   CCTV 3 [FOV Cone]                         |   |          (CCTV 1)        |
    |          |       \           /                         |   +--------------------------+
    |          |        \         /                          |
    |          |         \       /                           |
    |          |          (CCTV 3)                           |
    +----------+---------------------------------------------+
                            |
                            \/
                    [Open Workspace]
```

### Key Takeaways from the Coverage Diagram:
1. **Directly Observed Overlap:**
   - Exists exclusively between CCTV 2 and CCTV 3 at the Southern doorway.
   - Ground tie-points can be surveyed directly from video frames.
2. **Inferred Spatial Relationships:**
   - CCTV 1 and CCTV 2 are collinear along an East-West hallway.
   - The walking trajectory between `lobby_yellow` and `pantry_yellow` is essentially linear.
3. **Relationships Requiring Manual On-Site Confirmation:**
   - The exact physical distance between `lobby_yellow` and `pantry_yellow` ($L_{\text{blind}}$).
   - The angular deviation between the Lobby corridor axis and the Pantry floor grid axis ($\theta_{\text{offset}}$).
   - Floor plan CAD drawings or on-site laser distance measurements are required to bridge this unmonitored gap.

---

## 4. Rigorous Distinction of Overlap Evidence

To prevent invalid assumptions, all inter-camera relationships are classified into three strict evidentiary tiers:

| Camera Pair | Evidentiary Tier | Evidence Basis | Status & Action |
|---|---|---|---|
| **Pantry (CCTV 2) & Passage 1 (CCTV 3)** | **Directly Observed Overlap** | Co-visible stationary landmarks (tripod plant, planter base, doorway threshold); simultaneous subject visibility at $t = 12:42:55$. | **Verified.** Can proceed to direct tie-point registration. |
| **Lobby (CCTV 1) & Pantry (CCTV 2)** | **Inferred Spatial Relationship** | Observed subject transitions across timestamps ($12:42:25 \to 12:42:30$); architectural alignment of east-west doors. | **Requires Manual Confirmation.** Cannot establish world coordinate scale without measuring blind corridor length. |
| **Lobby (CCTV 1) & Passage 1 (CCTV 3)** | **Disjoint (No Direct Overlap)** | Separated by intermediate room; zero shared field of view; transit requires traversing Pantry. | **Verified Disjoint.** Gated strictly through transit topology. |

> **Critical Caveat:** Two cameras cannot be calibrated into a common coordinate frame merely because individuals wearing similar clothing appear in both feeds. Visual ReID appearance establishes identity candidates, but only physical metric survey or direct visual tie-points can establish coordinate transformations.
