# On-Site Physical Survey & Manual Measurement Checklist

> **Document Type:** On-Site Facility Measurement Checklist, Data Collection Protocol & Contingency Fallback  
> **Target Environment:** 3rd Floor Corporate Suite (Waiting Lobby, Pantry, Passage 1)  
> **Audience:** Facility Administrator, System Operator, Computer Vision Engineers  

---

## 1. Overview of Missing Physical Information

A monocular camera feed provides projective image coordinates $(u, v)$, but inherently lacks absolute physical scale. While geometric constraints (such as vanishing lines and parallel edges) provide relative affine proportions, **metric ground-plane homography requires real-world dimensional measurements that cannot be extracted from 2D CCTV footage alone**.

To establish an accurate, metric world coordinate system in meters, the following checklist specifies every missing physical property, why it is necessary, how to measure it on-site, and the fallback behavior if it remains unavailable.

---

## 2. Priority 1: High-Precision Metric Ground Anchors (MANDATORY for Metric System)

---

### Item 1.1: Standard Commercial Floor Tile Dimensions
- **Applicable Cameras:** CCTV 1 (Waiting Lobby), CCTV 2 (Pantry).
- **Physical Landmark:** Square polished grey porcelain tiles visible across the lobby and pantry floor.
- **Specific Measurements Needed:**
  1. Tile Width ($W_{\text{tile}}$): e.g., $600\text{ mm}$ ($0.60\text{ m}$) or $800\text{ mm}$ ($0.80\text{ m}$).
  2. Tile Length ($L_{\text{tile}}$): e.g., $600\text{ mm}$ ($0.60\text{ m}$) or $800\text{ mm}$ ($0.80\text{ m}$).
  3. Grout Joint Width: e.g., $2\text{ mm} - 3\text{ mm}$.
- **Why It Is Needed:** Provides a dense, repeating orthogonal metric calibration grid spanning the entire floor. Calibrates local metric scale with sub-centimeter precision without needing to survey individual coordinates.
- **Requirement Status:** **MANDATORY for single-camera metric homography.**
- **How to Obtain On-Site:**
  - Measure a single full tile in the lobby using a standard tape measure.
  - Alternatively, inspect architectural specification sheets or facility maintenance purchase records.
- **Fallback / Consequence if Unavailable:**
  - The system must fall back to an assumed standard tile dimension ($600 \times 600\text{ mm}$), inducing proportional scale dilation errors if actual tiles are $800 \times 800\text{ mm}$.

---

### Item 1.2: Pantry High Dining Bar Table Dimensions
- **Applicable Camera:** CCTV 2 (Pantry).
- **Physical Landmark:** High white rectangular bar table in the western quadrant.
- **Specific Measurements Needed:**
  1. Table Leg Spacing (Length along wood floor): Distance between front-right leg (`C2_P4`) and rear-right leg (`C2_P5`).
  2. Table Leg Spacing (Width across wood floor): Distance between left and right legs.
  3. Table Surface Height above Floor: Distance from floor to top surface.
- **Why It Is Needed:** Establishes a rigid, known ground-level rectangular baseline on the wooden parquet section, anchoring homography $\mathbf{H}_2$ across the western room quadrant.
- **Requirement Status:** **MANDATORY for high-precision Pantry calibration.**
- **How to Obtain On-Site:**
  - Measure distance between table leg centerpoints at floor level using a tape measure.
- **Fallback / Consequence if Unavailable:**
  - Must rely exclusively on tile seams, which are absent under the wooden parquet section.

---

### Item 1.3: Circular Planter Dimensions
- **Applicable Cameras:** CCTV 1 (Lobby), CCTV 2 (Pantry).
- **Physical Landmark:** Cylindrical white planter rings anchored to the floor.
- **Specific Measurements Needed:**
  1. Outer Base Diameter of Lobby Planter 1: Distance across circular base at floor level.
  2. Outer Base Diameter of Pantry Planter 2: Distance across circular base at floor level.
  3. Confirm whether Planter 1 and Planter 2 have identical manufactured dimensions.
- **Why It Is Needed:** A circle projected under perspective forms an ellipse. Knowing the exact metric radius $R$ allows closed-form calculation of camera tilt and focal length via circle-to-ellipse homography back-projection.
- **Requirement Status:** **MANDATORY (Highly Recommended).**
- **How to Obtain On-Site:**
  - Measure circumference $C$ with a flexible tape around the base and compute diameter $D = C / \pi$, or measure widest diameter directly at floor level.
- **Fallback / Consequence if Unavailable:**
  - Point-based DLT must be used instead of closed-form conic fitting.

---

### Item 1.4: Passage 1 Corridor Width & Doorway Dimensions
- **Applicable Camera:** CCTV 3 (Passage 1).
- **Physical Landmark:** Hallway carpet trench between Cabin 3G/3F door frames and right wall.
- **Specific Measurements Needed:**
  1. Clear Corridor Width at Near Threshold (between `C3_P1` and `C3_P2`): e.g., $1.65\text{ m}$.
  2. Clear Corridor Width at Far Pantry Doorway (between `C3_P5` and `C3_P6`): e.g., $1.65\text{ m}$.
  3. Total Corridor Length from South Wall to Pantry Threshold: e.g., $13.50\text{ m}$.
- **Why It Is Needed:** Establishes the lateral and longitudinal metric scaling for the narrow hallway, overcoming perspective compression at far depth.
- **Requirement Status:** **MANDATORY for CCTV 3 metric calibration.**
- **How to Obtain On-Site:**
  - Use a handheld laser distance meter (e.g. Bosch Blaze) placed against the south wall aiming at the pantry doorway frame.
- **Fallback / Consequence if Unavailable:**
  - Corridor tracking will be limited to 1D normalized progress percentages ($0.0 \to 1.0$) rather than physical meters.

---

## 3. Priority 2: Multi-Camera Global Alignment Anchors (MANDATORY for Global Frame)

---

### Item 2.1: Lobby-to-Pantry Blind Corridor Dimensions & Offset
- **Applicable Cameras:** CCTV 1 $\longleftrightarrow$ CCTV 2.
- **Physical Landmark:** Connecting hallway between `lobby_yellow` and `pantry_yellow`.
- **Specific Measurements Needed:**
  1. Total Walking Distance along Corridor: Straight-line or centerline distance from `lobby_yellow` tripwire threshold to `pantry_yellow` tripwire threshold ($L_{\text{blind}}$, expected $\approx 2.5\text{ m} - 4.0\text{ m}$).
  2. Lateral Offset / Jog: Whether the corridor is perfectly straight or contains an L-bend or dogleg.
  3. Orientation Angle ($\theta_{12}$): Relative compass or grid heading difference between Lobby corridor axis and Pantry room axis (typically $0^\circ$, $90^\circ$, or $180^\circ$).
- **Why It Is Needed:** **ABSOLUTELY CRITICAL.** Because CCTV 1 and CCTV 2 share zero visual overlap, this measurement is the ONLY mathematical link connecting CCTV 1 into the global world coordinate frame.
- **Requirement Status:** **MANDATORY for full 3-camera unified tracking.**
- **How to Obtain On-Site:**
  - Measure distance along floor baseboard using a laser meter or wheeled measuring tape.
  - Or export dimensions directly from architectural AutoCAD / PDF floor plans.
- **Fallback / Consequence if Unavailable:**
  - **CCTV 1 CANNOT be placed in the same metric frame as CCTV 2 and CCTV 3.**
  - *System Fallback:* CCTV 2 and CCTV 3 will operate in a unified metric frame ($X_g, Y_g$), while CCTV 1 will operate in an isolated local coordinate frame. Cross-camera handoff will rely on empirical time windows rather than metric velocity gating.

---

### Item 2.2: Confirmation of Shared Ground Tie-Points (Pantry $\longleftrightarrow$ Passage)
- **Applicable Cameras:** CCTV 2 $\longleftrightarrow$ CCTV 3.
- **Physical Landmark:** Southern doorway threshold connecting Pantry and Passage 1.
- **Specific Measurements Needed:**
  1. Distance from doorway center to Pantry Tripod Plant leg (`C2_P10` / `C3_P9`).
  2. Distance from doorway center to Pantry Planter 2 base (`C2_P6` / `C3_P10`).
  3. Confirm that the Tripod Plant is a permanent fixture (not shifted during cleaning).
- **Why It Is Needed:** Validates the visual homography registration matrix $\mathbf{H}_{3 \to 2}$ against direct physical ground distances.
- **Requirement Status:** **OPTIONAL (Recommended for Verification).**
- **How to Obtain On-Site:**
  - Laser meter measurement from the doorway threshold base to the tripod plant leg.
- **Fallback / Consequence if Unavailable:**
  - Direct visual tie-point correspondences between CCTV 2 and CCTV 3 image frames are sufficient to compute $\mathbf{H}_{3 \to 2}$ algorithmically.

---

## 4. Priority 3: Camera Installation Parameters (OPTIONAL / Auxiliary)

---

### Item 3.1: Camera Mounting Heights & Tilt Angles
- **Applicable Cameras:** All 3 Cameras.
- **Specific Information Needed:**
  1. Height of CCTV 1 lens above finished floor (expected $\approx 3.0\text{ m}$).
  2. Height of CCTV 2 lens above finished floor (expected $\approx 2.8\text{ m}$).
  3. Height of CCTV 3 lens above finished floor (expected $\approx 2.6\text{ m}$).
- **Why It Is Useful:** Enables 3D camera extrinsic pose estimation ($[\mathbf{R} \mid \mathbf{t}]$) and cross-validation of homography matrix decomposition.
- **Requirement Status:** **OPTIONAL.** (Planar homography $\mathbf{H}$ can be solved purely from 2D-to-2D ground correspondences without knowing camera height).
- **How to Obtain On-Site:**
  - Measure vertically from floor to camera dome casing with a laser distance meter.

---

## 5. Summary Action Matrix for User Review

| Measurement Item | Target Location | Measurement Tool | Priority | Impact on Final System |
|---|---|---|---|---|
| **Tile Dimensions ($W \times L$)** | Lobby & Pantry floor | Tape Measure | **MANDATORY** | Governs absolute metric accuracy of Lobby and Pantry tracking. |
| **Bar Table Leg Spacing** | Pantry parquet floor | Tape Measure | **MANDATORY** | Anchors Western quadrant of Pantry homography. |
| **Corridor Width & Length** | Passage 1 carpet | Laser Meter | **MANDATORY** | Calibrates longitudinal metric distance in Passage 1. |
| **Lobby-Pantry Corridor Length**| Connecting hallway | Laser Meter / CAD | **MANDATORY** | Links CCTV 1 into unified global facility coordinate frame. |
| **Circular Planter Diameters** | Lobby & Pantry bases | Flexible Tape | **HIGH** | Enables closed-form conic focal/tilt calibration. |
| **Doorway-to-Plant Distance** | Pantry doorway | Laser Meter | **MEDIUM** | Verifies visual tie-point registration accuracy. |
| **Camera Mounting Heights** | Ceiling mounts | Laser Meter | **LOW (OPTIONAL)**| Auxiliary verification of camera pose extrinsics. |
