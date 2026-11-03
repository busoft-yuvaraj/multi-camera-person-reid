# Candidate Ground-Plane Reference Point Analysis

> **Document Type:** Geometric Survey, Ground-Truth Candidate Identification & Point Validation  
> **Target Scope:** 3 CCTV Cameras (Waiting Lobby, Pantry, Passage 1)  
> **Source Frame:** `frame_0000` ($t = 0.00\text{ s}$) at resolution $1280 \times 720$ px  
> **Visual Reference Artifacts:** [`annotated_reference_points/`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/annotated_reference_points/)  

---

## 1. Geometric Ground-Plane Validity Criteria

A fundamental failure mode in video homography calibration is selecting visual feature points that do not strictly lie on the ground plane ($Z = 0$).

When an elevated 3D point $(X, Y, Z > 0)$—such as a table corner, doorknob, planter rim, or wall sign—is projected onto a 2D image and treated as a ground-plane point, planar homography back-projection induces severe, non-linear parallax displacement errors:
$$\Delta \mathbf{x} = \frac{Z}{Z_{\text{cam}}} \cdot (\mathbf{x} - \mathbf{c}_p)$$
where $Z_{\text{cam}}$ is camera mounting height and $\mathbf{c}_p$ is the principal point.

### Strict Ground-Plane Validation Rules Applied in this Study:
1. **Rule 1 (Strict $Z = 0$ Contact):** Only physical vertices where fixed architectural structures, floor material transitions, or stationary furniture legs directly touch the finished floor surface are classified as `Valid`.
2. **Rule 2 (No Elevated Surfaces):** Table tops, chair seats, counter surfaces, and planter top rims are explicitly rejected.
3. **Rule 3 (Non-Collinearity):** Selected reference point sets must contain at least four non-collinear points whose convex hull encloses the primary human locomotion corridors.
4. **Rule 4 (No Dynamic Landmarks):** Movable items (e.g. mobile trash cans, rolling chairs, loose backpacks) are disqualified.

---

## 2. Camera 1 – Waiting Lobby Reference Points

Visual Reference: [`cctv1_annotated_reference_points.jpg`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/annotated_reference_points/cctv1_annotated_reference_points.jpg)

| Point ID | Camera ID | Frame Timestamp | Approx Pixel $(x, y)$ | Physical Landmark Description | Ground-Plane Validity ($Z=0$) | Confidence | Technical Justification & Suitability |
|---|---|---|---|---|---|---|---|
| `C1_P1` | `waiting_lobby` | $0.00\text{ s}$ | $(282, 581)$ | Circular Planter 1 Base – Front Contact | **Valid ($Z=0$)** | High | Base of white concrete planter ring firmly planted on tile floor; defines left edge of monitored corridor. |
| `C1_P2` | `waiting_lobby` | $0.00\text{ s}$ | $(362, 515)$ | Circular Planter 1 Base – Rightmost Contact | **Valid ($Z=0$)** | High | Tangent point where circular base curves away toward the rear; sharp contrast between white base and grey tile. |
| `C1_P3` | `waiting_lobby` | $0.00\text{ s}$ | $(319, 472)$ | Circular Planter 1 Base – Rear Contact | **Valid ($Z=0$)** | Medium | Rear floor contact point; partially shadowed by plant foliage but clearly identifiable. |
| `C1_P4` | `waiting_lobby` | $0.00\text{ s}$ | $(530, 560)$ | Curved Meeting Booth – Tangent Base Track | **Valid ($Z=0$)** | High | Innermost point where curved aluminum floor channel of glass booth touches tile floor; defines right corridor boundary. |
| `C1_P5` | `waiting_lobby` | $0.00\text{ s}$ | $(590, 650)$ | Curved Meeting Booth – Foreground Base | **Valid ($Z=0$)** | High | Base of aluminum floor profile in near foreground right; establishes near-field lateral scale. |
| `C1_P6` | `waiting_lobby` | $0.00\text{ s}$ | $(516, 406)$ | Small Planter Pot Base near Glass Wall | **Valid ($Z=0$)** | High | Ceramic pot base sitting directly on floor tiles adjacent to glass partition. |
| `C1_P7` | `waiting_lobby` | $0.00\text{ s}$ | $(338, 265)$ | Rear Lift Lobby Door Threshold – Left Frame | **Valid ($Z=0$)** | High | Crisp junction where lobby porcelain tile meets stainless steel elevator door threshold; defines far-left depth anchor. |
| `C1_P8` | `waiting_lobby` | $0.00\text{ s}$ | $(412, 265)$ | Rear Lift Lobby Door Threshold – Center Mullion | **Valid ($Z=0$)** | High | Center vertical frame base meeting floor threshold; central longitudinal anchor. |
| `C1_P9` | `waiting_lobby` | $0.00\text{ s}$ | $(466, 265)$ | Rear Lift Lobby Door Threshold – Right Frame | **Valid ($Z=0$)** | High | Right door jamb base meeting floor threshold; defines far-right depth anchor. |
| `C1_P10` | `waiting_lobby` | $0.00\text{ s}$ | $(420, 530)$ | Tile Grout Intersection – Mid Corridor | **Valid ($Z=0$)** | High | Intersection of primary longitudinal grout line with transverse grout seam; perfect planar grid vertex. |
| `C1_P11` | `waiting_lobby` | $0.00\text{ s}$ | $(500, 530)$ | Tile Grout Intersection – Right Mid Corridor | **Valid ($Z=0$)** | High | Adjacent tile grout intersection; provides known orthogonal tile spacing unit. |
| `C1_P12` | `waiting_lobby` | $0.00\text{ s}$ | $(420, 410)$ | Tile Grout Intersection – Rear Midground | **Valid ($Z=0$)** | High | Grout intersection along same longitudinal axis further back; enables vanishing point computation. |

### Convex Hull & Distribution Assessment (CCTV 1):
- **Non-Collinear Condition:** SATISFIED. Points `C1_P1`, `C1_P5`, `C1_P7`, and `C1_P9` form a large trapezoid spanning $x: 282 - 590$ and $y: 265 - 650$.
- **Corridor Coverage:** The convex hull encloses $100\%$ of the active walking area monitored by the YOLO detector.

---

## 2. Camera 2 – Pantry Reference Points

Visual Reference: [`cctv2_annotated_reference_points.jpg`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/annotated_reference_points/cctv2_annotated_reference_points.jpg)

| Point ID | Camera ID | Frame Timestamp | Approx Pixel $(x, y)$ | Physical Landmark Description | Ground-Plane Validity ($Z=0$) | Confidence | Technical Justification & Suitability |
|---|---|---|---|---|---|---|---|
| `C2_P1` | `pantry` | $0.00\text{ s}$ | $(530, 490)$ | Carpet–Wood–Tile Triple Material Junction | **Valid ($Z=0$)** | **Very High** | Highest-precision physical landmark in facility; exact vertex where dark carpet, wood parquet, and grey tile converge. |
| `C2_P2` | `pantry` | $0.00\text{ s}$ | $(49, 532)$ | Wood Parquet Floor Corner (Rear-Left) | **Valid ($Z=0$)** | High | Crisp corner where wooden floor border turns parallel to kitchen counter. |
| `C2_P3` | `pantry` | $0.00\text{ s}$ | $(860, 650)$ | Entrance Carpet Mat Right Threshold | **Valid ($Z=0$)** | High | Right boundary where carpet mat meets tile floor along right wall line. |
| `C2_P4` | `pantry` | $0.00\text{ s}$ | $(465, 675)$ | High Bar Table – Front-Right Metal Leg Base | **Valid ($Z=0$)** | High | Heavy steel table leg resting on wood parquet floor; rigidly fixed and non-movable. |
| `C2_P5` | `pantry` | $0.00\text{ s}$ | $(535, 580)$ | High Bar Table – Rear-Right Metal Leg Base | **Valid ($Z=0$)** | High | Heavy steel rear table leg resting on wood parquet floor; provides known table length baseline. |
| `C2_P6` | `pantry` | $0.00\text{ s}$ | $(580, 595)$ | Circular Planter 2 Base – Front Rim Contact | **Valid ($Z=0$)** | High | White cylindrical planter base touching tile floor in center foreground. |
| `C2_P7` | `pantry` | $0.00\text{ s}$ | $(490, 535)$ | Circular Planter 2 Base – Left Rim Contact | **Valid ($Z=0$)** | High | Leftmost point of circular base on tile floor. |
| `C2_P8` | `pantry` | $0.00\text{ s}$ | $(675, 535)$ | Circular Planter 2 Base – Right Rim Contact | **Valid ($Z=0$)** | High | Rightmost point of circular base on tile floor; orthogonal diameter pair with `C2_P7`. |
| `C2_P9` | `pantry` | $0.00\text{ s}$ | $(580, 465)$ | Circular Planter 2 Base – Rear Rim Contact | **Valid ($Z=0$)** | High | Rearmost point of circular base on tile floor; completes circle geometry for ellipse fitting. |
| `C2_P10` | `pantry` | $0.00\text{ s}$ | $(748, 650)$ | Tripod Plant Stand – Left Leg Floor Contact | **Valid ($Z=0$)** | High | Tapered wooden leg sitting directly on tile floor in foreground right. |
| `C2_P11` | `pantry` | $0.00\text{ s}$ | $(765, 660)$ | Tripod Plant Stand – Center/Front Leg Floor Contact | **Valid ($Z=0$)** | High | Front wooden leg sitting on tile floor. |
| `C2_P12` | `pantry` | $0.00\text{ s}$ | $(790, 640)$ | Tripod Plant Stand – Right Leg Floor Contact | **Valid ($Z=0$)** | High | Right wooden leg sitting on tile floor. |
| `C2_P13` | `pantry` | $0.00\text{ s}$ | $(448, 472)$ | Floor Tile Grout Intersection | **Valid ($Z=0$)** | High | Sharp four-way tile grout intersection between planter and dining table. |
| `C2_P14` | `pantry` | $0.00\text{ s}$ | $(280, 354)$ | Rear Doorway Threshold Left Base | **Valid ($Z=0$)** | Medium | Base of rear door frame visible between chair legs; defines background depth anchor. |
| `C2_P15` | `pantry` | $0.00\text{ s}$ | $(392, 320)$ | Decorative Planter Base in Back Corner | **Valid ($Z=0$)** | Medium | Large grey ceramic pot sitting on tile floor against north wall. |

### Convex Hull & Distribution Assessment (CCTV 2):
- **Non-Collinear Condition:** SATISFIED with exceptional redundancy. Points `C2_P2`, `C2_P3`, `C2_P4`, `C2_P9`, and `C2_P15` span $x: 49 - 860$ and $y: 320 - 675$.
- **Geometric Value:** Contains both orthogonal line pairs (wood/carpet thresholds) and a complete inscribed circle (circular planter rim), enabling closed-form metric homography estimation from a single known radius.

---

## 3. Camera 3 – Passage 1 Reference Points

Visual Reference: [`cctv3_annotated_reference_points.jpg`](file:///c:/Users/Agira/Desktop/cctv-tracker/physical_calibration_analysis/annotated_reference_points/cctv3_annotated_reference_points.jpg)

| Point ID | Camera ID | Frame Timestamp | Approx Pixel $(x, y)$ | Physical Landmark Description | Ground-Plane Validity ($Z=0$) | Confidence | Technical Justification & Suitability |
|---|---|---|---|---|---|---|---|
| `C3_P1` | `passage` | $0.00\text{ s}$ | $(430, 700)$ | Cabin 3G Door Frame Base on Floor | **Valid ($Z=0$)** | High | Aluminum door frame base meeting carpet floor in near foreground left. |
| `C3_P2` | `passage` | $0.00\text{ s}$ | $(635, 700)$ | Corridor Right Wall Baseboard | **Valid ($Z=0$)** | High | Baseboard junction of right service partition meeting carpet; pairs with `C3_P1` for corridor entrance width. |
| `C3_P3` | `passage` | $0.00\text{ s}$ | $(460, 580)$ | Cabin 3F Door Frame Base on Floor | **Valid ($Z=0$)** | High | Second office cabin door frame base on floor; mid-depth anchor. |
| `C3_P4` | `passage` | $0.00\text{ s}$ | $(675, 580)$ | Right Service Door Frame Base | **Valid ($Z=0$)** | High | Service door threshold base on floor; mid-depth anchor. |
| `C3_P5` | `passage` | $0.00\text{ s}$ | $(500, 500)$ | Corridor Northern Threshold – Left Base | **Valid ($Z=0$)** | High | Point where left hallway wall ends at the open entrance into the Pantry. |
| `C3_P6` | `passage` | $0.00\text{ s}$ | $(730, 500)$ | Corridor Northern Threshold – Right Base | **Valid ($Z=0$)** | High | Point where right hallway wall ends at the open entrance into the Pantry. |
| `C3_P7` | `passage` | $0.00\text{ s}$ | $(575, 485)$ | Carpet–Wood Floor Threshold (Doorway) | **Valid ($Z=0$)** | High | Straight transition line where hallway carpet meets the wooden parquet of the pantry. |
| `C3_P8` | `passage` | $0.00\text{ s}$ | $(740, 485)$ | Carpet–Tile Floor Threshold (Doorway) | **Valid ($Z=0$)** | High | Straight transition line where hallway carpet meets the grey tile floor of the pantry. |
| `C3_P9` | `passage` | $0.00\text{ s}$ | $(915, 455)$ | **Shared Landmark:** Pantry Tripod Plant Leg | **Valid ($Z=0$)** | High | Left wooden leg of the tripod plant in the pantry, visible through the doorway. Corresponds to `C2_P10`. |
| `C3_P10` | `passage` | $0.00\text{ s}$ | $(635, 430)$ | **Shared Landmark:** Pantry Circular Planter Base | **Valid ($Z=0$)** | Medium-High | Base of the pantry circular planter rim visible through doorway. Corresponds to `C2_P6`/`C2_P8`. |

### Convex Hull & Distribution Assessment (CCTV 3):
- **Non-Collinear Condition:** SATISFIED within the corridor plane. Points `C3_P1`, `C3_P2`, `C3_P5`, and `C3_P6` form an elongated trapezoid spanning $x: 430 - 730$ and $y: 500 - 700$.
- **Cross-Camera Coupling Value:** Points `C3_P7`, `C3_P8`, `C3_P9`, and `C3_P10` reside physically inside the Pantry and are directly co-visible in CCTV 2, providing tie-points for cross-camera homography registration.

---

## 4. Critical Visual Pitfalls & Negative Findings

### Pitfall 1: Visual Duplication of Circular Planters
- **Observation:** CCTV 1 (Lobby) and CCTV 2 (Pantry) both feature a white circular cylindrical planter containing large broad-leafed plants.
- **Deceptive Risk:** An automated feature matching algorithm (e.g. SIFT/ORB) or an unverified manual calibration might treat them as the same physical object.
- **Physical Reality:** They are two distinct, architectural duplicate planters located approximately $10\text{ meters}$ apart in separate rooms separated by a concrete wall and hallway. **They must never be mapped as shared tie-points.**

### Pitfall 2: Elevated Dining Table Surface ($Z \approx 1.05\text{ m}$)
- In CCTV 2, the high bar table has a crisp white laminated top with sharp rectangular corners at $(130, 420)$, $(560, 80)$, etc.
- **Risk:** Treating the table corners as ground-plane points will result in an erroneous homography projection displaced toward the camera by over $1.5\text{ meters}$ due to height parallax.
- **Correct Protocol:** Only the floor contact points of the metal table legs (`C2_P4`, `C2_P5`) at $Z = 0$ may be used.

### Pitfall 3: Planter Top Rim vs. Base Rim ($Z \approx 0.45\text{ m}$)
- The circular planter in CCTV 1 has a top rim where foliage emerges and a bottom rim where the white cylinder meets the floor.
- In low-contrast lighting or under motion blur, the top rim can be mistaken for the base.
- **Correct Protocol:** Only the ground-level base contact points (`C1_P1` - `C1_P3`, `C2_P6` - `C2_P9`) are valid.
