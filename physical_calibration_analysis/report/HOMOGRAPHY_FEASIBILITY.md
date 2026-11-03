# Homography Feasibility & Coordinate System Assessment

> **Document Type:** Mathematical Formulation, Feasibility Assessment & Calibration Methodology  
> **Theoretical Basis:** 2D Projective Geometry, Planar Homography ($\mathbb{P}^2 \to \mathbb{P}^2$), Direct Linear Transformation (DLT)  
> **Target Scope:** 3 CCTV Cameras (Waiting Lobby, Pantry, Passage 1)  

---

## 1. Mathematical Formulation of Planar Homography

A planar homography is a projective transformation between two planes. In CCTV computer vision, when the physical world floor is planar ($Z_w = 0$), the mapping between a point on the ground plane in metric world coordinates $\mathbf{X}_w = [X_w, Y_w, 1]^T$ and its corresponding image projection $\mathbf{x} = [u, v, 1]^T$ is expressed as a $3 \times 3$ non-singular matrix $\mathbf{H}$:

$$s \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \mathbf{H} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix}$$

where $s$ is an arbitrary non-zero projective scale factor. Because $\mathbf{H}$ has 8 degrees of freedom (defined up to scale), it requires a minimum of **four point correspondences**, with no three points collinear, to solve uniquely via the Direct Linear Transformation (DLT) algorithm or Singular Value Decomposition (SVD):

$$\mathbf{A} \mathbf{h} = \mathbf{0}, \quad \text{where } \mathbf{h} = [h_{11}, h_{12}, \dots, h_{33}]^T$$

To project detected pedestrian footpoints $\mathbf{x}_{\text{foot}} = [u, v, 1]^T$ from the image canvas back into the world ground plane, the inverse homography $\mathbf{H}^{-1}$ is applied:

$$\begin{bmatrix} X_w \\ Y_w \\ 1 \end{bmatrix} \sim \mathbf{H}^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

$$X_w = \frac{h'_{11} u + h'_{12} v + h'_{13}}{h'_{31} u + h'_{32} v + h'_{33}}, \quad Y_w = \frac{h'_{21} u + h'_{22} v + h'_{23}}{h'_{31} u + h'_{32} v + h'_{33}}$$

where $\mathbf{H}' = \mathbf{H}^{-1}$.

---

## 2. Camera-by-Camera Homography Feasibility Evaluation

---

### 2.1 CCTV 1 – Waiting Lobby

| Evaluation Parameter | Assessment | Technical Explanation |
|---|---|---|
| **1. Ground-Plane Visibility** | **Sufficient ($40\%$ area)** | Central corridor has clear, unobstructed view from $y = 265$ to $y = 720$. |
| **2. Reference Points Availability** | **Abundant ($12$ identified)** | Multiple four-way tile grout intersections and structural base points available. |
| **3. Planarity Assumption** | **Strictly Planar** | Modern commercial porcelain tiles laid on leveled concrete subfloor. Zero grade slope. |
| **4. Perspective Distortion** | **Manageable** | Foreshortening ratio $\approx 2.5:1$ is standard for elevated indoor CCTV. No singular horizon points within active ROI. |
| **5. Manual Point Annotation** | **Required** | User must click and verify 4 to 8 ground points on frame 0 using a GUI tool. |
| **6. Metric Coordinates ($X_m, Y_m$)**| **Achievable** | If commercial tile dimensions ($L_{\text{tile}} \times W_{\text{tile}}$, e.g. $600\text{ mm} \times 600\text{ mm}$) are provided, exact metric scale is established. |
| **7. Relative Coordinates ($X_r, Y_r$)**| **Achievable immediately** | Arbitrary unit grid (e.g. $[0, 1] \times [0, 1]$) can be computed without external data. |

**Feasibility Verdict:** **HIGHLY FEASIBLE.** Metric homography $\mathbf{H}_1$ can be established with sub-centimeter accuracy across the primary corridor once tile dimensions are provided.

---

### 2.2 CCTV 2 – Pantry

| Evaluation Parameter | Assessment | Technical Explanation |
|---|---|---|
| **1. Ground-Plane Visibility** | **Exceptional ($65\%$ area)** | Wide open floor revealing ceramic tiles, wooden parquet, and entrance carpet mat. |
| **2. Reference Points Availability** | **Extremely Rich ($15$ identified)**| Triple floor material junction (`C2_P1`), bar table legs, circular planter rim, and tile grid. |
| **3. Planarity Assumption** | **Strictly Planar** | Flush transitions between carpet mat, parquet, and tile floor (commercial low-profile ADA transitions). |
| **4. Perspective Distortion** | **Optimal** | $35^\circ - 40^\circ$ downward pitch provides high ground separation and low tangential distortion. |
| **5. Manual Point Annotation** | **Required** | User digitizes table leg bases, material junctions, and planter rim points. |
| **6. Metric Coordinates ($X_m, Y_m$)**| **Achievable** | Direct metric scale can be derived from bar table length, planter diameter, or tile grid dimensions. |
| **7. Relative Coordinates ($X_r, Y_r$)**| **Achievable immediately** | Can be mapped to arbitrary normalized top-down coordinates. |

**Feasibility Verdict:** **HIGHLY FEASIBLE (Highest Accuracy Potential).** CCTV 2 represents an ideal ground-plane calibration environment. Planar homography $\mathbf{H}_2$ will exhibit minimal condition number ($\kappa(\mathbf{H}_2) \ll 10^3$).

---

### 2.3 CCTV 3 – Passage 1

| Evaluation Parameter | Assessment | Technical Explanation |
|---|---|---|
| **1. Ground-Plane Visibility** | **Sufficient ($45\%$ area)** | Full length of corridor floor is visible along the central axial trench. |
| **2. Reference Points Availability** | **Sufficient ($10$ identified)** | Door frame bases, corridor thresholds, plus shared landmarks inside the Pantry. |
| **3. Planarity Assumption** | **Strictly Planar** | Commercial low-pile carpet glued directly to leveled concrete slab. |
| **4. Perspective Distortion** | **CHALLENGING (High Foreshortening)**| Shallow pitch creates $7:1$ foreshortening ratio. Far depths ($y < 460$) compress $1\text{ meter}$ into $< 18\text{ px}$. |
| **5. Manual Point Annotation** | **Required** | User marks corridor door jambs and entrance threshold vertices. |
| **6. Metric Coordinates ($X_m, Y_m$)**| **Achievable with depth weighting** | Longitudinal metric mapping is reliable; lateral ($X$) metric mapping becomes sensitive at far depth. |
| **7. Relative Coordinates ($X_r, Y_r$)**| **Achievable** | Normalized corridor representation ($[0, W_{\text{corr}}] \times [0, L_{\text{corr}}]$). |

**Feasibility Verdict:** **FEASIBLE WITH DEPTH UNCERTAINTY FILTERING.** Homography $\mathbf{H}_3$ is feasible, but the inverse mapping $\mathbf{H}_3^{-1}$ must incorporate a depth-dependent covariance matrix $\mathbf{\Sigma}(y)$ that expands uncertainty ellipses as $y \to y_{\text{horizon}}$.

---

## 3. Comparative Evaluation of Calibration Paradigms

---

### Approach A: Relative Ground-Plane Calibration (Arbitrary Normalized Space)
- **Concept:** Map image pixels $(u, v)$ to an arbitrary normalized 2D plane (e.g. $[0, 1] \times [0, 1]$ or $[0, 1000] \times [0, 1000]$) using visual aspect ratios estimated from quadrilateral vertices.
- **Requirements:** 4 user-selected points forming an estimated rectangle on the floor.
- **Advantages:** Zero external measurements required; can be implemented immediately from footage alone.
- **Limitations:**
  - No physical scale ($1\text{ unit} \ne 1\text{ meter}$).
  - Speeds cannot be calculated in $\text{m/s}$ or $\text{km/h}$, preventing real-world human velocity gating ($0.5\text{ m/s} \le v \le 2.5\text{ m/s}$).
  - Each camera produces an arbitrary scale factor $\lambda_i$, making cross-camera distance comparisons invalid without inter-camera scale normalization.
- **Expected Reliability:** Low-to-Medium for cross-camera association; adequate only for single-camera trajectory smoothing.

---

### Approach B: Metric Ground-Plane Calibration (Known Physical Measurements)
- **Concept:** Map image pixels $(u, v)$ to real-world metric coordinates $(X_w, Y_w)$ in **meters** by supplying ground-truth distances between surveyed landmarks (e.g., standard tile dimensions, bar table length, corridor width).
- **Requirements:** At least one known orthogonal pair of metric lengths $(L_x, L_y)$ or surveyed coordinates for $\ge 4$ points.
- **Advantages:**
  - Yields genuine physical coordinates: $[X_w, Y_w]$ in meters.
  - Enables physics-based velocity filtering: $v = \frac{\sqrt{\Delta X_w^2 + \Delta Y_w^2}}{\Delta t}$, reliably rejecting physically impossible candidate transitions.
  - Enables direct bounding-box height validation and ground-plane spatial proximity clustering.
- **Limitations:** Requires on-site manual measurements with tape measure or laser meter.
- **Expected Reliability:** **HIGH.** Yields repeatable, sub-decimeter tracking accuracy on the floor plane.

---

### Approach C: Cross-Camera Alignment (Unified Multi-Camera World System)
- **Concept:** Transform all three camera views into a single, global facility coordinate system $(X_{\text{global}}, Y_{\text{global}})$:
  $$\mathbf{X}_{\text{global}} = \mathbf{T}_{i \to \text{global}} \cdot \mathbf{H}_i^{-1} \cdot \mathbf{x}_i$$
- **Requirements:**
  1. For CCTV 2 & CCTV 3: Direct visual tie-point correspondences across the shared southern doorway.
  2. For CCTV 1 & CCTV 2: Metric corridor translation vector $\mathbf{t}_{12} = [\Delta X_{12}, \Delta Y_{12}]^T$ and rotation angle $\theta_{12}$ from architectural floor plans.
- **Advantages:**
  - Continuous global trajectory tracking across the entire corporate floor.
  - Global Euclidean distance gating in `CameraAssociationManager`: $\mathcal{D}_{\text{spatial}}(i, j) = \|\mathbf{X}_i - \mathbf{X}_j\|_2$.
  - Seamless visualization on a single master floor-plan dashboard.
- **Limitations:** Dependent on accurate manual survey of the unmonitored blind corridor.
- **Expected Reliability:** **HIGH** across Pantry-Passage; **MEDIUM-HIGH** across Lobby-Pantry (governed by accuracy of manual corridor measurement).

---

## 4. Why Independent Homographies Do Not Establish a Common Coordinate System

A common misconception in multi-camera tracking is that computing an independent homography matrix for each camera ($\mathbf{H}_1$ for Lobby, $\mathbf{H}_2$ for Pantry, $\mathbf{H}_3$ for Passage) is sufficient to track individuals in a unified coordinate frame.

### Mathematical Proof of Incompatibility:
Let $\mathbf{H}_1$ map camera 1 pixels to an origin $(0, 0)$ defined at the corner of the Lobby circular planter:
$$\mathbf{X}_{w1} = \mathbf{H}_1^{-1} \mathbf{x}_1$$

Let $\mathbf{H}_2$ map camera 2 pixels to an origin $(0, 0)$ defined at the Pantry table leg:
$$\mathbf{X}_{w2} = \mathbf{H}_2^{-1} \mathbf{x}_2$$

Even if both coordinate systems are scaled in meters:
$$\mathbf{X}_{w1} \ne \mathbf{X}_{w2}$$
because they possess **different coordinate origins**, **different rotation axes**, and **an unknown spatial displacement vector**:

$$\mathbf{X}_{\text{global}} = \mathbf{R}_i \mathbf{X}_{wi} + \mathbf{t}_i$$

Without determining the extrinsic Euclidean rigid body transformation $[\mathbf{R}_i \mid \mathbf{t}_i]$ relating each camera's local ground frame to the global facility origin, the numerical outputs cannot be compared.

### Summary of Requirements to Form a Global System:
1. **Pantry (CCTV 2) is designated as the Global Master Origin:**
   $$\mathbf{X}_{\text{global}} = \mathbf{X}_{w2} \implies \mathbf{H}_{\text{global}, 2} = \mathbf{H}_2$$
2. **Passage 1 (CCTV 3) is registered to Master Origin via Direct Tie-Points:**
   $$\mathbf{H}_{\text{global}, 3} = \mathbf{H}_{3 \to 2} \cdot \mathbf{H}_3$$
3. **Waiting Lobby (CCTV 1) is registered to Master Origin via Surveyed Rigid Offset:**
   $$\mathbf{X}_{\text{global}} = \begin{bmatrix} \cos \theta_{12} & -\sin \theta_{12} & \Delta X_{12} \\ \sin \theta_{12} & \cos \theta_{12} & \Delta Y_{12} \\ 0 & 0 & 1 \end{bmatrix} \mathbf{X}_{w1}$$

This establishes a mathematically rigorous, unified global floor coordinate system across the entire facility.
