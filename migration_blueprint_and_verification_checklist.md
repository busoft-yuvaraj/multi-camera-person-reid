# Multi-Camera Tracking System
## Migration Blueprint & Post-Migration Verification Checklist

> **Target Repository:** `busoft-yuvaraj/multi-camera-person-reid` (`c:/Users/Agira/Desktop/cctv-tracker`)  
> **Source Specification:** [existing_tracking_implementation_doc.md](file:///c:/Users/Agira/Desktop/cctv-tracker/existing_tracking_implementation_doc.md)  
> **Purpose:** Step-by-step verification protocol, functional test scenarios, pass/fail acceptance criteria, and post-migration validation checklist for porting tracking features into the new architecture.

---

## 1. Migration Protocol & Verification Strategy

When migrating each component from the existing architecture into the new target architecture:
1. **Maintain Interface Boundaries:** Keep decoupled boundaries between (a) Frame Ingestion, (b) Local Tracking, (c) Quality Filtering, (d) Feature Extraction, (e) Physical Gating, (f) Visual Association, and (g) Journey Store.
2. **Execute Phase-by-Phase Verification:** Do not proceed to cross-camera association testing until local tracking and quality gating have passed all checks.
3. **Verify Against Edge Cases:** The test cases below specifically exercise corner cases (occlusions, rapid re-entries, multiple people in view, and camera boundary crossings).

---

## 2. Phase-by-Phase Post-Migration Verification Checklist

### Phase A: Frame Ingestion & Spatial Geometry
- [ ] **A.1 Coordinate Rescaling Test**
  - **Procedure:** Configure camera zoom crop (e.g. `pantry: [0, 0, 720, 720]`) on a $1280 \times 720$ canvas.
  - **Verification:** Confirm that polygon ROIs and line start/end points are auto-scaled at initialization:
    $$s_x = \frac{\text{cap\_w}}{c_w} = \frac{1280}{720} \approx 1.778, \quad s_y = \frac{\text{cap\_h}}{c_h} = 1.0$$
    Detections on the resized $(1280 \times 720)$ canvas must align with scaled ROI vertices.
  - **Pass Criteria:** Foot points for people at sink ($x \in [31, 528]$ in raw crop) correctly test inside the scaled polygon without coordinate drift.

- [ ] **A.2 Resolution Mismatch Protection**
  - **Procedure:** Feed an unexpected frame dimension (e.g. $640 \times 480$ into a camera configured for $1280 \times 720$).
  - **Verification:** Ensure `FrameResolutionMismatchError` is raised immediately before YOLO inference or ROI evaluation.
  - **Pass Criteria:** Pipeline halts cleanly with informative error message detailing expected vs. actual resolution; no silent coordinate misalignments.

- [ ] **A.3 Exclusion Zone Gating**
  - **Procedure:** Generate detections with normalized bounding box center $c_x \ge 0.70$ in `waiting_lobby` (mirror/glass reflection region).
  - **Verification:** Confirm bounding boxes are discarded prior to tracking or ReID extraction.
  - **Pass Criteria:** Zero tracks or embedding bank entries are created for detections inside exclusion zones.

- [ ] **A.4 `track_only_inside_roi` Gating**
  - **Procedure:** Configure `track_only_inside_roi: ["monitored"]` and introduce detections outside the monitored ROI (e.g., background seating).
  - **Verification:** Confirm detections outside the ROI are dropped.
  - **Pass Criteria:** Only persons within the specified ROI receive tracker IDs and proceed to ReID.

---

### Phase B: Feature Extraction, Quality Gating & Preprocessing
- [ ] **B.1 Blur Filter Rejection**
  - **Procedure:** Feed an artificially blurred person crop ($\text{Laplacian variance} < 10.0$).
  - **Verification:** Confirm `calculate_quality` returns `is_accepted = False`, `reason = "too_blurry"`.
  - **Pass Criteria:** Feature extraction (OSNet) is skipped for blurry crops, preventing degraded vectors from entering embedding banks.

- [ ] **B.2 Dimension & Boundary Margin Gates**
  - **Procedure:** Feed crops with width $< 45\text{px}$, height $< 90\text{px}$, or bounding box touching the canvas boundary ($x_1 \le 0$ or $x_2 \ge W$).
  - **Verification:** Crops below minimum dimensions are rejected (`"too_small"`). Crops touching boundaries receive $boundary\_score = 0.5$.
  - **Pass Criteria:** Small crops do not trigger extraction; edge crops receive reduced quality scores.

- [ ] **B.3 Vector Normalization**
  - **Procedure:** Extract embeddings using `OSNetExtractor.extract([crop])`.
  - **Verification:** Check the Euclidean L2 norm of the resulting embedding $\vec{e}$.
  - **Pass Criteria:** $\|\vec{e}\|_2 = 1.0 \pm 10^{-6}$ for all extracted vectors.

- [ ] **B.4 Embedding Bank Diversity Filtering**
  - **Procedure:** Feed two sequential crops with cosine similarity $\ge 0.90$ into `TrackEmbeddingBank.add_embedding`.
  - **Verification:** Confirm the second vector is rejected with status `"too_similar"`.
  - **Pass Criteria:** The embedding bank holds only diverse observations, capping at 15 vectors per viewpoint.

- [ ] **B.5 Viewpoint Context Estimation**
  - **Procedure:** Pass crops representing `FRONT`, `SIDE`, and `BACK` views into `PoseExtractor.extract_viewpoint`.
  - **Verification:** Verify viewpoint classification and confidence output based on COCO keypoints.
  - **Pass Criteria:** Classification accurately identifies profile ratios ($<0.38 \rightarrow$ `SIDE`) and facial keypoint visibility ($\ge 0.45 \rightarrow$ `FRONT`).

---

### Phase C: Stage 1 Physical Gating Verification
- [ ] **C.1 Topology Gate Hard Rejection**
  - **Procedure:** Simulate a candidate whose last recorded location is `pantry` appearing in an unreachable zone not in `topology[pantry]["next_zones"]`.
  - **Verification:** `TopologyGate.is_transition_valid` must return `False`.
  - **Pass Criteria:** Candidate is discarded with rejection reason `TOPOLOGY_REJECT`.

- [ ] **C.2 Synchronous Stream Transit Handling**
  - **Procedure:** Simulate an inter-camera handoff where elapsed time $\Delta t = 0.05\text{s}$ (cameras read in the same synchronous frame loop).
  - **Verification:** With a validated transition connecting the cameras, `TemporalGate.evaluate` must return `is_valid = True` with score $0.90$ (`TRANSITION_HANDOFF`).
  - **Pass Criteria:** Legitimate handoffs are not rejected as `TOO_FAST` during synchronous processing.

- [ ] **C.3 Direction-Aware Tripwire Gate**
  - **Procedure:** Simulate crossing `pantry_yellow` in reverse direction (`SIDE_B_TO_SIDE_A`, stepping inward to pantry).
  - **Verification:** `SpatialGate.evaluate` must check direction matching against transition rules.
  - **Pass Criteria:** Inward crossings do not receive full transit authorization score ($1.0$), avoiding false handoffs.

- [ ] **C.4 Internal Movement Debounce**
  - **Procedure:** Candidate crosses a tripwire at $t = 10.0\text{s}$, but continues to be tracked in the same room until $t = 15.0\text{s}$ ($\Delta > 2.0\text{s}$).
  - **Verification:** Candidate is subsequently evaluated in another camera.
  - **Pass Criteria:** `SpatialGate` identifies the event as `TRANSITION_INTERNAL`, returning a neutral score ($0.65$) without hard rejection.

- [ ] **C.5 Active Camera Exclusion**
  - **Procedure:** Track 1 holds `Person_001` in `waiting_lobby`. Track 2 appears in `waiting_lobby` on the same frame.
  - **Verification:** Check candidate pool for Track 2.
  - **Pass Criteria:** `Person_001` is rejected for Track 2 with `EXCLUDED_OCCUPIED_IN_CAMERA`.

---

### Phase D: Stage 2 Matching, Assignment & Identity Lifecycle
- [ ] **D.1 Observation Accumulator Buffer**
  - **Procedure:** Introduce a new local track and monitor association status across frames 0, 1, 2, and 3.
  - **Verification:** `AssociationManager.associate_track` must return status `PENDING`, decision `COLLECTING_EVIDENCE` for frames with $< 3$ embeddings.
  - **Pass Criteria:** Gallery candidate search begins only once $\ge 3$ quality observations are buffered.

- [ ] **D.2 Multi-Signal Score Calculation**
  - **Procedure:** Evaluate candidate with ReID cosine score $0.75$, PAR attribute agreement score $0.85$, and context score $0.90$.
  - **Verification:** Confirm composite score matches weighted formula:
    $$S = 0.80(0.75) + 0.15(0.85) + 0.05(0.90) = 0.60 + 0.1275 + 0.045 = 0.7725$$
  - **Pass Criteria:** Anti-veto safeguards enforce that ReID $< 0.40$ caps final score at $0.45$, and ReID $\ge 0.85$ guarantees $\ge 0.95 \cdot S_{\text{ReID}}$.

- [ ] **D.3 Decision Tree Thresholds**
  - **Procedure:** Test candidate similarity scores across decision boundaries:
    - Case 1: $S \ge 0.70$ and margin $\ge 0.04 \rightarrow$ `MATCHED` immediately.
    - Case 2: $0.55 \le S < 0.70 \rightarrow$ `PENDING` (promoted after 3 sustained frames).
    - Case 3: $S < 0.55$ for 15 consecutive checks $\rightarrow$ Spawns new Global ID (`Person_XXX`).
  - **Pass Criteria:** Strict promotion compliance; no premature new ID creation before 15 checks (~45 frames).

- [ ] **D.4 Identity Lock Across Camera Handoffs**
  - **Procedure:** Confirm `Person_001` on Track 1 in Camera A. Confirm `Person_001` on Track 2 in Camera B.
  - **Verification:** Query `active_tracks.get((Camera_A, 1))` in `JourneyManager`.
  - **Pass Criteria:** Mapping for Camera A is retained in `gid_to_camera_tracks` and **not deleted** when Camera B confirms the person.

- [ ] **D.5 Simultaneous Presence Conflict Detection**
  - **Procedure:** Attempt to assign `Person_001` to two active tracks in non-overlapping cameras at the same timestamp without transition.
  - **Verification:** `JourneyManager.confirm_association` must reject the second assignment.
  - **Pass Criteria:** Logs `GLOBAL_ID_CONFLICT` and returns `False`, preventing identity cloning.

- [ ] **D.6 Identity Recovery After Tracker Renumbering**
  - **Procedure:** Track 1 (`Person_001`) disappears for 30 frames ($> \text{track\_lost\_frames}$). Reintroduce the person under new tracker ID (Track 5).
  - **Verification:** Local track 5 buffers 3 observations and queries gallery.
  - **Pass Criteria:** Candidate filter recognizes `Person_001` as `TEMPORARILY_LOST` and re-assigns `Person_001` (`GLOBAL_ID_REACQUIRED`).

- [ ] **D.7 Conflict-Aware Batch Assignment**
  - **Procedure:** Present two active tracks in the same frame competing for the same Global ID.
  - **Verification:** Ensure global 1-to-1 matching (via Hungarian assignment or sorted confidence).
  - **Pass Criteria:** The track with higher similarity receives the identity; the other track is diverted to its second-best match or pending state without race conditions.

---

### Phase E: Event Detectors, Observability & Persistence
- [ ] **E.1 Retroactive Event Attachment**
  - **Procedure:** A track crosses a transition line while in `PENDING` state (frames 0–2). The track confirms association to `Person_001` at frame 6.
  - **Verification:** Inspect `journey_manager.identities["Person_001"].journey_history`.
  - **Pass Criteria:** Buffered transition and ROI events are retroactively attached to `Person_001` and written to `journey.jsonl`.

- [ ] **E.2 ROI Dwell Time Calculation**
  - **Procedure:** Simulate foot-point entering an ROI at $t = 5.0\text{s}$ and exiting at $t = 12.5\text{s}$.
  - **Verification:** Verify `ROI_EXIT` event payload in `logs/journey.jsonl`.
  - **Pass Criteria:** Event contains `"duration_seconds": 7.5` and enter/exit timestamps are recorded.

- [ ] **E.3 Run-Isolated Audit Logging**
  - **Procedure:** Execute two separate tracking runs sequentially.
  - **Verification:** Inspect `logs/journey.jsonl` and `logs/association_decisions.jsonl`.
  - **Pass Criteria:** Logs for the second run do not append to previous run data without run identifiers, ensuring clean metric evaluation.

---

## 3. Acceptance Criteria Summary Table

| Test Category | Target Metric / Behavior | Pass Threshold | Failure Action |
| :--- | :--- | :--- | :--- |
| **Resolution Validation** | Dimension Check | Exact match to expected resolution | Raise `FrameResolutionMismatchError` & halt |
| **Blur Rejection** | Laplacian variance | $\text{Var} \ge 10.0$ | Skip OSNet extraction |
| **Vector Norm** | L2 norm | $\|\vec{e}\|_2 = 1.0 \pm 10^{-6}$ | Assert normalization |
| **Bank Diversity** | Cosine similarity | $\cos(\vec{e}_{\text{new}}, \vec{e}_i) < 0.90$ | Reject duplicate vector |
| **Stage 1 Gating** | Filtered candidate pool size | $0 \le N_{\text{candidates}} \le 2$ | Re-verify topology/temporal bounds |
| **Immediate Confirmation** | Score & Margin | $\text{Score} \ge 0.70$, $\text{Margin} \ge 0.04$ | Enter `PENDING` if below threshold |
| **Pending Promotion** | Sustained frame count | $\ge 3$ consecutive frames | Continue collecting evidence |
| **New ID Spawning** | No-match timeout | $\ge 15$ checks (~45 frames) | Do not spawn prematurely |
| **Cross-Camera Handoff** | Primary ID retention | 1 primary Global ID across cameras | Flag ID fragmentation |
