# Multi-Camera Person Tracking & Re-Identification System
## Comprehensive Implementation Architecture, Feature Inventory & Operational Reference

> **Document Type:** Technical Architecture Specification, Component Inventory & Implementation Reference  
> **Target Repository:** `busoft-yuvaraj/multi-camera-person-reid` (`c:/Users/Agira/Desktop/cctv-tracker`)  
> **Audited Core Modules:** [app/multi_camera_processor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py), [app/reid/](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid), [app/association/](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association), [app/events/](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events), [app/journey/](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey), [app/violation/](file:///c:/Users/Agira/Desktop/cctv-tracker/app/violation), [config/app_config_new.yaml](file:///c:/Users/Agira/Desktop/cctv-tracker/config/app_config_new.yaml)  
> **Purpose:** Documents the complete runtime flow, every feature, algorithm, mathematical formulation, configuration parameter, and lifecycle behavior currently implemented in the codebase.

---

## 1. System Overview & Runtime Pipeline Flow

The system continuously processes synchronized video feeds from multiple CCTV cameras across a facility (e.g., `waiting_lobby`, `pantry`, `passage`). It maps ephemeral, camera-local detections (`track_id: 1, 2, ...`) produced by YOLOv11 and BoT-SORT to persistent, facility-wide physical identities (`Person_001`, `Person_002`, ...).

### 1.1 End-to-End Pipeline Flow

```mermaid
flowchart TD
    subgraph Capture["1. Synchronous Frame Ingestion & Transform"]
        V1[Camera 1: waiting_lobby] --> S1[Synchronous Frame Reader]
        V2[Camera 2: pantry] --> S1
        V3[Camera 3: passage] --> S1
        S1 --> Crop[Coordinate Transform & Zoom<br/>Pantry 720x720 -> 1280x720]
        Crop --> ResCheck[Resolution Validator<br/>app/events/roi_detector.py]
    end

    subgraph Tracking["2. Local Tracking & Spatial Filtering"]
        ResCheck --> YOLO[YOLOv11 Detection + BoT-SORT Tracker<br/>Independent state per camera]
        YOLO --> LocalMap[Tracker ID Isolation<br/>local_id_map -> sequential local_tid]
        LocalMap --> Excl[Exclusion Zone Filter<br/>Normalized BBox Center Gating]
        Excl --> TrackROI[track_only_inside_roi Gating]
    end

    subgraph Events["3. Floor-Plan Event Engine"]
        TrackROI --> FootPt[Foot-Point Calculation<br/>fx = (x1+x2)/2, fy = y2]
        FootPt --> TransDet[TransitionDetector<br/>Signed 2D Cross-Product Tripwires]
        FootPt --> RoiDet[RoiDetector<br/>OpenCV Polygon Containment]
        TransDet --> JMEvent[JourneyManager Buffer]
        RoiDet --> JMEvent
    end

    subgraph FeatureEng["4. Feature Extraction & Embedding Bank"]
        TrackROI --> QCheck{reid_interval == 0?<br/>calculate_quality}
        QCheck -- Pass --> OSNet[OSNet 512-D Extractor<br/>L2-Normalized Embedding]
        QCheck -- Pass --> Pose[PoseExtractor<br/>COCO Keypoints -> VIEWPOINT]
        QCheck -- Pass --> PAR[ParExtractor<br/>Upper/Lower HSV Colors]
        OSNet & Pose & PAR --> Bank[TrackEmbeddingBank<br/>15 vectors/view, diversity >= 0.90]
    end

    subgraph Stage1["5. Stage 1: Physical Gating (CandidateFilter)"]
        Bank --> CFilter{Observations >= 3?}
        CFilter -- Yes --> OccGate[Active Occupancy Gate<br/>Exclude currently occupied GIDs]
        OccGate --> TopoGate[TopologyGate<br/>Physical Adjacency Matrix]
        TopoGate --> TempGate[TemporalGate<br/>Min/Max Transit Seconds Bounds]
        TempGate --> SpatGate[SpatialGate<br/>Tripwire Direction & Handoff Priority]
        SpatGate --> ValidPool[Filtered Candidate Pool<br/>0 to 2 plausible GIDs]
    end

    subgraph Stage2["6. Stage 2: Identity Association & Scoring"]
        ValidPool --> Matcher[GlobalMatcher.compute_candidate_score]
        Matcher --> ScoreCalc[Composite Formula:<br/>w_reid*ReID + w_par*PAR + w_ctx*Context]
        ScoreCalc --> DecisionTree{Decision Tree}
        DecisionTree -- sim >= 0.70 & margin >= 0.04 --> Confirmed[CONFIRMED_MATCH]
        DecisionTree -- sim >= 0.55 or Cross-View --> Pending[PENDING_CANDIDATE<br/>Promote after 3 frames]
        DecisionTree -- No Candidates / Below Thresh --> NoMatchCounter[Increment no_match_counter<br/>Spawn New GID after 15 checks]
    end

    subgraph Persistence["7. State Lifecycle, Gallery & Audit"]
        Confirmed & Pending & NoMatchCounter --> JMUpdate[JourneyManager State Update<br/>Concurrent Camera Tracking]
        Confirmed --> GalUpdate[GlobalGallery & Qdrant Upsert<br/>If quality >= 0.60]
        JMUpdate --> AuditLogs[JourneyStore<br/>journey.jsonl & association_decisions.jsonl]
    end
```

### 1.2 Execution Lifecycle Sequence
1. **Camera Ingestion:** [MultiCameraProcessor.run](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L360-L460) sequentially reads frames across all cameras on each iteration `frame_idx`.
2. **Coordinate Slicing & Resizing:** Cameras with configured zoom crops (e.g. `pantry: [0, 0, 720, 720]`) are sliced and resized to base canvas resolution $(1280 \times 720)$.
3. **Detection & Local Tracking:** Independent YOLOv11 and BoT-SORT tracker instances run per camera view. Tracker IDs are mapped to sequential camera-local IDs via `local_id_map`.
4. **Spatial Filtering:** Normalized center exclusions and `track_only_inside_roi` filter spurious edge tracks or reflection zones.
5. **Event Checking:** Foot-point coordinates ($(x_1+x_2)/2, y_2$) are evaluated against directional tripwires and polygon ROIs.
6. **Feature Extraction:** Every `reid_interval` (3 frames), crops passing Laplacian sharpness and dimension filters are converted into 512-D OSNet embeddings, pose viewpoints, and PAR attributes.
7. **Two-Stage Association:**
   - **Stage 1 (Physical Gating):** Active occupancy, topology graph, transit duration bounds, and spatial tripwires reduce the candidate set to 0–2 plausible identities.
   - **Stage 2 (Identity Verification):** Multi-signal scoring evaluates visual appearance, attribute colors, and viewpoint difficulty to confirm, hold as pending, or spawn new identities.
8. **Audit Trail & Persistence:** All state changes and association decisions are written to append-only JSONL files.

---

## 2. Complete Inventory of Implemented Features, Algorithms & Behaviors

### 2.1 Multi-Camera Video Ingestion & Coordinate Geometry
- **Module:** [app/multi_camera_processor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py)
- **Purpose:** Ingests camera streams in synchronous lockstep, applies zoom cropping/resizing, validates coordinate spaces, and tracks presentation timestamps.
- **Implemented Algorithms & Logic:**
  - **Lockstep Reading:** `for cam in self.cameras_info:` sequentially reads one frame per camera per iteration `frame_idx` ([multi_camera_processor.py:367-375](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L367-L375)).
  - **Crop & Auto-Scale Preprocessing:** For cameras specifying a zoom crop (e.g., `pantry: [0, 0, 720, 720]`), raw frames are sliced to `[cy1:cy2, cx1:cx2]` and resized to the base canvas resolution `(w, h)` (default `1280x720`). Transition lines and ROI coordinates are auto-scaled at initialization:
    $$s_x = \frac{\text{cap\_w}}{c_w}, \quad s_y = \frac{\text{cap\_h}}{c_h}$$
    $$\text{pt}'_x = (\text{pt}_x - c_{x1}) \cdot s_x, \quad \text{pt}'_y = (\text{pt}_y - c_{y1}) \cdot s_y$$
  - **Strict Frame Resolution Validation:** [RoiDetector.validate_frame](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/roi_detector.py#L154-L182) validates dimensions before any detection or ROI evaluation occurs. Raises [FrameResolutionMismatchError](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/roi_detector.py#L42-L67) if dimensions differ from configured resolution, preventing coordinate distortion.
  - **Synthetic Timeline Synthesis:** Computes presentation timestamp and shared global timestamp:
    $$t_{\text{pres}} = \frac{\text{frame\_idx}}{\text{fps}}, \quad t_{\text{shared}} = t_{\text{pres}} + \text{start\_offset}$$
- **Configuration Parameters:**
  - `cameras.<cam_id>.video_path`: Path to video file or RTSP stream.
  - `cameras.<cam_id>.expected_resolution`: Expected `[W, H]` (e.g., `[1280, 720]`).
  - `cameras.<cam_id>.crop`: Bounding box `[x1, y1, x2, y2]` to crop before scaling.
  - `cameras.<cam_id>.start_offset`: Seconds offset to synchronize staggered camera recordings.
  - `raise_on_resolution_error`: Boolean (default `true`) to halt execution on mismatch.

---

### 2.2 Local Detection, Tracking & Spatial Isolation
- **Modules:** [app/multi_camera_processor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py), [app/tracker_manager.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/tracker_manager.py), [botsort.yaml](file:///c:/Users/Agira/Desktop/cctv-tracker/botsort.yaml)
- **Purpose:** Detects people in individual frames and maintains smooth local tracklets per camera view.
- **Implemented Algorithms & Logic:**
  - **Model:** Ultralytics YOLOv11 (`yolo11n.pt` / `yolo11m.pt`) configured for `classes=[0]` (person), confidence `detection_confidence: 0.25`, and image size `imgsz: 960` or `1280`.
  - **BoT-SORT Local Tracking:** DeepSORT/BoT-SORT tracker with Kalman filtering and camera motion compensation (`cmc_method: none` / `orb`). Each camera instantiates an independent YOLO instance to prevent state leakage ([multi_camera_processor.py:250](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L250)).
  - **Tracker ID Isolation:** Raw tracker IDs from YOLO are mapped through `state["local_id_map"]` to ensure sequential camera-local IDs (`1, 2, ...`) that cannot conflict across cameras ([multi_camera_processor.py:511-515](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L511-L515)).
  - **Exclusion Zone Filtering:** Tracks whose normalized bounding box center $(c_x, c_y)$ falls within configured exclusion rectangles are immediately dropped (used to discard mirror/glass partitions in `waiting_lobby` where $c_x \ge 0.70$) ([multi_camera_processor.py:521-539](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L521-L539)).
  - **ROI-Restricted Tracking (`track_only_inside_roi`):** If configured, tracks whose foot point $(f_x, f_y)$ is outside specified ROIs are discarded ([multi_camera_processor.py:540-560](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L540-L560)).
- **Configuration Parameters:**
  - `model_path`: Path to YOLO model (`yolo11n.pt`).
  - `tracker_config`: Tracker YAML (`botsort.yaml`).
  - `detection_confidence`: Threshold (default `0.25`).
  - `imgsz`: Inference image dimension (e.g. `960` or `1280`).
  - `exclusion_zones.<cam_id>`: List of `[x1_norm, y1_norm, x2_norm, y2_norm]`.
  - `cameras.<cam_id>.track_only_inside_roi`: List of allowed ROI IDs.

---

### 2.3 ReID Feature Extraction, Quality Gating & Masking
- **Modules:** [app/reid/extractor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractor.py), [app/reid/quality.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/quality.py), [app/reid/extractors/pose_extractor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractors/pose_extractor.py), [app/reid/extractors/par_extractor.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractors/par_extractor.py)
- **Purpose:** Extracts robust deep appearance embeddings, filters out degraded crops, and determines viewpoint context.
- **Implemented Algorithms & Logic:**
  - **Interval Decimation:** Extraction runs only when `frame_idx % reid_interval == 0` (default: every 3 frames, ~8–10 Hz) ([multi_camera_processor.py:674](file:///c:/Users/Agira/Desktop/cctv-tracker/app/multi_camera_processor.py#L674)).
  - **Quality Gate ([calculate_quality](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/quality.py#L4-L43)):**
    - Confidence check: $conf \ge min\_conf$ (`0.35`).
    - Minimum size check: $w \ge 45\text{px}$, $h \ge 90\text{px}$.
    - Sharpness / Blur check: Computes variance of Laplacian on grayscale crop:
      $$\text{Var}(\nabla^2 I) \ge \text{blur\_thresh} \quad (10.0)$$
    - Boundary penalty: $boundary\_score = 0.5$ if bounding box touches image edge; $1.0$ otherwise.
    - Composite quality formula:
      $$Q = 0.40 \cdot conf + 0.25 \cdot \text{size\_score} + 0.20 \cdot \text{sharpness\_score} + 0.15 \cdot boundary\_score$$
  - **OSNet Deep Feature Extraction:** [OSNetExtractor](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractor.py) (`osnet_x1_0`). Produces 512-dimensional feature vector, normalized to unit L2 norm:
    $$\vec{e} = \frac{\vec{f}}{\|\vec{f}\|_2}$$
  - **Viewpoint Context Estimation:** [PoseExtractor](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractors/pose_extractor.py) uses YOLOv11-Pose (`yolo11n-pose.pt`, 17 COCO keypoints):
    - Profile Ratio: $\text{ratio} = \frac{\text{shoulder\_dist}}{\text{torso\_height}}$. If $\text{ratio} < 0.38$ or one ear/eye dominant $\rightarrow$ `SIDE`.
    - Face Keypoint Visibility: Nose/eyes confidence $\ge 0.45 \rightarrow$ `FRONT`. Wide shoulders without facial keypoints $\rightarrow$ `BACK`.
  - **PAR Attribute Recognition:** [ParExtractor](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractors/par_extractor.py) segments upper body ($y \in [0.15h, 0.55h]$) and lower body ($y \in [0.55h, 0.90h]$) into HSV color histograms, voting on dominant color (`black`, `white`, `grey`, `blue`, `green`, `red`, `yellow`, `orange_brown`, `purple`).
- **Configuration Parameters:**
  - `reid_enabled`: Boolean flag.
  - `reid_model_path`: Model name/path (`osnet_x1_0`).
  - `reid_interval`: Frame step interval (`3`).
  - `min_reid_confidence`: Detection confidence gate (`0.35`).
  - `min_reid_width` / `min_reid_height`: Dimension gates (`45`, `90`).
  - `blur_threshold`: Laplacian variance threshold (`10.0`).

---

### 2.4 Embedding Bank & Gallery Diversity
- **Modules:** [app/reid/embedding_bank.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/embedding_bank.py), [app/reid/identity.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/identity.py), [app/reid/global_gallery.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/global_gallery.py), [app/reid/qdrant_gallery.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/qdrant_gallery.py)
- **Purpose:** Stores and manages temporal multi-view observations per tracklet and per Global ID without representation collapse or drift.
- **Implemented Algorithms & Logic:**
  - **TrackEmbeddingBank (Local Cache):** Keyed by `(camera_id, local_track_id)`. Partitions observations into 4 viewpoint sub-banks: `FRONT`, `SIDE`, `BACK`, `UNKNOWN`.
  - **Cosine Diversity Filter:** When a new embedding $\vec{e}_{\text{new}}$ is extracted, it is compared against all existing embeddings in the corresponding viewpoint sub-bank:
    $$\cos(\vec{e}_{\text{new}}, \vec{e}_i) \ge \text{diversity\_threshold} \quad (0.90)$$
    If similarity is $\ge 0.90$, the vector is deemed redundant and rejected (`"too_similar"`).
  - **Quality-Based Eviction:** When a viewpoint sub-bank reaches capacity (`max_size_per_view = 15`), the lowest-quality entry is replaced if $Q_{\text{new}} > Q_{\min}$ ([embedding_bank.py:43-50](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/embedding_bank.py#L43-L50)).
  - **Track Prototype Synthesis:** Computes the L2-normalized mean centroid across all valid observation vectors in the bank ([matcher.py:90-93](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/matcher.py#L90-L93)):
    $$\vec{t}_{\text{proto}} = \frac{\sum_{i=1}^N \hat{e}_i}{\left\|\sum_{i=1}^N \hat{e}_i\right\|_2}$$
  - **Global Gallery Update:** Confirmed tracks with quality $\ge 0.60$ update [GlobalGallery](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/global_gallery.py) and trigger background upload to [QdrantGallery](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/qdrant_gallery.py).
- **Configuration Parameters:**
  - `reid_bank_size`: Maximum vectors retained per viewpoint (`15`).
  - `embedding_diversity_threshold`: Cosine diversity limit (`0.90`).
  - `gallery_update_min_quality`: Minimum crop quality to admit into gallery (`0.60`).
  - `clear_gallery_on_start`: Purges database/pickle on startup (`true`).

---

### 2.5 Stage 1: Candidate Filtering Engine (Physical Gating)
- **Modules:** [app/association/candidate_filter.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/candidate_filter.py), [app/association/topology.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/topology.py), [app/association/temporal_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/temporal_gate.py), [app/association/spatial_gate.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/spatial_gate.py)
- **Purpose:** Filters the candidate pool down to 0–2 physically plausible identities before visual comparison, preventing spurious matches across the building.
- **Implemented Algorithms & Logic:**
  - **1. Active Occupancy Exclusion:** Any Global ID actively seen in the same camera on another track, or visible in another camera within $0.5\text{s}$, is placed into `active_camera_gids`. Candidate is rejected with `EXCLUDED_OCCUPIED_IN_CAMERA` ([candidate_filter.py:51-53](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/candidate_filter.py#L51-L53)).
  - **2. Topology Gate (Hard Constraint):** Evaluates if transit from candidate's `last_zone` to `current_zone` exists in the topology adjacency matrix:
    $$\text{current\_zone} \in \text{topology}[\text{last\_zone}][\text{"next\_zones"}]$$
    If not, rejects with `TOPOLOGY_REJECT` ([topology.py:20-34](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/topology.py#L20-L34)).
  - **3. Temporal Gate (Transit Bounds):** Computes elapsed time $\Delta t = t_{\text{curr}} - t_{\text{last\_seen}}$:
    - Same camera: $\Delta t \le \text{same\_camera\_reacquisition\_max\_seconds}$ ($60.0\text{s}$). If expired, rejected with `SAME_CAMERA_EXPIRED`.
    - Cross-camera: Checks `[min_seconds, max_seconds]` (typically $[0.5\text{s}, 45.0\text{s}]$).
      - If $\Delta t < \text{min\_seconds}$ ($0.5\text{s}$): Rejects as `TOO_FAST` **unless** a validated transition event explicitly connects the camera pair (`has_validated_transition = True`), in which case score is set to $0.90$ ([temporal_gate.py:54-60](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/temporal_gate.py#L54-L60)).
      - If $\Delta t > \text{max\_seconds}$ ($45.0\text{s}$): Rejects as `TOO_SLOW`.
  - **4. Spatial Gate (Tripwire Alignment):**
    - Checks candidate's `last_transition_id`, `direction`, and timestamp.
    - If candidate crossed a transition line matching the current camera within $45\text{s}$ in the expected exit direction, returns `spatial_score = 1.0` (`TRANSITION_MATCH`).
    - Reverse-direction crossings (stepping back into room) or internal room movement ($t_{\text{seen}} - t_{\text{trans}} > 2.0\text{s}$) degrade score to $0.60$–$0.65$ without hard rejection.
    - Unmatched transitions receive a soft penalty ($0.50$) without hard rejection ([spatial_gate.py:41-83](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/spatial_gate.py#L41-L83)).
- **Configuration Parameters:**
  - `topology.<zone>.next_zones`: List of reachable zones.
  - `temporal_constraints.<cam1>_to_<cam2>`: `min_seconds` (0.5) and `max_seconds` (45.0).
  - `temporal_constraints.same_camera_reacquisition_max_seconds`: Max occlusion time (60.0s).
  - `transition_rules`: List of `{from_camera, transition, direction, candidate_zones}`.

---

### 2.6 Stage 2: Identity Association, Scoring & State Machine
- **Modules:** [app/association/association_manager.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/association_manager.py), [app/reid/matcher.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/matcher.py)
- **Purpose:** Scores surviving candidates using appearance, attributes, and context; applies multi-frame hysteresis to prevent ID switching.
- **Implemented Algorithms & Logic:**
  - **Evidence Collection Buffer:** Unconfirmed tracks must accumulate $\ge \text{reid\_confirm\_frames}$ (3) observation vectors before candidate querying starts. Prior to 3 frames, returns `PENDING: COLLECTING_EVIDENCE` ([association_manager.py:174-180](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/association_manager.py#L174-L180)).
  - **ReID Appearance Score:** The probe track prototype $\vec{t}_{\text{proto}}$ is compared against **every stored vector** $\vec{e}_j$ in the candidate's gallery across all viewpoints:
    $$S_{\text{ReID}} = \max_j \left( \frac{\vec{t}_{\text{proto}} \cdot \vec{e}_j}{\|\vec{t}_{\text{proto}}\| \|\vec{e}_j\|} \right)$$
  - **PAR Attribute Score ($S_{\text{PAR}}$):** Weighted combination of upper garment color agreement ($45\%$), lower garment color agreement ($45\%$), and deep attribute feature similarity ($10\%$) ([par_extractor.py:217](file:///c:/Users/Agira/Desktop/cctv-tracker/app/reid/extractors/par_extractor.py#L217)).
  - **Context Score ($S_{\text{ctx}}$):** $0.45 \cdot S_{\text{transition}} + 0.35 \cdot S_{\text{temporal}} + 0.20 \cdot \text{difficulty}$.
  - **Composite Score Formula:**
    $$S_{\text{composite}} = w_{\text{reid}} \cdot S_{\text{ReID}} + w_{\text{par}} \cdot S_{\text{PAR}} + w_{\text{ctx}} \cdot S_{\text{ctx}}$$
    *(With defaults $w_{\text{reid}} = 0.80$, $w_{\text{par}} = 0.15$, $w_{\text{ctx}} = 0.05$)*.
  - **Anti-Veto Safeguards:**
    - If $S_{\text{ReID}} < 0.40$, final score capped at $0.45$ (color agreement cannot override poor ReID).
    - If $S_{\text{ReID}} \ge 0.85$, final score guaranteed at least $0.95 \cdot S_{\text{ReID}}$ (attribute noise cannot block pristine ReID).
  - **Gate Multiplier:** If not in pure ReID test mode, candidate score is adjusted:
    $$S_{\text{adjusted}} = S_{\text{composite}} \cdot (0.92 + 0.04 \cdot S_{\text{temporal}} + 0.04 \cdot S_{\text{spatial}})$$
  - **Decision Engine Thresholds:**
    - **Immediate Confirmation:** If $S_{\text{top}} \ge \text{threshold\_match}$ ($0.70$) and margin over second candidate $\ge \text{reid\_ambiguous\_margin}$ ($0.04$) (or transition-backed $S_{\text{top}} \ge 0.65$), assigned `MATCHED` immediately.
    - **Pending Promotion:** If $S_{\text{top}} \ge \text{threshold\_ambiguous}$ ($0.60$) or cross-view $S_{\text{top}} \ge \text{threshold\_cross\_view}$ ($0.55$), enters `PENDING`. Promoted to `MATCHED` after $\text{pending\_confirm\_frames}$ (3) consecutive checks.
    - **New Global ID Creation:** If candidates are rejected or similarity remains below threshold for $\text{reid\_new\_id\_confirm\_frames}$ (15) consecutive checks (~45 frames / 1.8s), registers a new Global ID (`Person_00X`).
  - **Hysteresis ID Switching & Identity Lock:**
    - If `lock_confirmed_id: true`, an assigned identity is locked and never changed ([association_manager.py:85-87](file:///c:/Users/Agira/Desktop/cctv-tracker/app/association/association_manager.py#L85-L87)).
    - If unlocked, switching requires a competing candidate to exceed $\text{hysteresis\_switch\_threshold}$ ($0.92$) with margin $\ge 0.12$ for $\ge 10$ consecutive frames.
- **Configuration Parameters:**
  - `weight_reid`: 0.80
  - `weight_par`: 0.15
  - `weight_context`: 0.05
  - `threshold_match`: 0.70
  - `threshold_ambiguous`: 0.60
  - `threshold_cross_view`: 0.55
  - `reid_ambiguous_margin`: 0.04
  - `pending_confirm_frames`: 3
  - `reid_new_id_confirm_frames`: 15
  - `lock_confirmed_id`: true
  - `hysteresis_switch_threshold`: 0.92
  - `hysteresis_switch_margin`: 0.12
  - `hysteresis_consecutive_frames`: 10

---

### 2.7 Floor-Plan Event Detectors (ROIs & Transitions)
- **Modules:** [app/events/transition_detector.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/transition_detector.py), [app/events/roi_detector.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/events/roi_detector.py)
- **Purpose:** Detects physical tripwire line crossings and polygon zone occupancy using bounding box foot points.
- **Implemented Algorithms & Logic:**
  - **Foot-Point Anchor:** Computes $f_x = \frac{x_1 + x_2}{2}, f_y = y_2$ representing person's floor contact.
  - **Transition Line Crossing:**
    - Uses 2D vector cross-product against line $(x_1, y_1) \rightarrow (x_2, y_2)$:
      $$d = (x_2 - x_1)(f_y - y_1) - (y_2 - y_1)(f_x - x_1)$$
      $d > 10^{-4} \rightarrow \text{SIDE\_A}$, $d < -10^{-4} \rightarrow \text{SIDE\_B}$.
    - Detects side transition $\text{SIDE\_A\_TO\_SIDE\_B}$ or $\text{SIDE\_B\_TO\_SIDE\_A}$.
    - Requires minimum displacement $\ge 5\text{px}$ and enforces `transition_cooldown_frames: 15` to eliminate boundary jitter.
  - **Polygon ROI Detector:**
    - Checks foot point via OpenCV `cv2.pointPolygonTest(poly, (fx, fy), False) \ge 0`.
    - Temporal debouncing: Requires `roi_debounce_frames: 3` consecutive frames inside to emit `ROI_ENTER`, and 3 consecutive frames outside to emit `ROI_EXIT`.
    - Dwell time tracking: Computes total duration upon exit ($t_{\text{exit}} - t_{\text{enter}}$).
    - Evacuation on track loss: If a track disappears while marked inside an ROI, emits `ROI_EXIT` automatically.
- **Configuration Parameters:**
  - `transition_cooldown_frames`: 15
  - `roi_debounce_frames`: 3
  - `cameras.<cam_id>.transitions.<trans_id>.points`: `{start: [x1, y1], end: [x2, y2]}`
  - `cameras.<cam_id>.roi.<roi_id>.points`: List of polygon vertices `[[x1, y1], ...]`.

---

### 2.8 Journey Lifecycle State Machine & Observability
- **Modules:** [app/journey/models.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/models.py), [app/journey/journey_manager.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/journey_manager.py), [app/journey/journey_store.py](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/journey_store.py)
- **Purpose:** Tracks global identity status, manages cross-camera handoffs, buffers unconfirmed events, and provides structured explainability logs.
- **Implemented Algorithms & Logic:**
  - **Identity Lifecycle States:**
    - `NEW`: Newly generated identity.
    - `ACTIVE`: Currently detected and tracked in an active camera.
    - `TEMPORARILY_LOST`: Undetected for $> \text{track\_lost\_frames}$ (60 frames / 2.0s). Preserved for reacquisition.
    - `EXPIRED`: Unseen for $> \text{same\_camera\_reacquisition\_max\_seconds}$ (20–60s). Evicted from active pool.
  - **Concurrent Camera Mapping:** Maps `gid_to_camera_tracks: Dict[str, Dict[str, int]]`, maintaining independent active local track mappings per camera.
  - **Retroactive Event Attachment:** Transitions and ROI events occurring while a track is `PENDING` are buffered in `pending_events[(camera_id, tid)]`. When the track is confirmed to a Global ID, all buffered events are retroactively attached and logged to `journey.jsonl` ([journey_manager.py:413-438](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/journey_manager.py#L413-L438)).
  - **Simultaneous Conflict Detection:** If an identity is active in Camera A and confirmed in Camera B within $< 1.0\text{s}$ without an expected transition path, emits `GLOBAL_ID_CONFLICT` and rejects association ([journey_manager.py:145-167](file:///c:/Users/Agira/Desktop/cctv-tracker/app/journey/journey_manager.py#L145-L167)).
  - **Structured Append-Only JSONL Logging:**
    - `logs/journey.jsonl`: Chronological audit trail of lifecycle events (`TRACK_STARTED`, `TRANSITION_CROSSED`, `ROI_ENTER`, `ROI_EXIT`, `GLOBAL_ID_REACQUIRED`, `CAMERA_DISAPPEARED`, `GLOBAL_ID_EXPIRED`).
    - `logs/association_decisions.jsonl`: Comprehensive per-track explainability log detailing probe viewpoint, all candidate scores, gate rejections, and selection rationale.
- **Configuration Parameters:**
  - `track_lost_frames`: 60
  - `occlusion_buffer_frames`: 20
  - `journey_log_path`: `"logs/journey.jsonl"`
  - `association_log_path`: `"logs/association_decisions.jsonl"`

---

## 3. Proven Algorithms & Behaviors That MUST Be Preserved

During migration to the new architecture, the following proven algorithms and behaviors must be preserved intact:

| Component | Proven Algorithm / Heuristic | Implementation Rationale |
| :--- | :--- | :--- |
| **Feature Extraction** | OSNet (`osnet_x1_0`) with 512-D L2-normalized vectors | Superior generalization on facility CCTV compared to generic ImageNet backbones. |
| **Quality Filter** | Laplacian variance ($\ge 10.0$) + dimension gates ($45 \times 90$) | Effectively filters blurry motion blur and edge crops before polluting banks. |
| **Embedding Bank** | View-partitioned sub-banks (`FRONT`, `SIDE`, `BACK`) + diversity threshold ($0.90$) | Prevents viewpoint collapse and avoids redundant vectors from standing persons. |
| **Prototype Synthesis** | Unit-normalized mean centroid across observation bank | Cancels single-frame pose noise and produces stable probe representations. |
| **Gallery Comparison** | Max-cosine similarity against candidate's multi-view embeddings | Evaluates distinct viewpoints without diluting specialized angles. |
| **Physical Gating** | Two-stage candidate filtering (Topology $\rightarrow$ Temporal $\rightarrow$ Spatial) | Constrains search space from dozens of gallery IDs down to 1–2 physically plausible candidates. |
| **Evidence Buffering** | 3-frame observation collection before candidate matching | Prevents premature assignment on tracker birth artifacts or initial detector noise. |
| **Spatial Events** | 2D signed cross-product tripwires with foot-point anchor | Robust against bounding box height jitter; accurately detects entry/exit direction. |
| **ROI Containment** | `cv2.pointPolygonTest` with bottom-center foot point + 3-frame debounce | Eliminates bounding box perimeter boundary noise. |
| **Coordinate Transform**| Initialization-time scaling of tripwires/ROIs for cropped video feeds | Preserves sub-pixel alignment when running crops on high-resolution cameras. |

---

## 4. Exhaustive Configuration Dictionary

| Parameter Key | Consuming Component | Default Value | Purpose & Behavioral Impact |
| :--- | :--- | :--- | :--- |
| `model_path` | `MultiCameraProcessor` | `"yolo11n.pt"` | Person detector model weights. |
| `tracker_config` | `MultiCameraProcessor` | `"botsort.yaml"` | Local tracker parameters (Kalman filter, track buffer). |
| `detection_confidence`| `MultiCameraProcessor` | `0.25` | Minimum YOLO bounding box confidence. |
| `imgsz` | `MultiCameraProcessor` | `960` | Detector inference canvas dimension. |
| `reid_enabled` | `MultiCameraProcessor` | `true` | Master switch for ReID appearance extraction. |
| `reid_model_path` | `OSNetExtractor` | `"osnet_x1_0"` | ReID backbone architecture. |
| `reid_interval` | `MultiCameraProcessor` | `3` | Frame interval for feature extraction (~8–10 Hz). |
| `min_reid_confidence` | `calculate_quality` | `0.35` | Minimum detection confidence to attempt ReID extraction. |
| `min_reid_width` | `calculate_quality` | `45` | Minimum crop width in pixels. |
| `min_reid_height` | `calculate_quality` | `90` | Minimum crop height in pixels. |
| `blur_threshold` | `calculate_quality` | `10.0` | Minimum Laplacian variance to reject motion blur. |
| `reid_bank_size` | `TrackEmbeddingBank` | `15` | Maximum vectors retained per viewpoint sub-bank. |
| `embedding_diversity_threshold` | `TrackEmbeddingBank` | `0.90` | Max cosine similarity to existing vector before rejecting duplicate. |
| `reid_confirm_frames` | `AssociationManager` | `3` | Number of observation vectors required before query. |
| `gallery_update_min_quality` | `MultiCameraProcessor` | `0.60` | Minimum crop quality required to admit vector into gallery. |
| `weight_reid` | `GlobalMatcher` | `0.80` | Weight of OSNet appearance in composite score. |
| `weight_par` | `GlobalMatcher` | `0.15` | Weight of PAR garment colors in composite score. |
| `weight_context` | `GlobalMatcher` | `0.05` | Weight of viewpoint/temporal context in composite score. |
| `threshold_match` | `AssociationManager` | `0.70` | Threshold for immediate confirmed association. |
| `threshold_ambiguous` | `AssociationManager` | `0.60` | Threshold for borderline match entering PENDING. |
| `threshold_cross_view`| `AssociationManager` | `0.55` | Threshold for cross-view candidate entering PENDING. |
| `reid_ambiguous_margin`| `AssociationManager` | `0.04` | Minimum margin over 2nd candidate for immediate match. |
| `pending_confirm_frames` | `AssociationManager` | `3` | Consecutive frames required to promote PENDING candidate. |
| `reid_new_id_confirm_frames` | `AssociationManager` | `15` | Consecutive checks without candidate before spawning new ID. |
| `lock_confirmed_id` | `AssociationManager` | `true` | Enforces permanent identity lock once assigned. |
| `hysteresis_switch_threshold`| `AssociationManager` | `0.92` | Score required for competing candidate to switch identity. |
| `hysteresis_switch_margin` | `AssociationManager` | `0.12` | Margin required for competing candidate to switch identity. |
| `hysteresis_consecutive_frames`| `AssociationManager`| `10` | Sustained frames required to confirm identity switch. |
| `track_lost_frames` | `MultiCameraProcessor` | `60` | Inactive frames before tracklet is marked TEMPORARILY_LOST. |
| `occlusion_buffer_frames` | `MultiCameraProcessor` | `20` | Grace frames during which dropped track holds GID exclusivity. |
| `transition_cooldown_frames` | `TransitionDetector` | `15` | Minimum frames between repeated tripwire events on same track. |
| `roi_debounce_frames` | `RoiDetector` | `3` | Consecutive frames required to confirm ROI enter or exit. |
| `exclusion_zones` | `MultiCameraProcessor` | `{...}` | Camera normalized boxes `[x1, y1, x2, y2]` to discard. |
| `track_only_inside_roi` | `MultiCameraProcessor` | `{...}` | List of ROIs outside of which tracks are discarded. |
| `clear_gallery_on_start`| `GlobalGallery` / `Qdrant` | `true` | Purges persistent storage on system initialization. |
