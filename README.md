# Multi-Camera CCTV Person Re-Identification & Journey Tracking (Phase 1)

An enterprise multi-camera CCTV tracking and person re-identification (Re-ID) system designed for facility monitoring (e.g., waiting lobbies, pantries, passages). Built on **YOLOv11**, **BoT-SORT**, **OSNet (Omni-Scale Network)**, and **Qdrant Vector Database**, augmented with **floor-plan-aware spatial-temporal topology gates**.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Capture["1. Synchronized Multi-Camera Capture"]
        C1[Camera 1: Waiting Lobby] --> Det1[YOLOv11 Detection + BoT-SORT Tracking]
        C2[Camera 2: Pantry] --> Det2[YOLOv11 Detection + BoT-SORT Tracking]
        C3[Camera 3: Passage] --> Det3[YOLOv11 Detection + BoT-SORT Tracking]
    end

    subgraph Feature["2. Feature Extraction & Embedding Bank"]
        Det1 & Det2 & Det3 --> Quality[Quality & Blur Filter]
        Quality --> OSNet[OSNet 512-D Appearance Extraction]
        OSNet --> Bank[Temporal Diversity Embedding Bank (15 vectors/track)]
    end

    subgraph SpatialEvents["3. Floor-Plan Spatial Event Detection"]
        Det1 & Det2 & Det3 --> TransDet[Line Crossing Transition Detector]
        Det1 & Det2 & Det3 --> RoiDet[Polygon ROI / Zone Detector]
        TransDet --> JourneyMgr[Journey Manager]
        RoiDet --> JourneyMgr
    end

    subgraph Association["4. Two-Stage Global ID Association"]
        Bank --> CandFilter[Stage 1: Candidate Filter Engine]
        JourneyMgr -.->|Live Identity States| CandFilter
        
        subgraph Stage1["Stage 1: Physical Gates"]
            CandFilter --> Gate1[Topology Gate: Physical Connectivity]
            Gate1 --> Gate2[Temporal Gate: Transit Time Bounds]
            Gate2 --> Gate3[Spatial Gate: Transition Line Crossings]
        end
        
        Gate3 --> ValidPool[Filtered Pool: 1-2 Valid Candidates]
        
        subgraph Stage2["Stage 2: Identity Verification"]
            ValidPool --> Qdrant[Candidate-Restricted Qdrant Vector Search]
            Qdrant --> ReIDScore[Pure OSNet Cosine Similarity Scoring]
            ReIDScore --> DecisionTree{Score Thresholds}
        end
    end

    subgraph Decision["5. Decision & Lifecycle"]
        DecisionTree -->|Score >= 0.70| Confirmed[CONFIRMED: Assigned Global ID]
        DecisionTree -->|0.55 <= Score < 0.70| Pending[PENDING: Multi-Frame Evidence Accumulator]
        Pending -->|Sustained 3 frames| Confirmed
        DecisionTree -->|< 0.55 or Empty Candidates| NoMatch[No Match Counter]
        NoMatch -->|>= 15 Frames Timeout| NewID[Register New Global ID in Qdrant & Gallery]
    end

    Confirmed --> Audit[Audit Trail: logs/journey.jsonl & logs/association_decisions.jsonl]
    NewID --> Audit
```

---

## Detailed Pipeline Flow

### 1. Multi-Camera Synchronized Tracking
- **Detection & Tracking:** Each camera runs independent YOLOv11 person detection paired with BoT-SORT local tracking (`tracker_config: "botsort.yaml"`).
- **Coordinate Space:** Cameras can define independent crops (e.g., `crop: [0, 0, 720, 720]` for pantry zoom) and polygon regions of interest (sink, food counter, etc.).
- **Local Track IDs:** Each camera produces local IDs (`track_id: 1, 2, ...`) that are ephemeral and local to that specific camera view.

### 2. Feature Extraction & Embedding Bank
- **Bounding Box Quality Filter:** Detections below minimum dimensions (`45x90px`), confidence (`0.35`), or image sharpness (`blur_threshold: 10.0`) are excluded to avoid noisy vectors.
- **OSNet (Omni-Scale Feature Extractor):** Extracts 512-dimensional L2-normalized deep appearance embeddings capturing body silhouette, clothing patterns, texture, and footwear.
- **Embedding Bank:** Accumulates up to 15 diverse embeddings per local track based on cosine diversity (`embedding_diversity_threshold: 0.90`). Once $\ge 3$ quality frames are collected, an averaged track prototype embedding is synthesized.

### 3. Spatial & Floor-Plan Event Monitoring
- **Transition Lines:** Directional tripwires detect entries and exits between adjacent zones (e.g., `pantry_green` crossing into `passage`, `pantry_yellow` into `waiting_lobby`).
- **Direction Awareness:** Line-crossing utilizes vector 2D cross-products to determine direction (`SIDE_A_TO_SIDE_B` vs `SIDE_B_TO_SIDE_A`), identifying the intended destination camera.
- **ROI / Zone Debouncing:** Debounced polygon inclusion tracks dwell time and area occupancy.

### 4. Stage 1: Physical Candidate Filtering
Rather than performing a global search across all identities in the building, the **Candidate Filter** applies hard spatio-temporal constraints based on the facility layout:
1. **Active Occupancy Check & Handover Bypass:** Excludes identities currently active in other cameras, **except** when the person recently crossed a transition line pointing directly to the current camera (preventing false lockout during camera handovers).
2. **Topology Gate:** Validates whether the transition between the candidate's last known zone and the current zone is physically possible according to the topology graph.
3. **Temporal Gate:** Enforces travel speed bounds (e.g., transit between pantry and passage must take between $0.5\text{s}$ and $45\text{s}$). Faster is rejected (teleportation), slower is rejected (stale identity).
4. **Spatial Gate:** Grants priority when a candidate's recorded line-crossing matches the physical entryway.

> **Result:** Shrinks a pool of dozens of identities down to **only 1 or 2 physically plausible candidates**.

### 5. Stage 2: Pure Re-ID Verification & Promotion
Once the candidate set is filtered:
- **Vector Search:** Queries the Qdrant Cloud vector database restricted strictly to the valid candidate IDs.
- **Pure ReID Appearance Scoring:** Evaluates cosine similarity between the probe prototype and the candidate's stored multi-view embeddings (`weight_reid: 1.00`, `weight_par: 0.00`, `weight_context: 0.00`). By relying directly on the 512-D deep embedding without color-classifier attribute noise, matching remains robust across varying camera white balances and lighting conditions.
- **Decision Engine:**
  - **Immediate Confirmation ($\text{Score} \ge 0.70$):** Instantly binds the local track to the existing Global ID if margin over the second candidate is $\ge 0.04$. If supported by an explicit line transition, confirmation threshold lowers to $\ge 0.65$.
  - **Pending Promotion ($0.55 \le \text{Score} < 0.70$):** Holds track in `PENDING` state. If sustained over 3 consecutive frames (`pending_confirm_frames: 3`), the track is promoted to confirmed match.
  - **New Global ID Creation:** If candidates are rejected or similarity remains below threshold for 15 consecutive frames (`reid_new_id_confirm_frames: 15`), the system registers a brand-new Global ID (`Person_00X`) in Qdrant and the global gallery.

### 6. Journey Store & Audit Trail
Every state transition is written in real time to append-only JSONL files:
- **[`logs/journey.jsonl`](file:///logs/journey.jsonl):** Logs lifecycle events: `TRACK_STARTED`, `TRANSITION_CROSSED`, `ZONE_ENTER`, `ZONE_EXIT`, `TRACK_LOST`.
- **[`logs/association_decisions.jsonl`](file:///logs/association_decisions.jsonl):** Comprehensive audit trail containing frame index, candidate evaluations, rejection reasons (`EXCLUDED_OCCUPIED_IN_CAMERA`, `TOPOLOGY_REJECT`, etc.), similarity scores, and final association verdicts.

---

## Directory Structure

```
cctv-tracker/
├── app/
│   ├── association/             # Stage 1 & Stage 2 matching engines
│   │   ├── association_manager.py  # Orchestrates candidate evaluation & lifecycle
│   │   ├── candidate_filter.py     # Multi-gate filter pipeline
│   │   ├── topology.py             # Floor-plan connectivity graph
│   │   ├── temporal_gate.py        # Transit duration bounds
│   │   └── spatial_gate.py         # Line-crossing spatial correlation
│   ├── events/                  # Physical boundary detectors
│   │   ├── transition_detector.py  # Directional line crossing
│   │   └── roi_detector.py         # Polygon zone dwell & occupancy
│   ├── journey/                 # Global identity state management
│   │   ├── journey_manager.py      # Cross-camera identity tracking & handoffs
│   │   ├── journey_store.py        # JSONL audit log writer
│   │   └── models.py               # GlobalIdentityState, JourneyEvent models
│   ├── reid/                    # Feature extraction & vector storage
│   │   ├── extractor.py            # OSNet PyTorch wrapper
│   │   ├── matcher.py              # Candidate similarity & Hungarian matching
│   │   ├── qdrant_gallery.py       # Qdrant Cloud vector search integration
│   │   ├── embedding_bank.py       # Per-track temporal feature accumulator
│   │   └── global_gallery.py       # In-memory prototype gallery
│   └── multi_camera_processor.py # Main multi-stream synchronizer & renderer
├── config/
│   ├── app_config.yaml          # Base configuration (root CCTV samples)
│   └── app_config_new.yaml      # Configuration for new test sets (e.g. test3)
├── logs/                        # Runtime audit logs & journey history
├── main.py                      # Application entry point
└── requirements.txt             # Project dependencies
```

---

## Key Configuration Parameters

| Parameter | Recommended Value | Purpose |
|---|---|---|
| `weight_reid` | `1.00` | Pure appearance weighting using OSNet 512-D embeddings |
| `weight_par` | `0.00` | Disabled attribute classification to avoid camera lighting bias |
| `weight_context` | `0.00` | Context handled strictly upfront by CandidateFilter |
| `threshold_match` | `0.70` | Cosine similarity threshold for immediate identity confirmation |
| `threshold_ambiguous` | `0.60` | Threshold to enter multi-frame `PENDING` state |
| `threshold_cross_view` | `0.55` | Floor threshold for cross-camera view adjustments |
| `pending_confirm_frames` | `3` | Number of consecutive matching frames to confirm a pending candidate |
| `reid_new_id_confirm_frames` | `15` | Frame timeout with no candidate match before registering a new Global ID |
| `occlusion_buffer_frames` | `20` | Grace frames to reacquire an identity temporarily lost behind occlusion |

---

## Setup & Running

### 1. Environment Setup
```bash
# Activate virtual environment
.venv\Scripts\activate   # Windows
# or: source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Variables (`.env`)
Create a `.env` file with your Qdrant vector database credentials:
```env
QDRANT_URL=https://<your-cluster-id>.us-east-1-1.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=<your-qdrant-api-key>
```

### 3. Running the Pipeline

```bash
# Run with standard CCTV dataset
python main.py --config config/app_config.yaml

# Run with test dataset (test3 trimmed samples)
python main.py --config config/app_config_new.yaml

# Run with a frame limit (e.g., 300 frames for quick testing)
python main.py --config config/app_config_new.yaml --max-frames 300
```
