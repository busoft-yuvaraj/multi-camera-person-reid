# Generic Line Transition Violation Architecture Specification

## Overview & Design Rationale

In multi-camera surveillance and compliance tracking (such as handwashing hygiene or formal attire compliance), evaluating compliance **immediately upon detection in a new camera feed leads to severe false positives and premature alerts**. 

A person may enter the field of view of a camera (e.g., turning their head, standing in a doorway, or stepping back to fetch something) without actually intending to pass through that zone. 

### Why Line Transition Triggering?
1. **Physical Intent Verification:** A violation or validation is only recorded when the subject physically commits to traversing into a zone by crossing a defined spatial threshold.
2. **Foot-Point Precision:** Using the bottom-center of the bounding box ($x_{\text{foot}} = \frac{x_1 + x_2}{2}, y_{\text{foot}} = y_2$) ties the event strictly to the floor plane contact point, completely immune to head/torso bounding-box fluctuations and perspective parallax.
3. **Directional Awareness:** Vector cross-products distinguish between a person entering a restricted area ($A \to B$) versus backing away or returning ($B \to A$).
4. **Idempotent Decisioning:** Prevents repeated alert spamming while the person remains visible within the camera frame.

---

## System Architecture & End-to-End Flow

```mermaid
flowchart TD
    subgraph Camera1["Camera 1: Source / Prep Zone (e.g., Pantry / Sink Area)"]
        D1[Detection & Tracking<br/>YOLO + BoT-SORT] --> HW[Handwash Compliance Engine]
        HW --> PS[Person State / Global ID Bank<br/>Status: HANDWASHED / NOT_HANDWASHED]
        D1 --> F1[Foot Point Calculation<br/>fx = (x1+x2)/2, fy = y2]
        F1 --> L1{Crosses Exit Line?<br/>exit_transition_line}
    end

    subgraph Camera2["Camera 2: Destination / Monitored Zone (e.g., Passage / Clean Corridor)"]
        D2[Detection & Re-ID<br/>Resolved to Global ID] --> F2[Foot Point Calculation<br/>fx = (x1+x2)/2, fy = y2]
        F2 --> L2{Crosses Passage Line?<br/>passage_transition_line}
    end

    subgraph DecisionEngine["Compliance Decision Matrix"]
        L1 -- "Crossed Exit Line" --> Eval1[Evaluate Exit Policy]
        L2 -- "Crossed Passage Line" --> Eval2[Evaluate Passage Policy]

        PS -. Query Stored Status .-> Eval1
        PS -. Query Stored Status .-> Eval2

        Eval1 -- "NOT_HANDWASHED" --> V1_OK[VALID: Exit to Public / Lobby Allowed]
        Eval1 -- "HANDWASHED" --> V1_INFO[VALID: Leaving Area Compliant]

        Eval2 -- "HANDWASHED" --> V2_OK[VALID: Compliant Clean Zone Entry]
        Eval2 -- "NOT_HANDWASHED" --> V2_VIO[VIOLATION: Unsanitized Entry Alert & Evidence]
        Eval2 -- "UNKNOWN" --> V2_WAIT[HOLD: Pending Confirmation - No Premature Flag]
    end

    subgraph Evidence["Evidence & Audit Logging"]
        V2_VIO --> Snap[Save Cropped Evidence Frame + Overlay]
        V2_VIO --> Audit[Append to journey.jsonl & violation logs]
    end
```

---

## The Two Transition Lines & Business Rules

### 1. Exit Transition Line (`exit_transition_line`)
* **Camera / Zone:** Camera 1 (e.g., `pantry` / `wash_area` / `waiting_lobby`).
* **Location:** Positioned across the doorway, threshold, or boundary leading **away** from the wash area toward public/exit corridors.
* **Purpose:** Determines when a subject has officially completed their stay in the wash station and exited into the common lobby.
* **Evaluation Matrix:**
  * `NOT_HANDWASHED` + Crossed Exit Line $\longrightarrow$ **VALID (Non-Violation)**. (Subject did not enter the sterile/clean room; they simply returned to the common or public zone).
  * `HANDWASHED` + Crossed Exit Line $\longrightarrow$ **VALID (Compliant Exit)**.

### 2. Passage Transition Line (`passage_transition_line`)
* **Camera / Zone:** Camera 2 (e.g., `passage_1` / `sterile_corridor` / `production_entry`).
* **Location:** Positioned across the entrance threshold into the sensitive, clean, or restricted work zone.
* **Purpose:** Gatekeeper tripwire. Triggers compliance verification only when the person commits their step past this threshold.
* **Evaluation Matrix:**
  * `HANDWASHED` + Crossed Passage Line $\longrightarrow$ **VALID (Non-Violation)**. Verified clean entry into work zone.
  * `NOT_HANDWASHED` + Crossed Passage Line $\longrightarrow$ **VIOLATION**. Immediate incident generation, snapshot capture, and security alert.
  * `UNKNOWN` status $\longrightarrow$ **HOLD / PENDING**. Do not prematurely classify until identity/evidence settles.

---

## Foot-Point Geometry & Line Crossing Vector Math

### 1. Foot-Point Derivation
For any tracked bounding box $B = (x_1, y_1, x_2, y_2)$:
$$x_{\text{foot}} = \frac{x_1 + x_2}{2}, \quad y_{\text{foot}} = y_2$$

The bottom center corresponds directly to ground plane contact, eliminating vertical height distortion and perspective skew.

### 2. Directed Line Representation
A transition line is defined as a directed vector from starting point $A(x_1^L, y_1^L)$ to ending point $B(x_2^L, y_2^L)$.

### 3. Signed 2D Cross-Product Side Evaluation
To test whether foot-point $P(p_x, p_y)$ is on the left or right of the directed line $\vec{AB}$:
$$d = (x_2^L - x_1^L) \cdot (p_y - y_1^L) - (y_2^L - y_1^L) \cdot (p_x - x_1^L)$$

$$\text{Side}(P) = \begin{cases} 
\text{SIDE\_A} & \text{if } d > \epsilon \\
\text{SIDE\_B} & \text{if } d < -\epsilon \\
\text{COLINEAR} & \text{if } |d| \le \epsilon 
\end{cases}$$

*(where $\epsilon \approx 10^{-4}$ to suppress boundary flickering)*

### 4. Transition State Machine
For each tracked individual (by `local_track_id` or `global_id`):
1. On first frame observed: Record initial `side = Side(P)`. Do **not** trigger crossing.
2. On subsequent frames:
   $$\text{Crossing Triggered} \iff \text{previous\_side} \neq \text{current\_side} \land \text{current\_side} \neq \text{COLINEAR}$$
3. Confirm crossing direction:
   $$\text{Direction} = \text{previous\_side} \to \text{current\_side} \quad (\text{e.g., } \text{SIDE\_B\_TO\_SIDE\_A})$$
4. **Debounce & Distance Verification:**
   - Minimum pixel distance between consecutive foot-points ($\Delta \text{dist} \ge 5\text{ px}$) to prevent standing jitter.
   - Frame cooldown ($N \ge 15\text{ frames}$) to prevent multi-triggering.

---

## Generic Configuration Schema (YAML)

Add this modular configuration block into your system's `app_config.yaml`:

```yaml
# ============================================================
# TRANSITION LINE VIOLATION CONFIGURATION
# ============================================================

violation_detection:
  enabled: true
  min_movement_px: 5.0
  cooldown_frames: 15
  evidence_dir: "output/violations"
  save_evidence_snapshots: true

  # Per-Camera Transition Lines
  cameras:
    # --------------------------------------------------------
    # CAMERA 1: Wash Area / Exit Zone
    # --------------------------------------------------------
    waiting_lobby:
      zone: "waiting_lobby"
      pathway: "exit"
      transition_line:
        enabled: true
        line_id: "exit_transition_line"
        type: "line"
        # Coordinates in native camera resolution: [X, Y]
        start: [420, 265]
        end: [662, 265]
        trigger_direction: "SIDE_B_TO_SIDE_A" # Movement towards exit
        color: [0, 165, 255] # BGR orange-yellow

    # --------------------------------------------------------
    # CAMERA 2: Passage / Clean Corridor Entry
    # --------------------------------------------------------
    passage_1:
      zone: "passage_1"
      pathway: "passage"
      transition_line:
        enabled: true
        line_id: "passage_transition_line"
        type: "line"
        # Coordinates in native camera resolution: [X, Y]
        start: [550, 600]
        end: [800, 600]
        trigger_direction: "SIDE_B_TO_SIDE_A" # Movement entering passage
        color: [0, 0, 255] # BGR red tripwire

# ============================================================
# COMPLIANCE POLICY RULES
# ============================================================
compliance_policy:
  pathway_rules:
    # Rule for Passage Entry
    passage:
      HANDWASHED: "VALID"
      NOT_HANDWASHED: "VIOLATION"
      UNKNOWN: "HOLD"

    # Rule for Exit Corridor
    exit:
      HANDWASHED: "VALID"
      NOT_HANDWASHED: "VALID"
      UNKNOWN: "VALID"
```

---

## Generic Python Implementation Blueprint

Here is the clean, self-contained Python module ready to drop into any pipeline architecture:

```python
"""
transition_line_violation.py
Generic Line Transition Compliance & Violation Engine.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional, Any, Set
import math
import time
import logging

@dataclass
class LineCrossingEvent:
    camera_id: str
    line_id: str
    track_id: int
    global_id: Optional[str]
    timestamp: float
    previous_side: str
    current_side: str
    direction: str
    foot_point: Tuple[float, float]
    line_start: Tuple[int, int]
    line_end: Tuple[int, int]


class LineTransitionDetector:
    """
    Tracks foot-point line crossings per camera with debounce and movement validation.
    """
    def __init__(
        self,
        camera_id: str,
        line_id: str,
        start_pt: Tuple[int, int],
        end_pt: Tuple[int, int],
        trigger_direction: Optional[str] = None,
        cooldown_frames: int = 15,
        min_movement_px: float = 5.0
    ):
        self.camera_id = camera_id
        self.line_id = line_id
        self.start_pt = (int(start_pt[0]), int(start_pt[1]))
        self.end_pt = (int(end_pt[0]), int(end_pt[1]))
        self.trigger_direction = trigger_direction
        self.cooldown_frames = cooldown_frames
        self.min_movement_px = min_movement_px

        # State tracking: track_id -> side ("SIDE_A", "SIDE_B")
        self.track_sides: Dict[int, str] = {}
        self.last_points: Dict[int, Tuple[float, float]] = {}
        self.last_trigger_frame: Dict[int, int] = {}

    @staticmethod
    def get_foot_point(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculates bottom-center foot contact point."""
        x1, y1, x2, y2 = bbox[:4]
        return float((x1 + x2) / 2.0), float(y2)

    @staticmethod
    def calculate_side(point: Tuple[float, float], p1: Tuple[int, int], p2: Tuple[int, int]) -> str:
        """Computes signed 2D cross-product to determine half-plane."""
        x1, y1 = p1
        x2, y2 = p2
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 1e-4:
            return "SIDE_A"
        elif cross < -1e-4:
            return "SIDE_B"
        return "COLINEAR"

    def check_crossing(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        frame_idx: int,
        global_id: Optional[str] = None,
        timestamp: Optional[float] = None
    ) -> Optional[LineCrossingEvent]:
        foot_pt = self.get_foot_point(bbox)
        current_side = self.calculate_side(foot_pt, self.start_pt, self.end_pt)

        # Ignore ambiguous colinear points
        if current_side == "COLINEAR":
            return None

        # First observation: seed initial side
        if track_id not in self.track_sides:
            self.track_sides[track_id] = current_side
            self.last_points[track_id] = foot_pt
            return None

        prev_side = self.track_sides[track_id]

        if current_side != prev_side:
            # 1. Cooldown verification
            last_frame = self.last_trigger_frame.get(track_id, -9999)
            if frame_idx - last_frame < self.cooldown_frames:
                self.last_points[track_id] = foot_pt
                return None

            # 2. Minimum movement verification
            last_pt = self.last_points.get(track_id, foot_pt)
            dist = math.hypot(foot_pt[0] - last_pt[0], foot_pt[1] - last_pt[1])
            if dist < self.min_movement_px:
                self.last_points[track_id] = foot_pt
                return None

            # Crossing confirmed!
            direction = f"{prev_side}_TO_{current_side}"
            self.track_sides[track_id] = current_side
            self.last_points[track_id] = foot_pt
            self.last_trigger_frame[track_id] = frame_idx

            # If specific trigger direction is enforced, check it
            if self.trigger_direction and direction != self.trigger_direction:
                return None

            return LineCrossingEvent(
                camera_id=self.camera_id,
                line_id=self.line_id,
                track_id=track_id,
                global_id=global_id,
                timestamp=timestamp or time.time(),
                previous_side=prev_side,
                current_side=current_side,
                direction=direction,
                foot_point=foot_pt,
                line_start=self.start_pt,
                line_end=self.end_pt
            )

        self.last_points[track_id] = foot_pt
        return None

    def reset_track(self, track_id: int):
        self.track_sides.pop(track_id, None)
        self.last_points.pop(track_id, None)
        self.last_trigger_frame.pop(track_id, None)


class LineTransitionViolationEngine:
    """
    Evaluates crossing events against person handwash status and pathway policy.
    """
    def __init__(self, policy_config: Dict[str, Any]):
        self.pathway_rules = policy_config.get("pathway_rules", {})
        self.recorded_events: Set[Tuple[str, str, str]] = set() # (camera, global_or_track_id, line_id)

    def evaluate_crossing(
        self,
        event: LineCrossingEvent,
        pathway: str,
        handwash_status: str
    ) -> Dict[str, Any]:
        """
        Calculates compliance status: VALID, VIOLATION, or HOLD.
        """
        entity_key = event.global_id or f"track_{event.track_id}"
        event_sig = (event.camera_id, entity_key, event.line_id)

        # Prevent duplicate triggering for the same individual across identical line
        if event_sig in self.recorded_events:
            return {"status": "ALREADY_RECORDED", "is_violation": False}

        status = (handwash_status or "UNKNOWN").upper().strip()
        rules_for_pathway = self.pathway_rules.get(pathway.lower(), {})
        compliance_outcome = rules_for_pathway.get(status, "HOLD")

        if compliance_outcome == "HOLD":
            return {
                "status": "HOLD",
                "is_violation": False,
                "reason": "handwash_status_unknown_wait_for_evidence"
            }

        # Mark as recorded
        self.recorded_events.add(event_sig)

        is_violation = (compliance_outcome == "VIOLATION")
        return {
            "status": compliance_outcome,
            "is_violation": is_violation,
            "camera_id": event.camera_id,
            "line_id": event.line_id,
            "entity_id": entity_key,
            "handwash_status": status,
            "pathway": pathway,
            "direction": event.direction,
            "timestamp": event.timestamp,
            "reason": f"pathway_{pathway}_{status.lower()}_{compliance_outcome.lower()}"
        }
```

---

## Camera Processing Loop Integration Hook

In your TL's camera processing loop (inside `_process_frame` or equivalent per-camera iteration):

```python
# -------------------------------------------------------------------------
# Per-Track Iteration within Camera Frame Loop
# -------------------------------------------------------------------------
for track in tracks:
    x1, y1, x2, y2, local_tid, conf, cls = track
    bbox = (x1, y1, x2, y2)

    # 1. Resolve Global ID (if re-identification is active)
    global_id = local_to_global_id_map.get(local_tid)

    # 2. Check Virtual Line Crossing (ONLY foot touch/crossing triggers evaluation)
    if camera_id in line_detectors:
        detector = line_detectors[camera_id]
        crossing_event = detector.check_crossing(
            track_id=local_tid,
            bbox=bbox,
            frame_idx=frame_idx,
            global_id=global_id,
            timestamp=current_timestamp
        )

        if crossing_event:
            # 3. Retrieve persistent handwash status for this person
            person_handwash_status = person_state_bank.get_handwash_status(global_id)

            # 4. Evaluate violation against transition policy
            result = violation_engine.evaluate_crossing(
                event=crossing_event,
                pathway=camera_pathway, # e.g., "passage" or "exit"
                handwash_status=person_handwash_status
            )

            # 5. Handle Outcome
            if result.get("is_violation"):
                logging.warning(f"🚨 VIOLATION: Person {global_id} crossed {crossing_event.line_id} without handwashing!")
                save_violation_evidence(frame=frame, bbox=bbox, event=result)
            elif result.get("status") == "VALID":
                logging.info(f"✅ COMPLIANT: Person {global_id} crossed {crossing_event.line_id} successfully.")
```

---

## Verification & Test Scenarios

| Test Case | Camera / Line | Handwash State | Expected Result | Rationale |
|:---|:---|:---|:---|:---|
| **Scenario 1** | `passage_1` (`passage_transition_line`) | `HANDWASHED` | **VALID** (Non-violation) | Compliant entry into sanitized zone. |
| **Scenario 2** | `passage_1` (`passage_transition_line`) | `NOT_HANDWASHED` | **VIOLATION** | Breach of hygiene barrier into clean corridor. |
| **Scenario 3** | `waiting_lobby` (`exit_transition_line`) | `NOT_HANDWASHED` | **VALID** (Non-violation) | Person left the facility without entering sanitized corridor. |
| **Scenario 4** | `passage_1` (Subject stands inside FOV without crossing line) | `NOT_HANDWASHED` | **NO ACTION** | Subject has not touched/crossed line; no violation triggered. |
| **Scenario 5** | `passage_1` (`passage_transition_line`) | `UNKNOWN` | **HOLD** | Identity or status pending; suppresses false positive alarm. |
| **Scenario 6** | `passage_1` (Walking backwards: $A \to B$) | `NOT_HANDWASHED` | **NO ACTION** | Opposite direction of sensitive trigger wire. |
