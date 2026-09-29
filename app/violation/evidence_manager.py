import os
import json
import datetime
import cv2
import numpy as np
from typing import Optional, Dict, Any, Tuple

class EvidenceManager:
    """
    Manages capturing, annotating, and persisting evidence images and JSON records
    for detected route/attire violations.
    """
    def __init__(self, output_dir: str = "output/violations", enabled: bool = True, save_image: bool = True, save_metadata: bool = True):
        self.output_dir = output_dir
        self.enabled = enabled
        self.save_image = save_image
        self.save_metadata = save_metadata

    def _get_daily_dir(self, timestamp_str: Optional[str] = None) -> str:
        if timestamp_str:
            try:
                date_part = timestamp_str.split("T")[0]
            except Exception:
                date_part = datetime.datetime.now().strftime("%Y-%m-%d")
        else:
            date_part = datetime.datetime.now().strftime("%Y-%m-%d")

        daily_path = os.path.join(self.output_dir, date_part)
        os.makedirs(daily_path, exist_ok=True)
        return daily_path

    def save_evidence(
        self,
        event,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        line_coords: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    ) -> Optional[str]:
        """
        Saves annotated violation frame and writes metadata record.
        Returns the saved image file path.
        """
        if not self.enabled:
            return None

        daily_dir = self._get_daily_dir(event.timestamp)
        gid_clean = str(event.global_id or "GID_NONE").replace(" ", "_")
        attire_clean = str(event.actual_attire).upper()
        zone_clean = str(event.zone).upper()

        image_filename = f"{event.event_id}_{gid_clean}_{attire_clean}_{zone_clean}.jpg"
        image_path = os.path.join(daily_dir, image_filename)
        event.evidence_image = image_path

        if self.save_image and frame is not None and frame.size > 0:
            annotated_frame = frame.copy()
            h, w = annotated_frame.shape[:2]

            # 1. Draw virtual line if provided
            if line_coords:
                p1, p2 = line_coords
                cv2.line(annotated_frame, p1, p2, (255, 255, 0), 3)
                cv2.circle(annotated_frame, p1, 5, (255, 255, 0), -1)
                cv2.circle(annotated_frame, p2, 5, (255, 255, 0), -1)
                cv2.putText(
                    annotated_frame, "Virtual Line", (p1[0] + 5, p1[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
                )

            # 2. Draw person bounding box
            x1, y1, x2, y2 = map(int, bbox[:4])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(
                annotated_frame, "VIOLATION", (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2
            )

            # 3. Draw structured HUD overlay card
            overlay_lines = [
                "VIOLATION",
                f"Global ID: {event.global_id or 'UNKNOWN'}",
                f"Track ID: {event.local_track_id}",
                f"Attire: {event.actual_attire.upper()}",
                f"Expected: {event.expected_attire.upper() if event.expected_attire else 'NONE'}",
                f"Camera: {event.camera_id.upper()}",
                f"Time: {event.timestamp.split('T')[-1][:8] if 'T' in event.timestamp else event.timestamp}"
            ]

            card_x, card_y = 20, 20
            card_w, card_h = 320, 20 + len(overlay_lines) * 26
            
            # Semi-transparent dark background for HUD
            sub_img = annotated_frame[card_y:card_y+card_h, card_x:card_x+card_w]
            black_rect = np.zeros_like(sub_img, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.25, black_rect, 0.75, 1.0)
            annotated_frame[card_y:card_y+card_h, card_x:card_x+card_w] = res
            cv2.rectangle(annotated_frame, (card_x, card_y), (card_x+card_w, card_y+card_h), (0, 0, 255), 2)

            for i, line_text in enumerate(overlay_lines):
                line_color = (0, 0, 255) if i == 0 else (255, 255, 255)
                scale = 0.7 if i == 0 else 0.55
                thickness = 2 if i == 0 else 1
                cv2.putText(
                    annotated_frame, line_text, (card_x + 12, card_y + 24 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, line_color, thickness
                )

            cv2.imwrite(image_path, annotated_frame)

        if self.save_metadata:
            # Save single event JSON
            single_json_path = os.path.join(daily_dir, f"{event.event_id}.json")
            with open(single_json_path, "w") as f:
                json.dump(event.to_dict(), f, indent=2)

            # Append to daily events.json
            daily_events_path = os.path.join(daily_dir, "events.json")
            existing_events = []
            if os.path.exists(daily_events_path):
                try:
                    with open(daily_events_path, "r") as f:
                        existing_events = json.load(f)
                except Exception:
                    existing_events = []
            
            existing_events.append(event.to_dict())
            with open(daily_events_path, "w") as f:
                json.dump(existing_events, f, indent=2)

        return image_path
