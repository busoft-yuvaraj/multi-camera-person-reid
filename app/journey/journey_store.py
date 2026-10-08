import os
import json
import logging
import threading
from typing import Dict, List, Optional, Any
from .models import JourneyEvent

class JourneyStore:
    """
    Abstract structured storage for Phase 1 journey events and diagnostic association logs.
    Persists events as append-only JSONL files:
        - logs/journey.jsonl: chronological events for every Global ID.
        - logs/association_decisions.jsonl: explainable decisions for each local-track assignment.
    """
    def __init__(
        self,
        journey_log_path: str = "logs/journey.jsonl",
        association_log_path: str = "logs/association_decisions.jsonl"
    ):
        self.journey_log_path = journey_log_path
        self.association_log_path = association_log_path
        self.lock = threading.Lock()
        
        # Ensure log directories exist
        os.makedirs(os.path.dirname(self.journey_log_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.association_log_path) or ".", exist_ok=True)

        # In-memory index of events by global_id for fast retrieval
        self._events_by_gid: Dict[str, List[Dict[str, Any]]] = {}

    def log_event(self, event: JourneyEvent):
        data = event.to_dict()
        gid = event.global_id or "PENDING"
        
        with self.lock:
            if gid not in self._events_by_gid:
                self._events_by_gid[gid] = []
            self._events_by_gid[gid].append(data)

            try:
                with open(self.journey_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(data) + "\n")
            except Exception as e:
                logging.error(f"[JOURNEY STORE] Failed to write event to {self.journey_log_path}: {e}")

    def log_association_decision(self, decision_data: Dict[str, Any]):
        """
        Records an explainable association decision answering:
        'Why was this track assigned Global ID X?'
        """
        with self.lock:
            try:
                with open(self.association_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(decision_data) + "\n")
            except Exception as e:
                logging.error(f"[JOURNEY STORE] Failed to write decision to {self.association_log_path}: {e}")

    def get_journey(self, global_id: str) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self._events_by_gid.get(global_id, []))
