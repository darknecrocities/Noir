"""Event store service for persisting and querying event records."""

import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional
from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_types import EventType

logger = get_logger("event_store")


class EventStore:
    """Persists events into SQLite database and provides temporal queries."""

    def __init__(self, db_connection_or_path: str):
        self.db_path = db_connection_or_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS events (
                        event_id TEXT PRIMARY KEY,
                        timestamp REAL NOT NULL,
                        event_type TEXT NOT NULL,
                        experiment_id TEXT NOT NULL,
                        training_step INTEGER NOT NULL,
                        epoch INTEGER NOT NULL,
                        payload_json TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_exp ON events(experiment_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_step ON events(training_step)")
                conn.commit()

    def save_event(self, event: NoirEvent) -> None:
        """Persist a single event to the database."""
        with self._lock:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO events
                        (event_id, timestamp, event_type, experiment_id, training_step, epoch, payload_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            event.event_id,
                            event.timestamp,
                            event.event_type.value,
                            event.experiment_id,
                            event.training_step,
                            event.epoch,
                            json.dumps(event.payload, default=str),
                        ),
                    )
                    conn.commit()
            except Exception as e:
                logger.error("Failed to save event %s: %s", event.event_id, e)

    def get_events(
        self,
        experiment_id: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[NoirEvent]:
        """Query events with optional filtering."""
        with self._lock:
            query = "SELECT event_id, timestamp, event_type, experiment_id, training_step, epoch, payload_json FROM events WHERE 1=1"
            params: List[Any] = []

            if experiment_id is not None:
                query += " AND experiment_id = ?"
                params.append(experiment_id)

            if event_type is not None:
                query += " AND event_type = ?"
                params.append(event_type.value)

            query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()

            results = []
            for row in rows:
                try:
                    payload = json.loads(row[6]) if row[6] else {}
                    results.append(
                        NoirEvent(
                            event_id=row[0],
                            timestamp=row[1],
                            event_type=EventType(row[2]),
                            experiment_id=row[3],
                            training_step=row[4],
                            epoch=row[5],
                            payload=payload,
                        )
                    )
                except Exception as e:
                    logger.warning("Failed to deserialize event row: %s", e)

            return results
