"""Repository for storing and querying training metrics."""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from sqlalchemy.orm import Session

from noir.core.logging import get_logger
from noir.storage.database import DatabaseManager, MetricModel

logger = get_logger("metrics_repository")


class MetricsRepository:
    """Handles time-series storage and queries for training and evaluation metrics."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._flush_batch_size = 50

    def log_metric(
        self,
        experiment_id: str,
        step: int,
        metric_name: str,
        metric_value: float,
        epoch: int = 0,
        flush_immediately: bool = False,
    ) -> None:
        """Record a single metric value."""
        entry = {
            "experiment_id": experiment_id,
            "step": step,
            "epoch": epoch,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "timestamp": time.time(),
        }
        with self._lock:
            self._buffer.append(entry)
            should_flush = flush_immediately or len(self._buffer) >= self._flush_batch_size

        if should_flush:
            self.flush()

    def log_metrics_dict(
        self,
        experiment_id: str,
        step: int,
        metrics: Dict[str, float],
        epoch: int = 0,
        flush_immediately: bool = False,
    ) -> None:
        """Record multiple metrics at the same step."""
        now = time.time()
        entries = [
            {
                "experiment_id": experiment_id,
                "step": step,
                "epoch": epoch,
                "metric_name": k,
                "metric_value": float(v),
                "timestamp": now,
            }
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ]
        with self._lock:
            self._buffer.extend(entries)
            should_flush = flush_immediately or len(self._buffer) >= self._flush_batch_size

        if should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush in-memory buffer to SQLite."""
        with self._lock:
            if not self._buffer:
                return
            to_insert = list(self._buffer)
            self._buffer.clear()

        try:
            with self.db.get_session() as session:
                models = [MetricModel(**entry) for entry in to_insert]
                session.add_all(models)
                session.commit()
        except Exception as e:
            logger.error("Failed to flush metrics to database: %s", e)

    def get_metric_history(
        self,
        experiment_id: str,
        metric_name: str,
        limit: int = 1000,
    ) -> List[Tuple[int, float]]:
        """Retrieve (step, value) pairs for a specific metric."""
        self.flush()
        with self.db.get_session() as session:
            rows = (
                session.query(MetricModel.step, MetricModel.metric_value)
                .filter(
                    MetricModel.experiment_id == experiment_id,
                    MetricModel.metric_name == metric_name,
                )
                .order_by(MetricModel.step.asc())
                .limit(limit)
                .all()
            )
            return [(r.step, r.metric_value) for r in rows]

    def get_latest_metrics(self, experiment_id: str) -> Dict[str, float]:
        """Get the most recent values for all recorded metric names for an experiment."""
        self.flush()
        with self.db.get_session() as session:
            # Query distinct metric names and their latest recorded value
            rows = (
                session.query(MetricModel.metric_name, MetricModel.metric_value)
                .filter(MetricModel.experiment_id == experiment_id)
                .order_by(MetricModel.step.desc())
                .limit(50)
                .all()
            )
            result: Dict[str, float] = {}
            for name, val in rows:
                if name not in result:
                    result[name] = val
            return result
