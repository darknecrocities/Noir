"""Crash recovery manager to detect and restore last valid checkpoint states."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from noir.core.exceptions import RecoveryError
from noir.core.logging import get_logger
from noir.storage.checkpoint_manager import CheckpointManager
from noir.storage.database import DatabaseManager, ExperimentModel

logger = get_logger("recovery")


class RecoveryManager:
    """Manages detection, verification, and recovery from previous sessions."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        checkpoint_manager: CheckpointManager,
        experiments_dir: str | Path = "experiments",
    ):
        self.db = db_manager
        self.checkpoint_manager = checkpoint_manager
        self.experiments_dir = Path(experiments_dir)

    def check_for_recovery(self) -> Optional[Dict[str, Any]]:
        """Inspects storage to find the most recent valid checkpoint across all runs.

        Returns:
            Dictionary with recovery details if a valid checkpoint is found, else None.
        """
        try:
            latest_exp_id: Optional[str] = None
            with self.db.get_session() as session:
                latest_exp = (
                    session.query(ExperimentModel)
                    .order_by(ExperimentModel.created_at.desc())
                    .first()
                )
                if latest_exp:
                    latest_exp_id = latest_exp.id

            # Check if checkpoint exists for latest experiment
            if latest_exp_id:
                ckpt = self.checkpoint_manager.get_latest_checkpoint(latest_exp_id)
                if ckpt:
                    meta_file = ckpt / "meta.json"
                    if meta_file.exists():
                        with open(meta_file, "r", encoding="utf-8") as f:
                            meta = json.load(f)
                            return {
                                "experiment_id": latest_exp_id,
                                "checkpoint_path": str(ckpt),
                                "step": meta.get("step", 0),
                                "epoch": meta.get("epoch", 0),
                                "timestamp": meta.get("timestamp", 0),
                                "metrics": meta.get("metrics", {}),
                                "emotion_state": meta.get("emotion_state", {}),
                                "config": meta.get("config", {}),
                            }

            # Fallback: scan all checkpoint directories on disk
            for exp_dir in sorted(self.checkpoint_manager.base_dir.iterdir(), reverse=True):
                if exp_dir.is_dir():
                    ckpt = self.checkpoint_manager.get_latest_checkpoint(exp_dir.name)
                    if ckpt:
                        meta_file = ckpt / "meta.json"
                        if meta_file.exists():
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                                return {
                                    "experiment_id": exp_dir.name,
                                    "checkpoint_path": str(ckpt),
                                    "step": meta.get("step", 0),
                                    "epoch": meta.get("epoch", 0),
                                    "timestamp": meta.get("timestamp", 0),
                                    "metrics": meta.get("metrics", {}),
                                    "emotion_state": meta.get("emotion_state", {}),
                                    "config": meta.get("config", {}),
                                }
        except Exception as e:
            logger.warning("Recovery scan encountered an issue: %s", e)

        return None
