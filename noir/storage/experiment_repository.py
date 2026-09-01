"""Repository for managing experiments, experiment branching, and comparison."""

import json
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from sqlalchemy.orm import Session

from noir.core.exceptions import StorageError
from noir.core.logging import get_logger
from noir.storage.database import DatabaseManager, ExperimentModel

logger = get_logger("experiment_repository")


class ExperimentRepository:
    """Handles experiment persistence, tree branching, and comparisons."""

    def __init__(self, db_manager: DatabaseManager, experiments_dir: str | Path = "experiments"):
        self.db = db_manager
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def create_experiment(
        self,
        name: str,
        config: Dict[str, Any],
        experiment_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        description: Optional[str] = None,
    ) -> ExperimentModel:
        """Create and initialize a new experiment record and directory structure."""
        exp_id = experiment_id or f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        exp_path = self.experiments_dir / exp_id
        exp_path.mkdir(parents=True, exist_ok=True)
        (exp_path / "checkpoints").mkdir(parents=True, exist_ok=True)
        (exp_path / "metrics").mkdir(parents=True, exist_ok=True)
        (exp_path / "events").mkdir(parents=True, exist_ok=True)
        (exp_path / "memory").mkdir(parents=True, exist_ok=True)

        with open(exp_path / "config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)

        with self.db.get_session() as session:
            exp = ExperimentModel(
                id=exp_id,
                name=name,
                parent_id=parent_id,
                config_json=json.dumps(config),
                status="CREATED",
                description=description,
                created_at=datetime.utcnow(),
            )
            session.add(exp)
            session.commit()
            logger.info("Created experiment '%s' (ID: %s, Parent: %s)", name, exp_id, parent_id)
            return exp

    def branch_experiment(
        self,
        parent_id: str,
        new_name: str,
        config_overrides: Optional[Dict[str, Any]] = None,
        from_checkpoint_path: Optional[str | Path] = None,
        description: Optional[str] = None,
    ) -> str:
        """Branches an experiment from a parent, optionally seeding with a checkpoint.

        Parent experiment files and state are preserved without modification.
        """
        with self.db.get_session() as session:
            parent = session.query(ExperimentModel).filter_by(id=parent_id).first()
            if not parent:
                raise StorageError(f"Cannot branch from non-existent parent experiment: {parent_id}")

            parent_config = json.loads(parent.config_json)

        # Merge config overrides
        branched_config = dict(parent_config)
        if config_overrides:
            self._deep_update(branched_config, config_overrides)

        new_exp_id = f"{parent_id}_branch_{uuid.uuid4().hex[:6]}"
        desc = description or f"Branched from {parent_id} with modifications: {config_overrides}"

        self.create_experiment(
            name=new_name,
            config=branched_config,
            experiment_id=new_exp_id,
            parent_id=parent_id,
            description=desc,
        )

        # Copy initial checkpoint if specified
        if from_checkpoint_path:
            src_ckpt = Path(from_checkpoint_path)
            if src_ckpt.exists():
                dst_latest = self.experiments_dir / new_exp_id / "checkpoints" / "latest"
                dst_latest.mkdir(parents=True, exist_ok=True)
                for item in src_ckpt.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dst_latest / item.name)

        logger.info("Successfully branched experiment %s -> %s", parent_id, new_exp_id)
        return new_exp_id

    def update_status(
        self,
        experiment_id: str,
        status: str,
        best_metric: Optional[float] = None,
        total_steps: Optional[int] = None,
        current_epoch: Optional[int] = None,
    ) -> None:
        """Update experiment runtime status and summary counters."""
        with self.db.get_session() as session:
            exp = session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if exp:
                exp.status = status
                if best_metric is not None:
                    exp.best_metric = best_metric
                if total_steps is not None:
                    exp.total_steps = total_steps
                if current_epoch is not None:
                    exp.current_epoch = current_epoch
                session.commit()

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve an experiment record as dictionary."""
        with self.db.get_session() as session:
            exp = session.query(ExperimentModel).filter_by(id=experiment_id).first()
            if not exp:
                return None
            return {
                "id": exp.id,
                "name": exp.name,
                "parent_id": exp.parent_id,
                "status": exp.status,
                "config": json.loads(exp.config_json),
                "best_metric": exp.best_metric,
                "total_steps": exp.total_steps,
                "current_epoch": exp.current_epoch,
                "description": exp.description,
                "created_at": exp.created_at.isoformat() if exp.created_at else None,
            }

    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all experiments in descending order of creation."""
        with self.db.get_session() as session:
            exps = session.query(ExperimentModel).order_by(ExperimentModel.created_at.desc()).all()
            return [
                {
                    "id": e.id,
                    "name": e.name,
                    "parent_id": e.parent_id,
                    "status": e.status,
                    "best_metric": e.best_metric,
                    "total_steps": e.total_steps,
                    "current_epoch": e.current_epoch,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                }
                for e in exps
            ]

    def compare_experiments(self, exp_id_1: str, exp_id_2: str) -> Dict[str, Any]:
        """Compare two experiments in terms of config diffs and performance metrics."""
        e1 = self.get_experiment(exp_id_1)
        e2 = self.get_experiment(exp_id_2)
        if not e1 or not e2:
            raise StorageError(f"Could not load experiments {exp_id_1} and {exp_id_2} for comparison")

        config1 = e1["config"]
        config2 = e2["config"]

        diffs = {}
        all_keys = set(config1.keys()).union(set(config2.keys()))
        for k in all_keys:
            v1 = config1.get(k)
            v2 = config2.get(k)
            if v1 != v2:
                diffs[k] = {"exp1": v1, "exp2": v2}

        return {
            "experiment_1": {"id": exp_id_1, "name": e1["name"], "best_metric": e1["best_metric"], "steps": e1["total_steps"]},
            "experiment_2": {"id": exp_id_2, "name": e2["name"], "best_metric": e2["best_metric"], "steps": e2["total_steps"]},
            "config_differences": diffs,
        }

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        for k, v in update_dict.items():
            if isinstance(v, dict) and k in base_dict and isinstance(base_dict[k], dict):
                self._deep_update(base_dict[k], v)
            else:
                base_dict[k] = v
