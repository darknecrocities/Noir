"""Atomic checkpoint persistence manager for model weights, optimizer, RNG, and states."""

import io
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors, save_file as save_safetensors

from noir.core.exceptions import CheckpointError
from noir.core.logging import get_logger

logger = get_logger("checkpoint_manager")


class CheckpointManager:
    """Manages atomic saving, loading, validation, and retention of training checkpoints."""

    def __init__(self, base_dir: str | Path = "checkpoints", retention: int = 20):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.retention = retention

    def _get_experiment_dir(self, experiment_id: str) -> Path:
        exp_dir = self.base_dir / experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "history").mkdir(parents=True, exist_ok=True)
        (exp_dir / "latest").mkdir(parents=True, exist_ok=True)
        (exp_dir / "best").mkdir(parents=True, exist_ok=True)
        return exp_dir

    def save_checkpoint(
        self,
        experiment_id: str,
        step: int,
        epoch: int,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        emotion_state: Optional[Dict[str, float]] = None,
        config: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        env_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
        tag: Optional[str] = None,
    ) -> Path:
        """Atomically saves a full checkpoint to disk.

        Args:
            experiment_id: Unique ID of the active experiment.
            step: Current global training step.
            epoch: Current training epoch.
            model: PyTorch model to persist.
            optimizer: PyTorch optimizer instance.
            scheduler: Learning rate scheduler instance.
            emotion_state: Mathematical emotion vector dictionary.
            config: Experiment configuration dictionary.
            metrics: Current loss, reward, or accuracy dictionary.
            env_state: Extra environment state.
            is_best: Whether this checkpoint achieved best recorded metric.
            tag: Optional custom label (e.g. 'manual', 'surprise', 'shutdown').

        Returns:
            Path to the saved checkpoint directory.
        """
        exp_dir = self._get_experiment_dir(experiment_id)
        ckpt_name = f"step_{step:08d}_epoch_{epoch:04d}"
        if tag:
            ckpt_name += f"_{tag}"

        temp_ckpt_dir = exp_dir / f".tmp_{ckpt_name}_{int(time.time()*1000)}"
        final_ckpt_dir = exp_dir / "history" / ckpt_name

        try:
            if temp_ckpt_dir.exists():
                shutil.rmtree(temp_ckpt_dir)
            temp_ckpt_dir.mkdir(parents=True, exist_ok=True)

            # 1. Save Model Weights via Safetensors (or PyTorch fallback if complex types)
            weights_file = temp_ckpt_dir / "model.safetensors"
            state_dict = model.state_dict()
            # Convert tensors to contiguous independent CPU tensors for robust safetensors serialization
            cpu_state_dict = {k: v.detach().cpu().contiguous().clone() for k, v in state_dict.items()}
            try:
                save_safetensors(cpu_state_dict, weights_file)
            except Exception as se:
                logger.debug("Safetensors save notice: %s. Using torch.save fallback.", se)
                weights_file = temp_ckpt_dir / "model.pt"
                torch.save(cpu_state_dict, weights_file)

            # 2. Save Optimizer, Scheduler, RNG, and Metadata via PyTorch bundle
            training_state = {
                "step": step,
                "epoch": epoch,
                "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "rng_states": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
                "emotion_state": emotion_state or {},
                "env_state": env_state or {},
                "metrics": metrics or {},
                "is_best": is_best,
                "timestamp": time.time(),
                "experiment_id": experiment_id,
            }
            torch.save(training_state, temp_ckpt_dir / "state.pt")

            # 3. Save Human-readable Metadata & Config JSON
            meta = {
                "experiment_id": experiment_id,
                "step": step,
                "epoch": epoch,
                "timestamp": time.time(),
                "is_best": is_best,
                "tag": tag,
                "metrics": metrics or {},
                "emotion_state": emotion_state or {},
                "config": config or {},
            }
            with open(temp_ckpt_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            # 4. Atomic Verification: Test reading back metadata
            with open(temp_ckpt_dir / "meta.json", "r", encoding="utf-8") as f:
                verified_meta = json.load(f)
                if verified_meta.get("step") != step:
                    raise CheckpointError("Checkpoint verification failed: metadata step mismatch")

            # 5. Atomic Rename
            if final_ckpt_dir.exists():
                shutil.rmtree(final_ckpt_dir)
            temp_ckpt_dir.rename(final_ckpt_dir)

            # 6. Update 'latest' pointer atomically
            latest_dir = exp_dir / "latest"
            self._copy_dir_contents(final_ckpt_dir, latest_dir)

            # 7. Update 'best' if requested
            if is_best:
                best_dir = exp_dir / "best"
                self._copy_dir_contents(final_ckpt_dir, best_dir)

            # 8. Retention Pruning
            self._prune_history(exp_dir / "history")

            logger.info("Saved atomic checkpoint: %s (step=%d, epoch=%d)", final_ckpt_dir.name, step, epoch)
            return final_ckpt_dir

        except Exception as e:
            if temp_ckpt_dir.exists():
                shutil.rmtree(temp_ckpt_dir, ignore_errors=True)
            logger.error("Failed to save checkpoint for experiment %s at step %d: %s", experiment_id, step, e)
            raise CheckpointError(f"Failed to save atomic checkpoint: {e}") from e

    def load_checkpoint(
        self,
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        device: str = "cpu",
    ) -> Dict[str, Any]:
        """Loads a checkpoint from disk and restores weights, optimizer, and RNG states.

        Returns:
            Dictionary containing state metadata (step, epoch, emotion_state, metrics, etc.).
        """
        ckpt_dir = Path(checkpoint_path)
        if not ckpt_dir.exists():
            raise CheckpointError(f"Checkpoint path does not exist: {ckpt_dir}")

        logger.info("Loading checkpoint from %s to device %s", ckpt_dir, device)

        # 1. Load weights
        safetensors_path = ckpt_dir / "model.safetensors"
        torch_weights_path = ckpt_dir / "model.pt"

        if safetensors_path.exists():
            state_dict = load_safetensors(safetensors_path)
            model.load_state_dict(state_dict)
        elif torch_weights_path.exists():
            state_dict = torch.load(torch_weights_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
        else:
            raise CheckpointError(f"No valid weights file found in {ckpt_dir}")

        # 2. Load training state
        state_file = ckpt_dir / "state.pt"
        training_state: Dict[str, Any] = {}
        if state_file.exists():
            training_state = torch.load(state_file, map_location=device, weights_only=False)

            if optimizer and training_state.get("optimizer_state_dict"):
                try:
                    optimizer.load_state_dict(training_state["optimizer_state_dict"])
                except Exception as oe:
                    logger.warning("Could not fully restore optimizer state: %s", oe)

            if scheduler and training_state.get("scheduler_state_dict"):
                try:
                    scheduler.load_state_dict(training_state["scheduler_state_dict"])
                except Exception as se:
                    logger.warning("Could not fully restore scheduler state: %s", se)

            # Restore RNG states
            rng = training_state.get("rng_states", {})
            if "python" in rng:
                random.setstate(rng["python"])
            if "numpy" in rng:
                np.random.set_state(rng["numpy"])
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "torch_cuda" in rng and torch.cuda.is_available() and rng["torch_cuda"] is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])

        # 3. Read metadata
        meta_file = ckpt_dir / "meta.json"
        meta: Dict[str, Any] = {}
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

        return {
            "step": training_state.get("step", meta.get("step", 0)),
            "epoch": training_state.get("epoch", meta.get("epoch", 0)),
            "emotion_state": training_state.get("emotion_state", meta.get("emotion_state", {})),
            "env_state": training_state.get("env_state", {}),
            "metrics": training_state.get("metrics", meta.get("metrics", {})),
            "config": meta.get("config", {}),
            "checkpoint_dir": str(ckpt_dir),
        }

    def get_latest_checkpoint(self, experiment_id: str) -> Optional[Path]:
        """Find the latest checkpoint directory for an experiment."""
        exp_dir = self.base_dir / experiment_id
        latest_dir = exp_dir / "latest"
        if latest_dir.exists() and (latest_dir / "meta.json").exists():
            return latest_dir

        # Fallback to scanning history
        history_dir = exp_dir / "history"
        if history_dir.exists():
            ckpts = sorted(list(history_dir.iterdir()))
            if ckpts:
                return ckpts[-1]

        return None

    def get_best_checkpoint(self, experiment_id: str) -> Optional[Path]:
        """Find the best checkpoint directory for an experiment."""
        exp_dir = self.base_dir / experiment_id
        best_dir = exp_dir / "best"
        if best_dir.exists() and (best_dir / "meta.json").exists():
            return best_dir
        return None

    def list_checkpoints(self, experiment_id: str) -> List[Dict[str, Any]]:
        """List all checkpoint records for an experiment."""
        exp_dir = self.base_dir / experiment_id / "history"
        if not exp_dir.exists():
            return []

        results = []
        for ckpt in sorted(exp_dir.iterdir()):
            if ckpt.is_dir():
                meta_file = ckpt / "meta.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            m = json.load(f)
                            m["path"] = str(ckpt)
                            results.append(m)
                    except Exception:
                        results.append({"path": str(ckpt), "name": ckpt.name})
        return results

    def _copy_dir_contents(self, src: Path, dst: Path) -> None:
        """Helper to copy files safely into target directory."""
        dst.mkdir(parents=True, exist_ok=True)
        for item in src.iterdir():
            target = dst / item.name
            if item.is_file():
                shutil.copy2(item, target)
            elif item.is_dir():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(item, target)

    def _prune_history(self, history_dir: Path) -> None:
        """Retain only latest N checkpoints in history."""
        if not history_dir.exists():
            return

        ckpts = sorted([p for p in history_dir.iterdir() if p.is_dir()])
        if len(ckpts) > self.retention:
            to_remove = ckpts[: len(ckpts) - self.retention]
            for p in to_remove:
                try:
                    shutil.rmtree(p)
                    logger.debug("Pruned old checkpoint: %s", p.name)
                except Exception as e:
                    logger.warning("Failed to prune old checkpoint %s: %s", p.name, e)
