"""Master orchestrator engine for Project NOIR."""

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import psutil
import torch
import torch.nn as nn

from noir.core.config import AppConfig, load_config
from noir.core.exceptions import EngineStateError, RecoveryError
from noir.core.lifecycle import LifecycleManager, LifecycleState
from noir.core.logging import get_logger
from noir.datasets.dataset_loader import create_classification_dataloaders
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_store import EventStore
from noir.events.event_types import EventType
from noir.mcp.server import MCPServer
from noir.memory.memory_manager import MemoryManager
from noir.mind.affective_engine import AffectiveEngine
from noir.models.actor_critic import ActorCriticNetwork
from noir.models.base import NoirBaseModel
from noir.models.mlp import NoirMLP
from noir.models.transformer import NoirTransformerLM
from noir.storage.checkpoint_manager import CheckpointManager
from noir.storage.database import DatabaseManager
from noir.storage.experiment_repository import ExperimentRepository
from noir.storage.metrics_repository import MetricsRepository
from noir.reporting.report_generator import TrainingReportGenerator
from noir.storage.recovery import RecoveryManager
from noir.strategy.llm_provider import create_llm_provider
from noir.strategy.strategist import Strategist
from noir.training.callbacks import (
    CheckpointCallback,
    EventEmissionCallback,
    MetricsLoggingCallback,
)
from noir.training.llm_trainer import OpenWebLLMTrainer
from noir.training.rl_trainer import RLTrainer
from noir.training.supervised_trainer import SupervisedTrainer

logger = get_logger("core.engine")


class NoirEngine:
    """Master application engine coordinating training, persistence, mind, and UI subsystems."""

    def __init__(self, config_path: Optional[str | Path] = None):
        self.config = load_config(config_path)
        self.lifecycle = LifecycleManager(LifecycleState.INITIALIZING)

        # 1. Event Subsystem
        self.event_bus = get_event_bus()

        # 2. Storage Subsystem
        self.db = DatabaseManager(self.config.storage.database)
        self.event_store = EventStore(self.config.storage.database)
        self.checkpoint_manager = CheckpointManager(
            base_dir=self.config.storage.checkpoint_dir,
            retention=self.config.storage.checkpoint_retention,
        )
        self.experiment_repo = ExperimentRepository(
            db_manager=self.db,
            experiments_dir=self.config.storage.experiments_dir,
        )
        self.metrics_repo = MetricsRepository(db_manager=self.db)
        self.recovery_manager = RecoveryManager(
            db_manager=self.db,
            checkpoint_manager=self.checkpoint_manager,
            experiments_dir=self.config.storage.experiments_dir,
        )

        # 3. Memory & Mind Subsystems
        self.current_experiment_id: str = "exp_default"
        self.memory_manager = MemoryManager(
            experiment_id=self.current_experiment_id,
            db_manager=self.db,
            memory_dir=self.config.storage.memory_dir,
            event_bus=self.event_bus,
        )
        self.affective_engine = AffectiveEngine(
            experiment_id=self.current_experiment_id,
            surprise_threshold=self.config.emotion.surprise_threshold,
            curiosity_weight=self.config.emotion.curiosity_weight,
            frustration_decay=self.config.emotion.frustration_decay,
            confidence_decay=self.config.emotion.confidence_decay,
            event_bus=self.event_bus,
            device=self._get_device_str(),
        )

        # 4. Strategist Subsystem
        llm_provider = create_llm_provider(
            provider_type=self.config.strategy.provider,
            api_base=self.config.strategy.api_base,
            api_key=self.config.strategy.api_key,
            model=self.config.strategy.model,
        )
        self.strategist = Strategist(
            experiment_id=self.current_experiment_id,
            provider=llm_provider,
            db_manager=self.db,
            event_bus=self.event_bus,
            analysis_interval_steps=self.config.strategy.analysis_interval_steps,
        )

        # 5. Active Training Model and Trainer
        self.model: Optional[NoirBaseModel] = None
        self.trainer: Optional[Any] = None

        # 6. MCP Server Subsystem
        self.mcp_server: Optional[MCPServer] = None
        if self.config.mcp.enabled:
            self.mcp_server = MCPServer(
                engine=self,
                host=self.config.mcp.host,
                port=self.config.mcp.port,
            )

        # 7. Telemetry Monitor Thread
        self._telemetry_running = True
        self._telemetry_thread = threading.Thread(target=self._telemetry_loop, daemon=True, name="NoirTelemetry")

        # Subscribe EventStore to persist events
        self.event_bus.subscribe(None, self._on_any_event)

        self.lifecycle.transition_to(LifecycleState.READY)
        logger.info("Project NOIR Engine initialized successfully.")

    def _get_device_str(self) -> str:
        if self.config.training.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.config.training.device

    def _on_any_event(self, event: NoirEvent) -> None:
        """Persist every event to the SQLite event store."""
        self.event_store.save_event(event)

    def start(self) -> None:
        """Start background services."""
        if not self._telemetry_thread.is_alive():
            self._telemetry_thread.start()
        if self.mcp_server:
            self.mcp_server.start()

    def start_supervised_experiment(
        self,
        name: Optional[str] = None,
        dataset_name: str = "digits",
        hidden_dims: Optional[List[int]] = None,
        learning_rate: float = 0.001,
        num_epochs: int = 100,
        batch_size: int = 64,
        parent_id: Optional[str] = None,
    ) -> str:
        """Initialize and start a real supervised training experiment on authentic real-world data."""
        self.stop_training()

        # 1. Automatically load and inspect real dataset
        dataset_bundle = create_classification_dataloaders(
            dataset_name=dataset_name,
            batch_size=batch_size,
        )

        input_dim = dataset_bundle.input_dim
        num_classes = dataset_bundle.num_classes
        hidden_dims = hidden_dims or self.config.training.supervised.hidden_dims

        exp_name = name or f"Real {dataset_bundle.name.replace('_', ' ').title()} Classification"

        exp_config = {
            "mode": "supervised",
            "name": exp_name,
            "dataset": dataset_bundle.name,
            "dataset_description": dataset_bundle.description,
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "num_classes": num_classes,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
        }

        # Create experiment record
        exp_record = self.experiment_repo.create_experiment(
            name=exp_name,
            config=exp_config,
            parent_id=parent_id,
        )
        self.current_experiment_id = exp_record.id
        self._update_subsystems_experiment_id(exp_record.id)

        # 2. Automatically instantiate neural model matching dataset dimensions
        self.model = NoirMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=num_classes,
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        callbacks = [
            EventEmissionCallback(self.event_bus),
            MetricsLoggingCallback(self.metrics_repo),
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                interval_steps=self.config.training.checkpoint_interval_steps,
                autosave_seconds=self.config.training.autosave_interval_seconds,
            ),
        ]

        self.trainer = SupervisedTrainer(
            experiment_id=exp_record.id,
            model=self.model,
            train_loader=dataset_bundle.train_loader,
            val_loader=dataset_bundle.val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=num_epochs,
            device=self._get_device_str(),
            callbacks=callbacks,
            affective_engine=self.affective_engine,
            config=exp_config,
        )

        self.experiment_repo.update_status(exp_record.id, "RUNNING")
        self.trainer.start_training()
        self.lifecycle.transition_to(LifecycleState.RUNNING)
        logger.info(
            "Supervised experiment started: %s (Dataset: %s, Features: %d, Classes: %d, ID: %s)",
            exp_name,
            dataset_bundle.name,
            input_dim,
            num_classes,
            exp_record.id,
        )
        return exp_record.id

    def start_rl_experiment(
        self,
        name: str = "PPO GridWorld Exploration",
        env_id: str = "GridWorld-v0",
        grid_size: int = 8,
        learning_rate: float = 0.0003,
        n_steps: int = 256,
        max_episodes: int = 500,
        parent_id: Optional[str] = None,
    ) -> str:
        """Initialize and start a real PPO Reinforcement Learning experiment."""
        self.stop_training()

        exp_config = {
            "mode": "reinforcement_learning",
            "name": name,
            "env_id": env_id,
            "grid_size": grid_size,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "max_episodes": max_episodes,
        }

        exp_record = self.experiment_repo.create_experiment(
            name=name,
            config=exp_config,
            parent_id=parent_id,
        )
        self.current_experiment_id = exp_record.id
        self._update_subsystems_experiment_id(exp_record.id)

        # Initialize ActorCritic model
        self.model = ActorCriticNetwork(state_dim=16, action_dim=4, hidden_dims=[128, 64])

        callbacks = [
            EventEmissionCallback(self.event_bus),
            MetricsLoggingCallback(self.metrics_repo),
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                interval_steps=self.config.training.checkpoint_interval_steps,
                autosave_seconds=self.config.training.autosave_interval_seconds,
            ),
        ]

        self.trainer = RLTrainer(
            experiment_id=exp_record.id,
            network=self.model,
            env_id=env_id,
            learning_rate=learning_rate,
            gamma=self.config.reinforcement_learning.gamma,
            gae_lambda=self.config.reinforcement_learning.gae_lambda,
            clip_eps=self.config.reinforcement_learning.clip_eps,
            entropy_coef=self.config.reinforcement_learning.entropy_coef,
            value_loss_coef=self.config.reinforcement_learning.value_loss_coef,
            n_steps=n_steps,
            max_episodes=max_episodes,
            device=self._get_device_str(),
            callbacks=callbacks,
            affective_engine=self.affective_engine,
            config=exp_config,
            event_bus=self.event_bus,
        )

        self.experiment_repo.update_status(exp_record.id, "RUNNING")
        self.trainer.start_training()
        self.lifecycle.transition_to(LifecycleState.RUNNING)
        logger.info("RL experiment started: %s (ID: %s)", name, exp_record.id)
        return exp_record.id

    def start_open_web_llm_experiment(
        self,
        name: str = "Open Web Live Internet LLM Stream",
        vocab_size: int = 256,
        block_size: int = 64,
        embed_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        learning_rate: float = 0.0005,
        batch_size: int = 16,
        max_steps: int = 10000,
        parent_id: Optional[str] = None,
    ) -> str:
        """Initialize and start real-time Causal Transformer learning on live internet data."""
        self.stop_training()

        exp_config = {
            "mode": "open_web_llm",
            "name": name,
            "vocab_size": vocab_size,
            "block_size": block_size,
            "embed_dim": embed_dim,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_steps": max_steps,
        }

        exp_record = self.experiment_repo.create_experiment(
            name=name,
            config=exp_config,
            parent_id=parent_id,
        )
        self.current_experiment_id = exp_record.id
        self._update_subsystems_experiment_id(exp_record.id)

        # 1. Instantiate Causal Transformer on GPU
        self.model = NoirTransformerLM(
            vocab_size=vocab_size,
            block_size=block_size,
            n_layers=n_layers,
            n_heads=n_heads,
            embed_dim=embed_dim,
        )

        callbacks = [
            EventEmissionCallback(self.event_bus),
            MetricsLoggingCallback(self.metrics_repo),
            CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                interval_steps=self.config.training.checkpoint_interval_steps,
                autosave_seconds=self.config.training.autosave_interval_seconds,
            ),
        ]

        self.trainer = OpenWebLLMTrainer(
            experiment_id=exp_record.id,
            model=self.model,
            learning_rate=learning_rate,
            batch_size=batch_size,
            block_size=block_size,
            max_steps=max_steps,
            device=self._get_device_str(),
            callbacks=callbacks,
            affective_engine=self.affective_engine,
            config=exp_config,
        )

        self.experiment_repo.update_status(exp_record.id, "RUNNING")
        self.trainer.start_training()
        self.lifecycle.transition_to(LifecycleState.RUNNING)
        logger.info("Open Web LLM experiment started on device %s (ID: %s)", self._get_device_str(), exp_record.id)
        return exp_record.id

    def start_autonomous_master_training(self, learning_rate: float = 0.0005) -> str:
        """Start the unified autonomous master research loop across internet text, real datasets, and curiosity RL."""
        return self.start_open_web_llm_experiment(
            name="Project NOIR Autonomous Master Research (All)",
            learning_rate=learning_rate,
            max_steps=50000,
        )

    def pause_training(self) -> None:
        if self.trainer:
            self.trainer.pause_training()
            self.experiment_repo.update_status(self.current_experiment_id, "PAUSED")
            self.lifecycle.transition_to(LifecycleState.PAUSED)

    def resume_training(self) -> None:
        if self.trainer:
            self.trainer.resume_training()
            self.experiment_repo.update_status(self.current_experiment_id, "RUNNING")
            self.lifecycle.transition_to(LifecycleState.RUNNING)

    def stop_training(self) -> None:
        if self.trainer:
            self.trainer.stop_training(wait=True)
            self.experiment_repo.update_status(self.current_experiment_id, "STOPPED")
            self.lifecycle.transition_to(LifecycleState.STOPPED)

    def save_checkpoint(self, tag: Optional[str] = None) -> Path:
        """Trigger an immediate atomic checkpoint."""
        if not self.trainer or not self.model:
            raise EngineStateError("No active model or trainer to checkpoint")

        emotion_state = self.affective_engine.current_state.to_dict()
        path = self.checkpoint_manager.save_checkpoint(
            experiment_id=self.current_experiment_id,
            step=self.trainer.global_step,
            epoch=self.trainer.current_epoch,
            model=self.model,
            optimizer=self.trainer.optimizer,
            scheduler=getattr(self.trainer, "scheduler", None),
            emotion_state=emotion_state,
            config=self.trainer.config,
            metrics=self.trainer.latest_metrics,
            tag=tag or "manual",
        )

        # Generate human-readable and ELI5 plain-text summary Markdown files
        try:
            resources = []
            if hasattr(self.trainer, "streamer") and hasattr(self.trainer.streamer, "get_resource_history"):
                resources = self.trainer.streamer.get_resource_history()

            sample_text = getattr(self.trainer, "latest_generated_text", None)
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

            TrainingReportGenerator.save_report_files(
                experiment_id=self.current_experiment_id,
                step=self.trainer.global_step,
                epoch=self.trainer.current_epoch,
                metrics=self.trainer.latest_metrics,
                emotion_state=emotion_state,
                resources=resources,
                generated_sample=sample_text,
                device_name=device_name,
                target_dirs=[path, path.parent.parent / "latest"],
                tag=tag or "manual",
            )
        except Exception as re:
            logger.debug("Report generation notice: %s", re)

        self.event_bus.publish(
            NoirEvent.create(
                EventType.CHECKPOINT_CREATED,
                experiment_id=self.current_experiment_id,
                training_step=self.trainer.global_step,
                checkpoint_path=str(path),
            )
        )
        return path

    def load_checkpoint(self, checkpoint_path: str | Path) -> Dict[str, Any]:
        """Load checkpoint weights and state."""
        if not self.model:
            raise EngineStateError("Cannot load weights without initializing a model architecture")

        optimizer = getattr(self.trainer, "optimizer", None)
        scheduler = getattr(self.trainer, "scheduler", None)

        meta = self.checkpoint_manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self._get_device_str(),
        )

        if self.trainer:
            self.trainer.global_step = meta["step"]
            self.trainer.current_epoch = meta["epoch"]

        if meta.get("emotion_state"):
            self.affective_engine.state = self.affective_engine.state.from_dict(meta["emotion_state"])

        self.event_bus.publish(
            NoirEvent.create(
                EventType.CHECKPOINT_LOADED,
                experiment_id=self.current_experiment_id,
                training_step=meta["step"],
                checkpoint_path=str(checkpoint_path),
            )
        )
        return meta

    def branch_experiment(self, new_name: str, config_overrides: Optional[Dict[str, Any]] = None) -> str:
        """Branch the current experiment from latest checkpoint."""
        latest_ckpt = self.checkpoint_manager.get_latest_checkpoint(self.current_experiment_id)
        new_exp_id = self.experiment_repo.branch_experiment(
            parent_id=self.current_experiment_id,
            new_name=new_name,
            config_overrides=config_overrides,
            from_checkpoint_path=latest_ckpt,
        )

        self.event_bus.publish(
            NoirEvent.create(
                EventType.EXPERIMENT_BRANCH_CREATED,
                experiment_id=new_exp_id,
                parent_id=self.current_experiment_id,
                new_name=new_name,
            )
        )
        return new_exp_id

    def recover_from_previous_session(self, action: str = "resume") -> Optional[str]:
        """Attempt to restore the most recent valid checkpoint."""
        recovery_info = self.recovery_manager.check_for_recovery()
        if not recovery_info:
            logger.info("No recoverable previous sessions found.")
            return None

        exp_id = recovery_info["experiment_id"]
        ckpt_path = recovery_info["checkpoint_path"]
        config = recovery_info.get("config", {})
        mode = config.get("mode", "supervised")

        logger.info("Recovering previous session: Exp=%s, Mode=%s, Step=%d", exp_id, mode, recovery_info["step"])

        if mode in ("open_web_llm", "llm", "autonomous"):
            self.start_open_web_llm_experiment(
                name=config.get("name", f"Recovered {exp_id}"),
                vocab_size=config.get("vocab_size", 256),
                block_size=config.get("block_size", 64),
                embed_dim=config.get("embed_dim", 128),
                n_layers=config.get("n_layers", 4),
                n_heads=config.get("n_heads", 4),
                learning_rate=config.get("learning_rate", 0.0005),
                batch_size=config.get("batch_size", 16),
                max_steps=config.get("max_steps", 50000),
            )
        elif mode == "supervised":
            self.start_supervised_experiment(
                name=config.get("name", f"Recovered {exp_id}"),
                dataset_name=config.get("dataset", "digits"),
                hidden_dims=config.get("hidden_dims", [128, 64, 32]),
                learning_rate=config.get("learning_rate", 0.001),
                num_epochs=config.get("num_epochs", 100),
            )
        else:
            self.start_rl_experiment(
                name=config.get("name", f"Recovered {exp_id}"),
                env_id=config.get("env_id", "GridWorld-v0"),
                learning_rate=config.get("learning_rate", 0.0003),
            )

        # Restore weights & state
        self.load_checkpoint(ckpt_path)

        if action == "load_only":
            self.pause_training()

        return exp_id

    def _update_subsystems_experiment_id(self, exp_id: str) -> None:
        self.memory_manager.experiment_id = exp_id
        self.affective_engine.experiment_id = exp_id
        self.strategist.experiment_id = exp_id

    def _telemetry_loop(self) -> None:
        """Background daemon gathering live hardware statistics."""
        while self._telemetry_running:
            try:
                cpu_percent = psutil.cpu_percent(interval=None)
                ram_percent = psutil.virtual_memory().percent

                gpu_percent = 0.0
                gpu_memory = 0.0
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB

                telemetry = {
                    "cpu_percent": cpu_percent,
                    "ram_percent": ram_percent,
                    "gpu_percent": gpu_percent,
                    "gpu_memory_mb": round(gpu_memory, 2),
                    "cuda_available": torch.cuda.is_available(),
                }

                self.event_bus.publish(
                    NoirEvent.create(
                        EventType.SYSTEM_METRICS_UPDATED,
                        experiment_id=self.current_experiment_id,
                        training_step=getattr(self.trainer, "global_step", 0) if self.trainer else 0,
                        **telemetry,
                    ),
                    asynchronous=True,
                )

                # Periodic Strategist Analysis
                if self.trainer and self.trainer.lifecycle.is_running():
                    self.strategist.analyze_async(
                        step=self.trainer.global_step,
                        metrics=self.trainer.latest_metrics,
                        emotion_state=self.affective_engine.current_state.to_dict(),
                        config=self.trainer.config,
                    )

            except Exception as e:
                logger.debug("Telemetry gathering error: %s", e)

            time.sleep(1.0)

    def shutdown(self) -> None:
        """Gracefully terminate engine and all worker threads with automatic checkpoint preservation."""
        logger.info("Project NOIR Engine shutting down gracefully...")
        # Automatically preserve latest training weights & state before exiting
        if self.trainer and self.model and getattr(self.trainer, "global_step", 0) > 0:
            try:
                ckpt_path = self.save_checkpoint(tag="auto_exit")
                logger.info("[AUTO-SAVED] Model weights and training state successfully preserved to: %s", ckpt_path.name)
            except Exception as e:
                logger.debug("Auto-save on exit notice: %s", e)

        self.stop_training()
        self._telemetry_running = False
        if self.mcp_server:
            self.mcp_server.stop()
        self.memory_manager.save_to_disk()
        self.event_bus.shutdown()
        self.lifecycle.transition_to(LifecycleState.STOPPED)
        logger.info("[SHUTDOWN] Engine and all worker threads terminated safely.")
