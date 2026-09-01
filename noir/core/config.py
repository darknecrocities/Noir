"""Configuration system with Pydantic validation and YAML/ENV support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from noir.core.exceptions import ConfigurationError

# Load .env file automatically
load_dotenv()


class ProjectConfig(BaseModel):
    name: str = "Project NOIR"
    version: str = "0.1.0"
    description: str = "Real-time AI and Machine Learning Research Environment"


class SupervisedConfig(BaseModel):
    dataset: str = "synthetic_classification"
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 64, 32])
    num_classes: int = 4
    input_dim: int = 16


class TrainingConfig(BaseModel):
    device: str = "auto"  # "auto", "cuda", "cpu"
    learning_rate: float = 0.0003
    batch_size: int = 64
    checkpoint_interval_steps: int = 100
    autosave_interval_seconds: int = 60
    num_epochs: int = 100
    gradient_clip_val: float = 0.5
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    supervised: SupervisedConfig = Field(default_factory=SupervisedConfig)


class RLConfig(BaseModel):
    algorithm: str = "PPO"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    learning_rate: float = 0.0003
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 10
    env_id: str = "GridWorld-v0"
    grid_size: int = 8
    max_episode_steps: int = 150


class EmotionConfig(BaseModel):
    enabled: bool = True
    curiosity_weight: float = 0.2
    frustration_decay: float = 0.95
    confidence_decay: float = 0.99
    surprise_threshold: float = 0.70
    novelty_weight: float = 0.3
    uncertainty_weight: float = 0.25
    goal_progress_weight: float = 0.4
    persistence_factor: float = 0.85
    caution_threshold: float = 0.65


class MemoryConfig(BaseModel):
    short_term_capacity: int = 100
    episodic_capacity: int = 1000
    semantic_capacity: int = 500
    replay_buffer_size: int = 10000


class StrategyConfig(BaseModel):
    enabled: bool = True
    provider: str = "local"  # "local", "openai", "mock"
    model: str = "llama3"
    api_base: str = "http://localhost:11434/v1"
    api_key: str = ""
    analysis_interval_steps: int = 250
    auto_hypothesize: bool = True


class MCPConfig(BaseModel):
    enabled: bool = True
    port: int = 8765
    host: str = "127.0.0.1"


class VisualizationConfig(BaseModel):
    enabled: bool = True
    update_interval_ms: int = 50
    max_visualized_nodes: int = 500
    max_visualized_connections: int = 2000
    camera_distance: float = 4.5
    theme: str = "noir_dark"


class StorageConfig(BaseModel):
    database: str = "noir.db"
    checkpoint_dir: str = "checkpoints"
    experiments_dir: str = "experiments"
    logs_dir: str = "logs"
    memory_dir: str = "memory"
    checkpoint_retention: int = 20


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    reinforcement_learning: RLConfig = Field(default_factory=RLConfig)
    emotion: EmotionConfig = Field(default_factory=EmotionConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    strategy: StrategyConfig = Field(default_factory=StrategyConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(config_path: Optional[str | Path] = None) -> AppConfig:
    """Load configuration from YAML file and apply environment variable overrides."""
    data: Dict[str, Any] = {}

    if config_path is None:
        default_paths = [
            Path("config/default.yaml"),
            Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml",
        ]
        for p in default_paths:
            if p.exists():
                config_path = p
                break

    if config_path and Path(config_path).exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded and isinstance(loaded, dict):
                    data = loaded
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration from {config_path}: {e}") from e

    # Apply environment variable overrides
    if "NOIR_DB_PATH" in os.environ:
        data.setdefault("storage", {})["database"] = os.environ["NOIR_DB_PATH"]
    if "NOIR_DEVICE" in os.environ:
        data.setdefault("training", {})["device"] = os.environ["NOIR_DEVICE"]
    if "NOIR_LLM_PROVIDER" in os.environ:
        data.setdefault("strategy", {})["provider"] = os.environ["NOIR_LLM_PROVIDER"]
    if "NOIR_LLM_MODEL" in os.environ:
        data.setdefault("strategy", {})["model"] = os.environ["NOIR_LLM_MODEL"]
    if "NOIR_LLM_API_BASE" in os.environ:
        data.setdefault("strategy", {})["api_base"] = os.environ["NOIR_LLM_API_BASE"]
    if "NOIR_LLM_API_KEY" in os.environ:
        data.setdefault("strategy", {})["api_key"] = os.environ["NOIR_LLM_API_KEY"]
    if "NOIR_MCP_ENABLED" in os.environ:
        data.setdefault("mcp", {})["enabled"] = os.environ["NOIR_MCP_ENABLED"].lower() in ("true", "1", "yes")

    try:
        return AppConfig(**data)
    except Exception as e:
        raise ConfigurationError(f"Configuration schema validation failed: {e}") from e


def save_config(config: AppConfig, path: str | Path) -> None:
    """Save the configuration model to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)
    except Exception as e:
        raise ConfigurationError(f"Failed to write configuration to {path}: {e}") from e
