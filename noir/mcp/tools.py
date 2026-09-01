"""Model Context Protocol (MCP) tool definitions and dispatchers for Project NOIR."""

import json
from typing import Any, Callable, Dict, List, Optional
from noir.core.logging import get_logger

logger = get_logger("mcp.tools")


class MCPToolRegistry:
    """Maintains and executes MCP-compliant tools."""

    def __init__(self, engine: Any):
        self.engine = engine
        self._tools: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def register(self, name: str, description: str, parameters: Dict[str, Any], func: Callable[..., Any]) -> None:
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "func": func,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool schemas."""
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["parameters"],
            }
            for t in self._tools.values()
        ]

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with given arguments."""
        if tool_name not in self._tools:
            return {"error": f"Tool '{tool_name}' not found", "success": False}

        tool_meta = self._tools[tool_name]
        try:
            result = tool_meta["func"](**arguments)
            return {"result": result, "success": True}
        except Exception as e:
            logger.error("MCP Tool execution error on '%s': %s", tool_name, e)
            return {"error": str(e), "success": False}

    def _register_default_tools(self) -> None:
        # 1. get_training_status
        self.register(
            name="get_training_status",
            description="Retrieve the current lifecycle state, active experiment ID, step, and epoch.",
            parameters={"type": "object", "properties": {}},
            func=lambda: {
                "state": self.engine.lifecycle.current_state.value,
                "experiment_id": self.engine.current_experiment_id,
                "step": getattr(self.engine.trainer, "global_step", 0) if self.engine.trainer else 0,
                "epoch": getattr(self.engine.trainer, "current_epoch", 0) if self.engine.trainer else 0,
            },
        )

        # 2. get_latest_metrics
        self.register(
            name="get_latest_metrics",
            description="Retrieve recent training, evaluation, and reward metrics.",
            parameters={"type": "object", "properties": {}},
            func=lambda: getattr(self.engine.trainer, "latest_metrics", {}) if self.engine.trainer else {},
        )

        # 3. get_emotion_state
        self.register(
            name="get_emotion_state",
            description="Retrieve the mathematical affective/emotion state vector (Confidence, Frustration, Uncertainty, Curiosity, etc.).",
            parameters={"type": "object", "properties": {}},
            func=lambda: self.engine.affective_engine.current_state.to_dict() if self.engine.affective_engine else {},
        )

        # 4. inspect_model
        self.register(
            name="inspect_model",
            description="Inspect model architecture, parameter count, and weight statistics.",
            parameters={"type": "object", "properties": {}},
            func=lambda: {
                "architecture": self.engine.model.get_architecture_summary() if hasattr(self.engine, "model") and self.engine.model else [],
                "statistics": self.engine.model.get_weight_statistics() if hasattr(self.engine, "model") and self.engine.model else {},
            },
        )

        # 5. inspect_layer
        self.register(
            name="inspect_layer",
            description="Inspect activations and gradient statistics for a specific model layer.",
            parameters={"type": "object", "properties": {"layer_name": {"type": "string"}}, "required": ["layer_name"]},
            func=lambda layer_name: self._inspect_layer(layer_name),
        )

        # 6. create_checkpoint
        self.register(
            name="create_checkpoint",
            description="Trigger an immediate atomic checkpoint save.",
            parameters={"type": "object", "properties": {"tag": {"type": "string"}}},
            func=lambda tag="mcp_manual": str(self.engine.save_checkpoint(tag=tag)),
        )

        # 7. pause_training
        self.register(
            name="pause_training",
            description="Pause active training.",
            parameters={"type": "object", "properties": {}},
            func=lambda: self.engine.pause_training(),
        )

        # 8. resume_training
        self.register(
            name="resume_training",
            description="Resume paused training.",
            parameters={"type": "object", "properties": {}},
            func=lambda: self.engine.resume_training(),
        )

        # 9. branch_experiment
        self.register(
            name="branch_experiment",
            description="Create a new experiment branch with modified hyperparameters from a checkpoint.",
            parameters={
                "type": "object",
                "properties": {
                    "new_name": {"type": "string"},
                    "config_overrides": {"type": "object"},
                },
                "required": ["new_name"],
            },
            func=lambda new_name, config_overrides=None: self.engine.branch_experiment(new_name, config_overrides),
        )

    def _inspect_layer(self, layer_name: str) -> Dict[str, Any]:
        if not hasattr(self.engine, "model") or not self.engine.model:
            return {"error": "No model loaded"}

        activations = self.engine.model.get_layer_activations()
        weights = self.engine.model.get_layer_weights()

        res = {"layer": layer_name}
        if layer_name in weights:
            w = weights[layer_name]
            res["weight_shape"] = list(w.shape)
            res["weight_mean"] = float(w.mean().item())
            res["weight_std"] = float(w.std().item())
        if layer_name in activations:
            a = activations[layer_name]
            res["activation_shape"] = list(a.shape)
            res["activation_mean"] = float(a.mean().item())
        return res
