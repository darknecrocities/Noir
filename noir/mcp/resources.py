"""MCP Resource providers for Project NOIR."""

from typing import Any, Dict, List
import json


class MCPResourceManager:
    """Manages MCP-exposed URI resources."""

    def __init__(self, engine: Any):
        self.engine = engine

    def list_resources(self) -> List[Dict[str, Any]]:
        return [
            {
                "uri": "noir://metrics/latest",
                "name": "Latest Metrics",
                "mimeType": "application/json",
                "description": "Real-time loss, accuracy, and reward metrics.",
            },
            {
                "uri": "noir://mind/affective_state",
                "name": "Affective State",
                "mimeType": "application/json",
                "description": "Mathematical emotion vector (E_t).",
            },
            {
                "uri": "noir://model/architecture",
                "name": "Model Architecture",
                "mimeType": "application/json",
                "description": "Layer specification and parameter summary.",
            },
        ]

    def read_resource(self, uri: str) -> str:
        if uri == "noir://metrics/latest":
            metrics = getattr(self.engine.trainer, "latest_metrics", {}) if self.engine.trainer else {}
            return json.dumps(metrics, indent=2)
        elif uri == "noir://mind/affective_state":
            state = self.engine.affective_engine.current_state.to_dict() if self.engine.affective_engine else {}
            return json.dumps(state, indent=2)
        elif uri == "noir://model/architecture":
            summary = self.engine.model.get_architecture_summary() if hasattr(self.engine, "model") and self.engine.model else []
            return json.dumps(summary, indent=2)
        else:
            return json.dumps({"error": f"Resource not found: {uri}"})
