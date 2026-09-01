"""Neural network 3D graph model representation."""

from typing import Any, Dict, List, Optional
import torch

from noir.models.base import NoirBaseModel
from noir.visualization.tensor_projection import TensorProjector


class NeuralGraph:
    """Encapsulates the 3D visual graph representing active model architecture and states."""

    def __init__(self, max_nodes: int = 500, max_connections: int = 2000):
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        self.projector = TensorProjector(max_nodes_per_layer=24, max_connections_per_layer=64)

        self.nodes: List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.layer_metadata: List[Dict[str, Any]] = []
        self.pulse_phase: float = 0.0

    def update_from_model(self, model: NoirBaseModel) -> None:
        """Extract live layer topology, activations, and weights from PyTorch model."""
        if not isinstance(model, NoirBaseModel):
            return

        activations = model.get_layer_activations()
        weights = model.get_layer_weights()
        gradients = model.get_layer_gradients()

        # Identify layers
        layer_items = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and hasattr(module, "weight"):
                num_units = getattr(module, "out_features", getattr(module, "out_channels", 16))
                layer_items.append((name, module, num_units))

        if not layer_items:
            # Fallback for empty layers
            layer_items = [("input", None, 16), ("hidden", None, 32), ("output", None, 4)]

        total_layers = len(layer_items)
        all_nodes: List[Dict[str, Any]] = []
        all_connections: List[Dict[str, Any]] = []
        layer_nodes_map: List[List[Dict[str, Any]]] = []

        # 1. Project nodes for each layer
        for idx, (name, mod, units) in enumerate(layer_items):
            act = activations.get(name)
            grad = gradients.get(name)
            layer_nodes = self.projector.project_layer_nodes(
                layer_idx=idx,
                total_layers=total_layers,
                num_units=units,
                activations=act,
                gradients=grad,
            )
            all_nodes.extend(layer_nodes)
            layer_nodes_map.append(layer_nodes)

        # 2. Project connections between consecutive layers
        for idx in range(total_layers - 1):
            src_nodes = layer_nodes_map[idx]
            tgt_nodes = layer_nodes_map[idx + 1]
            tgt_name = layer_items[idx + 1][0]
            w_tensor = weights.get(tgt_name)

            conns = self.projector.project_layer_connections(
                source_nodes=src_nodes,
                target_nodes=tgt_nodes,
                weight_tensor=w_tensor,
            )
            all_connections.extend(conns)

        self.nodes = all_nodes
        self.connections = all_connections
        self.layer_metadata = [
            {"idx": i, "name": item[0], "units": item[2]} for i, item in enumerate(layer_items)
        ]
