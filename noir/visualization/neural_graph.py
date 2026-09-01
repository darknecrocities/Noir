"""Neural network 3D graph model representation with intelligent layer grouping and performance LOD."""

import re
from typing import Any, Dict, List, Optional, Tuple
import torch

from noir.models.base import NoirBaseModel
from noir.visualization.tensor_projection import TensorProjector


class NeuralGraph:
    """Encapsulates the 3D visual graph representing active model architecture and states."""

    def __init__(self, max_nodes: int = 150, max_connections: int = 400):
        self.max_nodes = max_nodes
        self.max_connections = max_connections
        # Optimized for clean visual fidelity and 60 FPS rendering
        self.projector = TensorProjector(max_nodes_per_layer=14, max_connections_per_layer=28)

        self.nodes: List[Dict[str, Any]] = []
        self.connections: List[Dict[str, Any]] = []
        self.layer_metadata: List[Dict[str, Any]] = []
        self.pulse_phase: float = 0.0

    def _clean_layer_name(self, raw_name: str) -> str:
        """Convert raw PyTorch dot paths to clean, readable titles."""
        if "wte" in raw_name or "embed" in raw_name:
            return "Token Embedding"
        if "wpe" in raw_name or "pos" in raw_name:
            return "Positional Embedding"
        if "lm_head" in raw_name or "out" in raw_name:
            return "LM Head Output"

        # Transformer blocks
        match = re.search(r"blocks\.(\d+)", raw_name)
        if match:
            b_num = int(match.group(1)) + 1
            if "attn" in raw_name:
                return f"Block {b_num} Attention"
            if "mlp" in raw_name:
                return f"Block {b_num} MLP"
            return f"Transformer Block {b_num}"

        # Standard MLP
        match = re.search(r"network\.(\d+)", raw_name)
        if match:
            idx = int(match.group(1))
            return f"Dense Layer {idx}"

        return raw_name.replace("_", " ").title()

    def update_from_model(self, model: NoirBaseModel) -> None:
        """Extract live layer topology, activations, and weights with intelligent semantic clustering."""
        if not isinstance(model, NoirBaseModel):
            return

        activations = model.get_layer_activations()
        weights = model.get_layer_weights()
        gradients = model.get_layer_gradients()

        # Identify distinct weight-bearing layers
        raw_layers = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0 and hasattr(module, "weight") and module.weight is not None:
                num_units = getattr(module, "out_features", getattr(module, "out_channels", getattr(module, "embedding_dim", 16)))
                raw_layers.append((name, module, num_units))

        if not raw_layers:
            raw_layers = [("input", None, 16), ("hidden", None, 32), ("output", None, 4)]

        # If too many sub-layers (e.g. 20+ in Transformer), select salient representative milestones
        if len(raw_layers) > 6:
            filtered_layers = []
            # Keep first (embedding)
            filtered_layers.append(raw_layers[0])
            # Keep one per transformer block or middle layers
            seen_blocks = set()
            for name, mod, units in raw_layers[1:-1]:
                block_match = re.search(r"blocks\.(\d+)", name)
                if block_match:
                    b_id = block_match.group(1)
                    if b_id not in seen_blocks and ("c_attn" in name or "c_proj" in name or "0" in name):
                        seen_blocks.add(b_id)
                        filtered_layers.append((name, mod, units))
                elif len(filtered_layers) < 5:
                    filtered_layers.append((name, mod, units))

            # Keep last (output head)
            if raw_layers[-1] not in filtered_layers:
                filtered_layers.append(raw_layers[-1])

            layer_items = filtered_layers
        else:
            layer_items = raw_layers

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
            {
                "idx": i,
                "raw_name": item[0],
                "name": self._clean_layer_name(item[0]),
                "units": item[2],
            }
            for i, item in enumerate(layer_items)
        ]
