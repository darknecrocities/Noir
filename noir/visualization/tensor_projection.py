"""Tensor projection and dimensionality reduction for 3D neural graph rendering."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class TensorProjector:
    """Projects high-dimensional tensor weights and activations into 3D spatial representations."""

    def __init__(self, max_nodes_per_layer: int = 32, max_connections_per_layer: int = 128):
        self.max_nodes_per_layer = max_nodes_per_layer
        self.max_connections_per_layer = max_connections_per_layer

    def project_layer_nodes(
        self,
        layer_idx: int,
        total_layers: int,
        num_units: int,
        activations: Optional[torch.Tensor] = None,
        gradients: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Compute 3D coordinates and visual properties for a layer's nodes.

        Layout:
            X: Layer depth (from -2.0 to +2.0 across network depth)
            Y: Horizontal spread of neurons in layer
            Z: Vertical spread of neurons in layer
        """
        # Determine actual number of sampled visualization nodes
        sample_count = min(num_units, self.max_nodes_per_layer)

        # X position along depth axis
        if total_layers > 1:
            x_pos = -2.5 + (5.0 * (layer_idx / (total_layers - 1)))
        else:
            x_pos = 0.0

        # Extract activation magnitudes
        act_vals = np.zeros(sample_count, dtype=np.float32)
        if activations is not None and activations.numel() > 0:
            act_flat = activations.detach().cpu().numpy().flatten()
            if len(act_flat) >= sample_count:
                step = len(act_flat) // sample_count
                act_vals = np.abs(act_flat[::step][:sample_count])
            else:
                act_vals[: len(act_flat)] = np.abs(act_flat)

        # Normalize activations
        max_act = np.max(act_vals) if np.max(act_vals) > 1e-6 else 1.0
        norm_acts = act_vals / max_act

        # Arrange in a 2D grid/cylinder on Y-Z plane
        nodes = []
        cols = int(np.ceil(np.sqrt(sample_count)))
        rows = int(np.ceil(sample_count / cols))

        spacing_y = 2.4 / max(1, cols - 1) if cols > 1 else 0.0
        spacing_z = 2.4 / max(1, rows - 1) if rows > 1 else 0.0

        for i in range(sample_count):
            r = i // cols
            c = i % cols
            y_pos = -1.2 + c * spacing_y if cols > 1 else 0.0
            z_pos = -1.2 + r * spacing_z if rows > 1 else 0.0

            nodes.append({
                "id": f"L{layer_idx}_N{i}",
                "layer_idx": layer_idx,
                "node_idx": i,
                "pos": (float(x_pos), float(y_pos), float(z_pos)),
                "activation": float(norm_acts[i]),
                "raw_activation": float(act_vals[i]),
            })

        return nodes

    def project_layer_connections(
        self,
        source_nodes: List[Dict[str, Any]],
        target_nodes: List[Dict[str, Any]],
        weight_tensor: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        """Compute visual connections between two adjacent layers."""
        connections = []
        if not source_nodes or not target_nodes:
            return connections

        num_src = len(source_nodes)
        num_tgt = len(target_nodes)

        weights = np.zeros((num_src, num_tgt), dtype=np.float32)
        if weight_tensor is not None and weight_tensor.ndim >= 2:
            w_np = weight_tensor.detach().cpu().numpy()
            if w_np.ndim > 2:
                w_np = w_np.reshape(w_np.shape[0], -1)

            # Subsample weight matrix to match visualization node grid
            src_step = max(1, w_np.shape[1] // num_src)
            tgt_step = max(1, w_np.shape[0] // num_tgt)

            sub_w = w_np[::tgt_step, ::src_step][:num_tgt, :num_src].T
            h, w = sub_w.shape
            weights[:h, :w] = sub_w

        # Normalize weights for visual encoding
        abs_weights = np.abs(weights)
        max_w = np.max(abs_weights) if np.max(abs_weights) > 1e-6 else 1.0

        # Sample highest magnitude weights up to connection budget
        for s_idx, src in enumerate(source_nodes):
            for t_idx, tgt in enumerate(target_nodes):
                w_val = float(weights[s_idx, t_idx]) if s_idx < weights.shape[0] and t_idx < weights.shape[1] else 0.1
                norm_w = float(abs(w_val) / max_w)

                if norm_w > 0.05:  # Prune negligible synaptic weights
                    connections.append({
                        "source_id": src["id"],
                        "target_id": tgt["id"],
                        "source_pos": src["pos"],
                        "target_pos": tgt["pos"],
                        "weight": w_val,
                        "normalized_weight": norm_w,
                        "sign": 1 if w_val >= 0 else -1,
                    })

        # Cap total connections
        if len(connections) > self.max_connections_per_layer:
            connections.sort(key=lambda c: c["normalized_weight"], reverse=True)
            connections = connections[: self.max_connections_per_layer]

        return connections
