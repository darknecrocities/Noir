"""Interactive 3D Neural Network Visualizer using PySide6 vector projection."""

import math
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QRadialGradient,
)
from PySide6.QtWidgets import QWidget

from noir.visualization.neural_graph import NeuralGraph


class NeuralVisualizer3D(QWidget):
    """Real-time interactive 3D viewport rendering neural network activations and weights."""

    node_selected = Signal(str, dict)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self.graph = NeuralGraph()

        # Camera Parameters
        self.camera_azimuth = 0.65  # Radians
        self.camera_elevation = 0.35  # Radians
        self.camera_distance = 6.0
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Mouse state
        self._last_mouse_pos = None
        self._is_rotating = False
        self._is_panning = False

        # Visual Effects & Pulses
        self.pulse_phase = 0.0
        self.shockwave_radius = 0.0
        self.energy_burst = 0.0
        self.selected_node_id: Optional[str] = None
        self.hovered_node: Optional[Dict[str, Any]] = None

        # Render Timer (60 FPS)
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_animate)
        self._anim_timer.start(16)

    def set_graph(self, graph: NeuralGraph) -> None:
        self.graph = graph
        self.update()

    def trigger_surprise_shock(self, intensity: float = 1.0) -> None:
        """Trigger visual perturbation on surprise event."""
        self.shockwave_radius = 1.0 * intensity
        self.update()

    def trigger_reward_pulse(self, amount: float = 1.0) -> None:
        """Trigger glowing energy pulse on reward."""
        self.energy_burst = min(2.0, 0.5 + amount * 0.2)
        self.update()

    def _on_animate(self) -> None:
        self.pulse_phase = (self.pulse_phase + 0.05) % (2 * math.pi)
        if self.shockwave_radius > 0.0:
            self.shockwave_radius = max(0.0, self.shockwave_radius - 0.03)
        if self.energy_burst > 0.0:
            self.energy_burst = max(0.0, self.energy_burst - 0.04)
        self.update()

    # 3D Math & Perspective Projection
    def _project_point(self, x: float, y: float, z: float, width: float, height: float) -> Tuple[float, float, float, bool]:
        """Project 3D world coordinate (x, y, z) onto 2D screen plane (sx, sy).

        Returns:
            (screen_x, screen_y, depth, is_in_front)
        """
        # Apply camera rotation around Y and X axes
        # 1. Azimuth rotation (around Z axis)
        cos_az = math.cos(self.camera_azimuth)
        sin_az = math.sin(self.camera_azimuth)
        x1 = x * cos_az - y * sin_az
        y1 = x * sin_az + y * cos_az
        z1 = z

        # 2. Elevation rotation (around X axis)
        cos_el = math.cos(self.camera_elevation)
        sin_el = math.sin(self.camera_elevation)
        x2 = x1
        y2 = y1 * cos_el - z1 * sin_el
        z2 = y1 * sin_el + z1 * cos_el

        # 3. Camera translation
        cam_x = x2 + self.pan_x
        cam_y = y2 + self.pan_y
        cam_z = z2 + self.camera_distance

        if cam_z <= 0.1:
            return 0.0, 0.0, cam_z, False

        # 4. Perspective division
        fov_scale = min(width, height) * 0.9
        sx = (width / 2.0) + (cam_x / cam_z) * fov_scale
        sy = (height / 2.0) - (cam_y / cam_z) * fov_scale

        return sx, sy, cam_z, True

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        try:
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

            w = float(self.width())
            h = float(self.height())

            # 1. Dark Research Laboratory Background with subtle gradient
            bg_gradient = QRadialGradient(w / 2.0, h / 2.0, max(w, h) * 0.7)
            bg_gradient.setColorAt(0.0, QColor(14, 18, 28))
            bg_gradient.setColorAt(1.0, QColor(6, 8, 14))
            painter.fillRect(self.rect(), bg_gradient)

            # 2. Draw 3D Grid Plane (Z = -1.5)
            self._draw_grid_plane(painter, w, h)

            # 3. Project and sort all nodes and connections by depth
            projected_nodes = []
            for node in self.graph.nodes:
                nx, ny, nz = node["pos"]
                # Apply shockwave jitter
                if self.shockwave_radius > 0.0:
                    jitter = math.sin(nx * 5.0 + self.pulse_phase * 4.0) * 0.08 * self.shockwave_radius
                    ny += jitter
                    nz += jitter

                sx, sy, depth, visible = self._project_point(nx, ny, nz, w, h)
                if visible:
                    projected_nodes.append({
                        "data": node,
                        "sx": sx,
                        "sy": sy,
                        "depth": depth,
                    })

            # Node coordinate lookup
            node_pos_map = {pn["data"]["id"]: pn for pn in projected_nodes}

            # 4. Draw Synaptic Connections
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            for conn in self.graph.connections:
                src = node_pos_map.get(conn["source_id"])
                tgt = node_pos_map.get(conn["target_id"])
                if not src or not tgt:
                    continue

                # Connection depth
                avg_depth = (src["depth"] + tgt["depth"]) / 2.0
                depth_scale = max(0.2, min(1.0, 5.0 / avg_depth))

                norm_w = conn["normalized_weight"]
                sign = conn["sign"]

                # Cyan for positive weight, Magenta/Purple for negative
                if sign >= 0:
                    base_color = QColor(0, 220, 255)
                else:
                    base_color = QColor(255, 60, 160)

                alpha = int(max(15, min(180, norm_w * 200 * depth_scale)))
                base_color.setAlpha(alpha)

                pen_width = max(1.0, norm_w * 2.5 * depth_scale)
                painter.setPen(QPen(base_color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
                painter.drawLine(QPointF(src["sx"], src["sy"]), QPointF(tgt["sx"], tgt["sy"]))

                # Draw traveling pulse signal on active weights
                if norm_w > 0.35:
                    t_pulse = (self.pulse_phase / (2 * math.pi) + (hash(conn["source_id"]) % 100) / 100.0) % 1.0
                    px = src["sx"] + (tgt["sx"] - src["sx"]) * t_pulse
                    py = src["sy"] + (tgt["sy"] - src["sy"]) * t_pulse

                    pulse_color = QColor(255, 255, 255, int(180 * norm_w))
                    painter.setBrush(QBrush(pulse_color))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawEllipse(QPointF(px, py), 2.0 * depth_scale, 2.0 * depth_scale)

            # 5. Sort nodes back-to-front for proper depth occlusion
            projected_nodes.sort(key=lambda item: item["depth"], reverse=True)

            # 6. Draw Nodes with Glowing Activation Fields
            for pn in projected_nodes:
                node = pn["data"]
                sx = pn["sx"]
                sy = pn["sy"]
                depth = pn["depth"]

                depth_scale = max(0.3, min(1.2, 5.0 / depth))
                act = node.get("activation", 0.0)

                base_radius = (4.0 + act * 7.0) * depth_scale
                if self.energy_burst > 0.0:
                    base_radius *= (1.0 + self.energy_burst * 0.2)

                is_selected = (node["id"] == self.selected_node_id)
                is_hovered = (self.hovered_node and node["id"] == self.hovered_node.get("id"))

                # Outer glow radial gradient
                glow_radius = base_radius * 2.2
                glow = QRadialGradient(sx, sy, glow_radius)
                if act > 0.6:
                    glow_color = QColor(0, 255, 180, int(130 * act))
                elif act > 0.2:
                    glow_color = QColor(0, 180, 255, int(100 * act))
                else:
                    glow_color = QColor(80, 100, 150, 35)

                glow.setColorAt(0.0, glow_color)
                glow.setColorAt(1.0, QColor(0, 0, 0, 0))

                painter.setBrush(QBrush(glow))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(QPointF(sx, sy), glow_radius, glow_radius)

                # Node Core
                core_grad = QRadialGradient(sx - base_radius * 0.3, sy - base_radius * 0.3, base_radius)
                if is_selected or is_hovered:
                    core_grad.setColorAt(0.0, QColor(255, 255, 255))
                    core_grad.setColorAt(0.6, QColor(255, 220, 0))
                    core_grad.setColorAt(1.0, QColor(200, 150, 0))
                    stroke_pen = QPen(QColor(255, 255, 255), 2.0)
                elif act > 0.5:
                    core_grad.setColorAt(0.0, QColor(255, 255, 255))
                    core_grad.setColorAt(0.5, QColor(0, 255, 200))
                    core_grad.setColorAt(1.0, QColor(0, 150, 120))
                    stroke_pen = QPen(QColor(0, 255, 200, 180), 1.0)
                else:
                    core_grad.setColorAt(0.0, QColor(140, 170, 220))
                    core_grad.setColorAt(1.0, QColor(30, 50, 80))
                    stroke_pen = QPen(QColor(100, 130, 180, 100), 1.0)

                painter.setBrush(QBrush(core_grad))
                painter.setPen(stroke_pen)
                painter.drawEllipse(QPointF(sx, sy), base_radius, base_radius)

            # 7. Draw Layer Labels with clean staggered layout
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            for layer_meta in self.graph.layer_metadata:
                l_idx = layer_meta["idx"]
                matched = [pn for pn in projected_nodes if pn["data"]["layer_idx"] == l_idx]
                if matched:
                    top_pn = min(matched, key=lambda pn: pn["sy"])
                    y_offset = 24 if (l_idx % 2 == 0) else 44
                    painter.setPen(QColor(0, 255, 200, 230) if l_idx == len(self.graph.layer_metadata) - 1 else QColor(130, 180, 230, 200))
                    painter.drawText(
                        int(top_pn["sx"] - 45),
                        int(top_pn["sy"] - y_offset),
                        f"{layer_meta['name']}",
                    )

            # 8. Overlay HUD (Camera & Node Telemetry)
            self._draw_hud(painter, w, h)
        finally:
            painter.end()

    def _draw_grid_plane(self, painter: QPainter, w: float, h: float) -> None:
        painter.setPen(QPen(QColor(40, 55, 80, 60), 1.0, Qt.PenStyle.DotLine))
        grid_z = -1.5
        for gx in np.linspace(-3.0, 3.0, 7):
            sx1, sy1, _, v1 = self._project_point(gx, -2.0, grid_z, w, h)
            sx2, sy2, _, v2 = self._project_point(gx, 2.0, grid_z, w, h)
            if v1 and v2:
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

        for gy in np.linspace(-2.0, 2.0, 5):
            sx1, sy1, _, v1 = self._project_point(-3.0, gy, grid_z, w, h)
            sx2, sy2, _, v2 = self._project_point(3.0, gy, grid_z, w, h)
            if v1 and v2:
                painter.drawLine(QPointF(sx1, sy1), QPointF(sx2, sy2))

    def _draw_hud(self, painter: QPainter, w: float, h: float) -> None:
        painter.setFont(QFont("Consolas", 8))
        painter.setPen(QColor(100, 140, 180, 180))

        hud_text = f"3D NEURAL VIEW | NODES: {len(self.graph.nodes)} | SYNAPSES: {len(self.graph.connections)} | ORBIT: {math.degrees(self.camera_azimuth):.0f}°/{math.degrees(self.camera_elevation):.0f}°"
        painter.drawText(12, 20, hud_text)

        if self.hovered_node:
            info_text = f"PROBE [{self.hovered_node['id']}] Act: {self.hovered_node.get('raw_activation', 0.0):.4f}"
            painter.setPen(QColor(255, 215, 0, 220))
            painter.drawText(12, 36, info_text)

    # Mouse Interaction Handlers
    def mousePressEvent(self, event) -> None:
        self._last_mouse_pos = event.position()
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_rotating = True
            self._check_node_click(event.position().x(), event.position().y())
        elif event.button() == Qt.MouseButton.RightButton:
            self._is_panning = True

    def mouseMoveEvent(self, event) -> None:
        pos = event.position()
        if self._last_mouse_pos is not None:
            dx = pos.x() - self._last_mouse_pos.x()
            dy = pos.y() - self._last_mouse_pos.y()

            if self._is_rotating:
                self.camera_azimuth += dx * 0.01
                self.camera_elevation = max(-1.4, min(1.4, self.camera_elevation - dy * 0.01))
                self.update()
            elif self._is_panning:
                self.pan_x += dx * 0.008
                self.pan_y -= dy * 0.008
                self.update()

        self._last_mouse_pos = pos
        self._check_node_hover(pos.x(), pos.y())

    def mouseReleaseEvent(self, event) -> None:
        self._is_rotating = False
        self._is_panning = False
        self._last_mouse_pos = None

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        self.camera_distance = max(2.0, min(15.0, self.camera_distance - delta * 0.005))
        self.update()

    def mouseDoubleClickEvent(self, event) -> None:
        """Reset camera view on double click."""
        self.camera_azimuth = 0.65
        self.camera_elevation = 0.35
        self.camera_distance = 6.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.update()

    def _check_node_hover(self, mx: float, my: float) -> None:
        w = float(self.width())
        h = float(self.height())
        closest = None
        min_dist = 15.0  # Pixel threshold

        for node in self.graph.nodes:
            nx, ny, nz = node["pos"]
            sx, sy, _, visible = self._project_point(nx, ny, nz, w, h)
            if visible:
                dist = math.hypot(mx - sx, my - sy)
                if dist < min_dist:
                    min_dist = dist
                    closest = node

        if closest != self.hovered_node:
            self.hovered_node = closest
            self.update()

    def _check_node_click(self, mx: float, my: float) -> None:
        if self.hovered_node:
            self.selected_node_id = self.hovered_node["id"]
            self.node_selected.emit(self.selected_node_id, self.hovered_node)
            self.update()
