"""Real-time training metrics plots using PyQtGraph with anti-distortion guardrails."""

from typing import Dict, List, Optional
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

# Configure PyQtGraph globally for cyberpunk dark aesthetics
pg.setConfigOption("background", "#0d121f")
pg.setConfigOption("foreground", "#8da2c0")
pg.setConfigOption("antialias", True)


class MetricsPanel(QFrame):
    """Dual plot widgets showing live Loss curves and Accuracy/Reward progression with bounded scaling."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #0d121f;
                border: 1px solid #1a233a;
                border-radius: 6px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel("REAL-TIME TRAINING TELEMETRY")
        title.setStyleSheet("color: #00e5ff; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        # Plot 1: Loss & Validation Loss
        self.plot_loss = pg.PlotWidget()
        self.plot_loss.showGrid(x=True, y=True, alpha=0.15)
        self.plot_loss.setLabel("left", "Loss (Train / Val)", color="#8da2c0", size="10pt")
        self.plot_loss.setLabel("bottom", "Training Step", color="#8da2c0", size="10pt")
        self.plot_loss.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_loss.setLimits(xMin=0, yMin=0.0, yMax=15.0)

        self.curve_loss = self.plot_loss.plot(pen=pg.mkPen(color="#ff3366", width=2.0), name="Train Loss")
        self.curve_val_loss = self.plot_loss.plot(pen=pg.mkPen(color="#ffaa00", width=1.8, style=Qt.PenStyle.DashLine), name="Val Loss")
        layout.addWidget(self.plot_loss)

        # Plot 2: Accuracy, Reward, or Perplexity
        self.plot_reward = pg.PlotWidget()
        self.plot_reward.showGrid(x=True, y=True, alpha=0.15)
        self.plot_reward.setLabel("left", "Reward / Acc / PPL", color="#8da2c0", size="10pt")
        self.plot_reward.setLabel("bottom", "Training Step", color="#8da2c0", size="10pt")
        self.plot_reward.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot_reward.setLimits(xMin=0, yMin=0.0, yMax=100.0)

        self.curve_reward = self.plot_reward.plot(pen=pg.mkPen(color="#00e676", width=2.0), name="Metric Curve")
        layout.addWidget(self.plot_reward)

        self.loss_steps: List[int] = []
        self.loss_values: List[float] = []
        self.val_steps: List[int] = []
        self.val_values: List[float] = []

        self.reward_steps: List[int] = []
        self.reward_values: List[float] = []

    def add_loss_point(self, step: int, loss: float, val_loss: Optional[float] = None) -> None:
        """Add bounded loss point and update curves without viewbox distortion."""
        clean_loss = float(max(0.0, min(15.0, loss)))
        self.loss_steps.append(step)
        self.loss_values.append(clean_loss)
        if len(self.loss_steps) > 500:
            self.loss_steps.pop(0)
            self.loss_values.pop(0)

        self.curve_loss.setData(self.loss_steps, self.loss_values)

        if val_loss is not None:
            clean_val = float(max(0.0, min(15.0, val_loss)))
            self.val_steps.append(step)
            self.val_values.append(clean_val)
            if len(self.val_steps) > 500:
                self.val_steps.pop(0)
                self.val_values.pop(0)
            self.curve_val_loss.setData(self.val_steps, self.val_values)

        # Keep plot tightly auto-ranged to the latest 500 steps
        if self.loss_steps:
            min_x = self.loss_steps[0]
            max_x = max(min_x + 10, self.loss_steps[-1])
            max_y = max(2.0, max(self.loss_values) * 1.15)
            self.plot_loss.setRange(xRange=(min_x, max_x), yRange=(0.0, min(15.0, max_y)), padding=0.02)

    def add_reward_point(self, step: int, reward: float) -> None:
        """Add bounded metric point."""
        clean_reward = float(max(0.0, min(100.0, reward)))
        self.reward_steps.append(step)
        self.reward_values.append(clean_reward)
        if len(self.reward_steps) > 500:
            self.reward_steps.pop(0)
            self.reward_values.pop(0)

        self.curve_reward.setData(self.reward_steps, self.reward_values)
        if self.reward_steps:
            min_x = self.reward_steps[0]
            max_x = max(min_x + 10, self.reward_steps[-1])
            max_y = max(1.0, max(self.reward_values) * 1.15)
            self.plot_reward.setRange(xRange=(min_x, max_x), yRange=(0.0, min(100.0, max_y)), padding=0.02)

    def clear(self) -> None:
        self.loss_steps.clear()
        self.loss_values.clear()
        self.val_steps.clear()
        self.val_values.clear()
        self.reward_steps.clear()
        self.reward_values.clear()
        self.curve_loss.setData([], [])
        self.curve_val_loss.setData([], [])
        self.curve_reward.setData([], [])
