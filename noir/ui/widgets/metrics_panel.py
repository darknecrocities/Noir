"""Real-time training metrics plots using PyQtGraph."""

from typing import Dict, List
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

# Configure PyQtGraph globally for cyberpunk dark aesthetics
pg.setConfigOption("background", "#0d121f")
pg.setConfigOption("foreground", "#8da2c0")
pg.setConfigOption("antialias", True)


class MetricsPanel(QFrame):
    """Dual plot widgets showing live Loss curves and Accuracy/Reward progression."""

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

        # Plot 1: Loss & Gradient Norm
        self.plot_loss = pg.PlotWidget()
        self.plot_loss.showGrid(x=True, y=True, alpha=0.15)
        self.plot_loss.setLabel("left", "Loss", color="#8da2c0", size="10pt")
        self.plot_loss.setLabel("bottom", "Training Step", color="#8da2c0", size="10pt")
        self.curve_loss = self.plot_loss.plot(pen=pg.mkPen(color="#ff3366", width=2.0), name="Loss")
        self.curve_val_loss = self.plot_loss.plot(pen=pg.mkPen(color="#ff9100", width=1.5, style=Qt.PenStyle.DashLine), name="Val Loss")
        layout.addWidget(self.plot_loss)

        # Plot 2: Accuracy or Reward
        self.plot_reward = pg.PlotWidget()
        self.plot_reward.showGrid(x=True, y=True, alpha=0.15)
        self.plot_reward.setLabel("left", "Reward / Acc", color="#8da2c0", size="10pt")
        self.plot_reward.setLabel("bottom", "Step / Epoch", color="#8da2c0", size="10pt")
        self.curve_reward = self.plot_reward.plot(pen=pg.mkPen(color="#00e676", width=2.0), name="Reward/Acc")
        layout.addWidget(self.plot_reward)

        self.loss_steps: List[int] = []
        self.loss_values: List[float] = []
        self.reward_steps: List[int] = []
        self.reward_values: List[float] = []

    def add_loss_point(self, step: int, loss: float) -> None:
        self.loss_steps.append(step)
        self.loss_values.append(loss)
        if len(self.loss_steps) > 500:
            self.loss_steps.pop(0)
            self.loss_values.pop(0)
        self.curve_loss.setData(self.loss_steps, self.loss_values)

    def add_reward_point(self, step: int, reward: float) -> None:
        self.reward_steps.append(step)
        self.reward_values.append(reward)
        if len(self.reward_steps) > 500:
            self.reward_steps.pop(0)
            self.reward_values.pop(0)
        self.curve_reward.setData(self.reward_steps, self.reward_values)

    def clear(self) -> None:
        self.loss_steps.clear()
        self.loss_values.clear()
        self.reward_steps.clear()
        self.reward_values.clear()
        self.curve_loss.setData([], [])
        self.curve_reward.setData([], [])
