"""Central Dashboard View integrating 3D Neural Net, Emotion, Metrics, and Controls."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from noir.ui.widgets.emotion_panel import EmotionPanel
from noir.ui.widgets.event_timeline import EventTimeline
from noir.ui.widgets.metrics_panel import MetricsPanel
from noir.ui.widgets.training_controls import TrainingControls
from noir.visualization.visualizer_3d import NeuralVisualizer3D


class DashboardView(QWidget):
    """Main research laboratory command dashboard."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Controls bar at the very top
        self.controls = TrainingControls(self)
        main_layout.addWidget(self.controls)

        # Main splitter (Top: 3D Net + Emotion, Bottom: Telemetry + Timeline)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setChildrenCollapsible(False)

        # Top Section: 3D Neural View (Left/Center) + Emotion Panel & Strategist (Right)
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(8)

        # 3D Visualizer Container
        self.visualizer_3d = NeuralVisualizer3D(top_widget)
        top_layout.addWidget(self.visualizer_3d, stretch=3)

        # Right Side: Emotion Panel
        right_box = QVBoxLayout()
        right_box.setSpacing(8)

        self.emotion_panel = EmotionPanel(top_widget)
        right_box.addWidget(self.emotion_panel)

        # Hypothesis Banner
        self.hypothesis_card = QFrame()
        self.hypothesis_card.setStyleSheet("""
            QFrame {
                background-color: #121927;
                border: 1px solid #1e293b;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        hyp_layout = QVBoxLayout(self.hypothesis_card)
        hyp_layout.setContentsMargins(8, 6, 8, 6)
        hyp_layout.setSpacing(4)

        hyp_title = QLabel("STRATEGIST HYPOTHESIS")
        hyp_title.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold;")
        self.lbl_hypothesis = QLabel("Observing initial dynamics...")
        self.lbl_hypothesis.setStyleSheet("color: #e2e8f0; font-size: 11px;")
        self.lbl_hypothesis.setWordWrap(True)

        hyp_layout.addWidget(hyp_title)
        hyp_layout.addWidget(self.lbl_hypothesis)
        right_box.addWidget(self.hypothesis_card)

        top_layout.addLayout(right_box, stretch=1)
        v_splitter.addWidget(top_widget)

        # Bottom Section: Metrics Panel (Left) + Event Timeline (Right)
        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)

        self.metrics_panel = MetricsPanel(bottom_widget)
        bottom_layout.addWidget(self.metrics_panel, stretch=2)

        self.event_timeline = EventTimeline(bottom_widget)
        bottom_layout.addWidget(self.event_timeline, stretch=2)

        v_splitter.addWidget(bottom_widget)
        v_splitter.setSizes([450, 250])

        main_layout.addWidget(v_splitter)

    def set_hypothesis_text(self, text: str) -> None:
        self.lbl_hypothesis.setText(text)
