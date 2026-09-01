"""Affective / Emotion state telemetry panel."""

from typing import Dict
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)


class EmotionGauge(QWidget):
    """Single affective dimension gauge with name, bar, and exact value."""

    def __init__(self, name: str, color_hex: str, parent: QWidget = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)

        self.lbl_name = QLabel(name.ljust(12))
        self.lbl_name.setFixedWidth(90)
        self.lbl_name.setStyleSheet("color: #9cb3d0; font-family: 'Consolas', monospace; font-size: 11px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(50)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(10)
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #121927;
                border: 1px solid #1e293b;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {color_hex};
                border-radius: 3px;
            }}
        """)

        self.lbl_val = QLabel("0.50")
        self.lbl_val.setFixedWidth(40)
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_val.setStyleSheet(f"color: {color_hex}; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")

        layout.addWidget(self.lbl_name)
        layout.addWidget(self.bar)
        layout.addWidget(self.lbl_val)

    def set_value(self, val: float) -> None:
        clamped = max(0.0, min(1.0, float(val)))
        self.bar.setValue(int(clamped * 100))
        self.lbl_val.setText(f"{clamped:.2f}")


class EmotionPanel(QFrame):
    """Panel rendering all 8 mathematical affective state dimensions."""

    EMOTIONS = [
        ("Curiosity", "#00e5ff"),
        ("Confidence", "#ffd700"),
        ("Uncertainty", "#b388ff"),
        ("Frustration", "#ff5252"),
        ("Anticipation", "#40c4ff"),
        ("Satisfaction", "#00e676"),
        ("Caution", "#ff9100"),
        ("Persistence", "#64ffda"),
    ]

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
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        title = QLabel("AFFECTIVE STATE (E_t)")
        title.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; letter-spacing: 1px;")
        layout.addWidget(title)

        self.gauges: Dict[str, EmotionGauge] = {}
        for name, color in self.EMOTIONS:
            gauge = EmotionGauge(name, color, self)
            self.gauges[name.lower()] = gauge
            layout.addWidget(gauge)

    def update_state(self, state_dict: Dict[str, float]) -> None:
        """Update all emotion gauges from state dictionary."""
        for key, gauge in self.gauges.items():
            if key in state_dict:
                gauge.set_value(state_dict[key])
