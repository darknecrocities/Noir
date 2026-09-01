"""Startup crash and checkpoint recovery dialog."""

from typing import Any, Dict
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class RecoveryDialog(QDialog):
    """Interactive modal dialog offering session recovery options."""

    def __init__(self, recovery_info: Dict[str, Any], parent: QWidget = None):
        super().__init__(parent)
        self.setWindowTitle("Project NOIR — Session Recovery Detected")
        self.setFixedSize(520, 360)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.selected_action = "new"  # "resume", "load_only", "new"

        self.setStyleSheet("""
            QDialog {
                background-color: #0d121f;
                color: #e2e8f0;
                font-family: 'Segoe UI', sans-serif;
            }
            QFrame#card {
                background-color: #121927;
                border: 1px solid #1a233a;
                border-radius: 6px;
                padding: 12px;
            }
            QLabel#title {
                color: #00e5ff;
                font-size: 16px;
                font-weight: bold;
            }
            QLabel#subtitle {
                color: #8da2c0;
                font-size: 12px;
            }
            QLabel.field {
                color: #64ffda;
                font-family: 'Consolas', monospace;
                font-size: 12px;
            }
            QPushButton {
                background-color: #1a2436;
                color: #e2e8f0;
                border: 1px solid #2d3b55;
                border-radius: 4px;
                padding: 8px 16px;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #24324a;
                border-color: #00e5ff;
            }
            QPushButton#btn_resume {
                background-color: #00875a;
                border-color: #00b875;
                color: #ffffff;
            }
            QPushButton#btn_resume:hover {
                background-color: #00a86b;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        lbl_title = QLabel("RECOVERY CHECKPOINT DETECTED")
        lbl_title.setObjectName("title")
        lbl_sub = QLabel("Project NOIR discovered an existing valid checkpoint from a previous session.")
        lbl_sub.setObjectName("subtitle")
        lbl_sub.setWordWrap(True)

        layout.addWidget(lbl_title)
        layout.addWidget(lbl_sub)

        # Info Card
        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(6)

        exp_id = recovery_info.get("experiment_id", "Unknown")
        step = recovery_info.get("step", 0)
        epoch = recovery_info.get("epoch", 0)
        metrics = recovery_info.get("metrics", {})

        card_layout.addWidget(QLabel(f"EXPERIMENT:  {exp_id}"))
        card_layout.addWidget(QLabel(f"GLOBAL STEP: {step:,}"))
        card_layout.addWidget(QLabel(f"EPOCH:       {epoch}"))

        metrics_summary = ", ".join(f"{k}={v:.4f}" for k, v in list(metrics.items())[:3])
        card_layout.addWidget(QLabel(f"METRICS:     {metrics_summary or 'N/A'}"))

        layout.addWidget(card)

        layout.addStretch()

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(10)

        btn_resume = QPushButton("Resume Training")
        btn_resume.setObjectName("btn_resume")
        btn_resume.clicked.connect(self._on_resume)

        btn_load = QPushButton("Load Without Resuming")
        btn_load.clicked.connect(self._on_load)

        btn_new = QPushButton("Start New Experiment")
        btn_new.clicked.connect(self._on_new)

        btn_layout.addWidget(btn_resume)
        btn_layout.addWidget(btn_load)
        btn_layout.addWidget(btn_new)

        layout.addLayout(btn_layout)

    def _on_resume(self) -> None:
        self.selected_action = "resume"
        self.accept()

    def _on_load(self) -> None:
        self.selected_action = "load_only"
        self.accept()

    def _on_new(self) -> None:
        self.selected_action = "new"
        self.reject()
