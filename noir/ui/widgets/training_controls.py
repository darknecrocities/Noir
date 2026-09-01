"""Training controls and mode selector bar."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)


class TrainingControls(QFrame):
    """Control panel providing Start, Pause, Resume, Stop, and Checkpoint triggers."""

    start_clicked = Signal(str, float)  # (mode, lr)
    pause_clicked = Signal()
    resume_clicked = Signal()
    stop_clicked = Signal()
    checkpoint_clicked = Signal()

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedHeight(54)
        self.setStyleSheet("""
            QFrame {
                background-color: #0d121f;
                border: 1px solid #1a233a;
                border-radius: 6px;
            }
            QPushButton {
                background-color: #1a2436;
                color: #e2e8f0;
                border: 1px solid #2d3b55;
                border-radius: 4px;
                padding: 6px 14px;
                font-family: 'Segoe UI', sans-serif;
                font-weight: bold;
                font-size: 11px;
            }
            QPushButton:hover {
                background-color: #24324a;
                border-color: #00e5ff;
                color: #ffffff;
            }
            QPushButton#btn_start {
                background-color: #00875a;
                border-color: #00b875;
                color: #ffffff;
            }
            QPushButton#btn_start:hover {
                background-color: #00a86b;
            }
            QPushButton#btn_stop {
                background-color: #a81c3a;
                border-color: #de350b;
                color: #ffffff;
            }
            QPushButton#btn_stop:hover {
                background-color: #c92a4e;
            }
            QComboBox, QDoubleSpinBox {
                background-color: #121927;
                border: 1px solid #24324a;
                border-radius: 4px;
                color: #00e5ff;
                padding: 4px 8px;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QLabel {
                color: #8da2c0;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(10)

        # Mode / Real Dataset Selection
        layout.addWidget(QLabel("DATASET & MODE:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("ALL: Autonomous Master Research (Live Internet + Multimodal)", "all:autonomous")
        self.combo_mode.addItem("Open Web Internet Stream (Transformer LLM)", "llm:open_web")
        self.combo_mode.addItem("Real Digits (8x8 Handwritten)", "supervised:digits")
        self.combo_mode.addItem("Fashion-MNIST (Clothing Benchmark)", "supervised:fashion_mnist")
        self.combo_mode.addItem("Wine Analysis (Chemical Sensors)", "supervised:wine")
        self.combo_mode.addItem("Breast Cancer (Biomedical FNA)", "supervised:breast_cancer")
        self.combo_mode.addItem("MNIST (Classic 28x28 Digits)", "supervised:mnist")
        self.combo_mode.addItem("CIFAR-10 (Color Objects)", "supervised:cifar10")
        self.combo_mode.addItem("RL: PPO GridWorld Navigation", "rl:gridworld")
        layout.addWidget(self.combo_mode)

        # Learning Rate
        layout.addWidget(QLabel("LR:"))
        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setRange(0.00001, 0.1)
        self.spin_lr.setDecimals(5)
        self.spin_lr.setSingleStep(0.0001)
        self.spin_lr.setValue(0.001)
        layout.addWidget(self.spin_lr)

        layout.addStretch()

        # Action Buttons
        self.btn_start = QPushButton("START")
        self.btn_start.setObjectName("btn_start")
        self.btn_start.clicked.connect(self._on_start)
        layout.addWidget(self.btn_start)

        self.btn_pause = QPushButton("PAUSE")
        self.btn_pause.clicked.connect(self.pause_clicked.emit)
        layout.addWidget(self.btn_pause)

        self.btn_resume = QPushButton("RESUME")
        self.btn_resume.clicked.connect(self.resume_clicked.emit)
        layout.addWidget(self.btn_resume)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setObjectName("btn_stop")
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.btn_stop)

        self.btn_ckpt = QPushButton("SAVE CHECKPOINT")
        self.btn_ckpt.clicked.connect(self.checkpoint_clicked.emit)
        layout.addWidget(self.btn_ckpt)

    def _on_start(self) -> None:
        mode = self.combo_mode.currentData()
        lr = self.spin_lr.value()
        self.start_clicked.emit(mode, lr)
