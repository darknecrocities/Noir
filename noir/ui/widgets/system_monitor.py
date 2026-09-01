"""Hardware and training telemetry status bar widget."""

from typing import Dict
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from noir.events.event import NoirEvent
from noir.events.event_types import EventType


class SystemMonitorBar(QFrame):
    """Bottom telemetry bar displaying real hardware usage and training counters."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setFixedHeight(36)
        self.setStyleSheet("""
            QFrame {
                background-color: #0b0f19;
                border-top: 1px solid #1a2333;
                color: #8da2c0;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 11px;
            }
            QLabel {
                padding: 0 8px;
            }
            QLabel.highlight {
                color: #00ffcc;
                font-weight: bold;
            }
            QLabel.alert {
                color: #ff3366;
                font-weight: bold;
            }
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(16)

        self.lbl_status = QLabel("STATUS: READY")
        self.lbl_status.setStyleSheet("color: #00ffcc; font-weight: bold;")

        self.lbl_step = QLabel("STEP: 0")
        self.lbl_epoch = QLabel("EPOCH: 0")
        self.lbl_loss = QLabel("LOSS: --")
        self.lbl_reward = QLabel("REWARD: --")
        self.lbl_cpu = QLabel("CPU: 0%")
        self.lbl_ram = QLabel("RAM: 0%")
        self.lbl_gpu = QLabel("GPU: N/A")
        self.lbl_device = QLabel("DEVICE: CPU")

        layout.addWidget(self.lbl_status)
        layout.addWidget(self.lbl_step)
        layout.addWidget(self.lbl_epoch)
        layout.addWidget(self.lbl_loss)
        layout.addWidget(self.lbl_reward)
        layout.addStretch()
        layout.addWidget(self.lbl_cpu)
        layout.addWidget(self.lbl_ram)
        layout.addWidget(self.lbl_gpu)
        layout.addWidget(self.lbl_device)

    def update_status(self, status: str) -> None:
        self.lbl_status.setText(f"STATUS: {status.upper()}")
        if status.upper() in ("RUNNING", "LIVE"):
            self.lbl_status.setStyleSheet("color: #00ffcc; font-weight: bold;")
        elif status.upper() == "PAUSED":
            self.lbl_status.setStyleSheet("color: #ffbb00; font-weight: bold;")
        elif status.upper() == "ERROR":
            self.lbl_status.setStyleSheet("color: #ff3366; font-weight: bold;")
        else:
            self.lbl_status.setStyleSheet("color: #8da2c0; font-weight: normal;")

    def update_training_metrics(self, step: int, epoch: int, loss: float = None, reward: float = None) -> None:
        self.lbl_step.setText(f"STEP: {step:,}")
        self.lbl_epoch.setText(f"EPOCH: {epoch}")
        if loss is not None:
            self.lbl_loss.setText(f"LOSS: {loss:.4f}")
        if reward is not None:
            self.lbl_reward.setText(f"REWARD: {reward:.2f}")

    def update_hardware(self, cpu: float, ram: float, gpu: float = None, vram: float = None, cuda: bool = False) -> None:
        self.lbl_cpu.setText(f"CPU: {cpu:.0f}%")
        self.lbl_ram.setText(f"RAM: {ram:.0f}%")
        if cuda:
            self.lbl_gpu.setText(f"GPU: {vram:.1f}MB")
            self.lbl_device.setText("DEVICE: CUDA")
            self.lbl_device.setStyleSheet("color: #76b900; font-weight: bold;")
        else:
            self.lbl_gpu.setText("GPU: OFF")
            self.lbl_device.setText("DEVICE: CPU")
