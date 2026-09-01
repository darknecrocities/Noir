"""Live streaming Event Timeline widget."""

from datetime import datetime
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from noir.events.event import NoirEvent
from noir.events.event_types import EventType


class EventTimeline(QFrame):
    """Real-time scrolling table of training events, surprises, and checkpoints."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        self.setStyleSheet("""
            QFrame {
                background-color: #0d121f;
                border: 1px solid #1a233a;
                border-radius: 6px;
            }
            QTableWidget {
                background-color: #0b0f19;
                border: 1px solid #162032;
                gridline-color: #1a233a;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
            QHeaderView::section {
                background-color: #121927;
                color: #64ffda;
                padding: 4px;
                border: 1px solid #1a233a;
                font-family: 'Segoe UI', sans-serif;
                font-size: 10px;
                font-weight: bold;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)

        title = QLabel("EVENT STREAM TIMELINE")
        title.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        layout.addWidget(title)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["TIME", "EVENT", "STEP", "DETAILS"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)

    def add_event(self, event: NoirEvent) -> None:
        """Insert new event at the top of timeline."""
        row = 0
        self.table.insertRow(row)

        time_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        item_time = QTableWidgetItem(time_str)
        item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        item_type = QTableWidgetItem(event.event_type.value)
        item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Style based on event type
        if event.event_type == EventType.SURPRISE_DETECTED:
            item_type.setForeground(Qt.GlobalColor.magenta)
        elif event.event_type in (EventType.CHECKPOINT_CREATED, EventType.CHECKPOINT_LOADED):
            item_type.setForeground(Qt.GlobalColor.cyan)
        elif event.event_type == EventType.REWARD_RECEIVED:
            item_type.setForeground(Qt.GlobalColor.green)
        elif event.event_type == EventType.ERROR_OCCURRED:
            item_type.setForeground(Qt.GlobalColor.red)
        elif event.event_type == EventType.HYPOTHESIS_GENERATED:
            item_type.setForeground(Qt.GlobalColor.yellow)

        item_step = QTableWidgetItem(str(event.training_step))
        item_step.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

        # Format details string
        payload_summary = ", ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in list(event.payload.items())[:3])
        item_details = QTableWidgetItem(payload_summary)

        self.table.setItem(row, 0, item_time)
        self.table.setItem(row, 1, item_type)
        self.table.setItem(row, 2, item_step)
        self.table.setItem(row, 3, item_details)

        # Keep max 200 rows
        if self.table.rowCount() > 200:
            self.table.removeRow(self.table.rowCount() - 1)
