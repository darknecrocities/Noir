"""Live streaming Event Timeline widget with high-performance batch buffering."""

from collections import deque
from datetime import datetime
from typing import List
from PySide6.QtCore import QTimer, Qt
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

        # High-performance event ring buffer
        self._event_queue: deque[NoirEvent] = deque(maxlen=200)
        self._max_displayed_rows = 100

        # Batch UI flush timer (updates at 10 Hz to prevent Qt event loop starvation)
        self._flush_timer = QTimer(self)
        self._flush_timer.setInterval(100)
        self._flush_timer.timeout.connect(self._flush_pending_events)
        self._flush_timer.start()

    def add_event(self, event: NoirEvent) -> None:
        """Enqueue event for throttled batch UI flush."""
        # For high-frequency weights/metrics events, sample to avoid flooding
        if event.event_type in (EventType.WEIGHTS_UPDATED, EventType.METRICS_UPDATED):
            if event.training_step % 10 != 0:
                return

        self._event_queue.append(event)

    def _flush_pending_events(self) -> None:
        """Flush enqueued events into table in a single atomic UI operation."""
        if not self._event_queue:
            return

        events_to_add: List[NoirEvent] = []
        while self._event_queue:
            events_to_add.append(self._event_queue.popleft())

        self.table.setUpdatesEnabled(False)
        try:
            for event in reversed(events_to_add):
                self.table.insertRow(0)

                time_str = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
                item_time = QTableWidgetItem(time_str)
                item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

                item_type = QTableWidgetItem(event.event_type.value)
                item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

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

                # Format details string cleanly
                payload_summary = ", ".join(
                    f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in list(event.payload.items())[:3]
                )
                item_details = QTableWidgetItem(payload_summary)

                self.table.setItem(0, 0, item_time)
                self.table.setItem(0, 1, item_type)
                self.table.setItem(0, 2, item_step)
                self.table.setItem(0, 3, item_details)

            # Enforce max row cap
            while self.table.rowCount() > self._max_displayed_rows:
                self.table.removeRow(self.table.rowCount() - 1)
        finally:
            self.table.setUpdatesEnabled(True)
