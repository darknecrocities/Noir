"""Cognitive memory and knowledge base inspector view."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MemoryView(QWidget):
    """View inspecting Episodic Experiences and Semantic Knowledge."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Column: Episodic Memories Table
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        lbl_ep = QLabel("EPISODIC MEMORY (Salient Experiences & Discoveries)")
        lbl_ep.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_ep)

        self.table_episodic = QTableWidget(0, 4)
        self.table_episodic.setHorizontalHeaderLabels(["STEP", "TYPE", "IMPORTANCE", "DESCRIPTION"])
        self.table_episodic.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_episodic.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.table_episodic)

        splitter.addWidget(left_widget)

        # Right Column: Semantic Concepts
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        lbl_sem = QLabel("SEMANTIC KNOWLEDGE BASE (Rules & Invariants)")
        lbl_sem.setStyleSheet("color: #b388ff; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold;")
        right_layout.addWidget(lbl_sem)

        self.txt_semantic = QTextEdit()
        self.txt_semantic.setReadOnly(True)
        self.txt_semantic.setStyleSheet("""
            QTextEdit {
                background-color: #0b0f19;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_semantic.setPlaceholderText("Semantic concepts and learned invariants will appear here...")
        right_layout.addWidget(self.txt_semantic)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 400])

        layout.addWidget(splitter)

    def populate_episodic(self, episodes: list) -> None:
        self.table_episodic.setRowCount(0)
        for ep in episodes:
            row = self.table_episodic.rowCount()
            self.table_episodic.insertRow(row)

            self.table_episodic.setItem(row, 0, QTableWidgetItem(str(ep.get("step", 0))))
            self.table_episodic.setItem(row, 1, QTableWidgetItem(ep.get("event_type", "")))
            self.table_episodic.setItem(row, 2, QTableWidgetItem(f"{ep.get('importance', 1.0):.2f}"))
            self.table_episodic.setItem(row, 3, QTableWidgetItem(ep.get("description", "")))
