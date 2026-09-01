"""Cognitive memory and real-time ingested internet knowledge base inspector view."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MemoryView(QWidget):
    """View inspecting Ingested Internet Sources, Episodic Experiences, and Semantic Knowledge."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(10)

        # 1. Top Telemetry Cards Banner
        top_cards_frame = QFrame(self)
        top_cards_frame.setStyleSheet("""
            QFrame {
                background-color: #0b0f19;
                border: 1px solid #162032;
                border-radius: 6px;
                padding: 6px;
            }
        """)
        top_cards_layout = QHBoxLayout(top_cards_frame)
        top_cards_layout.setContentsMargins(10, 4, 10, 4)
        top_cards_layout.setSpacing(20)

        self.lbl_total_sources = QLabel("TOTAL SOURCES: 0")
        self.lbl_total_sources.setStyleSheet("color: #00e5ff; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_sources)

        self.lbl_total_tokens = QLabel("TOKENS INGESTED: 0")
        self.lbl_total_tokens.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_tokens)

        self.lbl_total_chars = QLabel("CHARACTERS: 0")
        self.lbl_total_chars.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_chars)

        self.lbl_active_source = QLabel("ACTIVE SOURCE: None")
        self.lbl_active_source.setStyleSheet("color: #ff9100; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_active_source, stretch=1)

        layout.addWidget(top_cards_frame)

        # 2. Main Splitter (Top: Ingested Knowledge Sources, Bottom: Episodic & Semantic Memory)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_splitter.setChildrenCollapsible(False)

        # Top Pane: Ingested Knowledge Sources Table
        ingested_widget = QWidget()
        ingested_layout = QVBoxLayout(ingested_widget)
        ingested_layout.setContentsMargins(0, 0, 0, 0)
        ingested_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        lbl_ingested = QLabel("INGESTED INTERNET SOURCES & DATASETS (Real-Time Knowledge Ingestion)")
        lbl_ingested.setStyleSheet("color: #00e5ff; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold;")
        header_layout.addWidget(lbl_ingested)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter sources by title, URL, or type...")
        self.search_box.setStyleSheet("""
            QLineEdit {
                background-color: #121927;
                border: 1px solid #1a233a;
                border-radius: 4px;
                color: #e2e8f0;
                padding: 4px 8px;
                font-size: 11px;
                max-width: 320px;
            }
        """)
        self.search_box.textChanged.connect(self._filter_sources_table)
        header_layout.addWidget(self.search_box)
        ingested_layout.addLayout(header_layout)

        self.table_sources = QTableWidget(0, 7)
        self.table_sources.setHorizontalHeaderLabels(["STATUS", "SOURCE TYPE", "TITLE", "TOKENS", "CHARS", "URL / ORIGIN", "INGESTED TIME"])
        self.table_sources.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_sources.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_sources.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_sources.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_sources.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_sources.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table_sources.horizontalHeader().setSectionResizeMode(6, QHeaderView.ResizeMode.ResizeToContents)
        self.table_sources.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_sources.itemSelectionChanged.connect(self._on_source_selected)
        ingested_layout.addWidget(self.table_sources)

        v_splitter.addWidget(ingested_widget)

        # Bottom Pane: Horizontal Splitter for Details / Episodic Memories & Semantic Rules
        bottom_h_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Bottom Left: Source Text Snippet & Abstract Inspector
        source_detail_widget = QWidget()
        source_detail_layout = QVBoxLayout(source_detail_widget)
        source_detail_layout.setContentsMargins(0, 0, 0, 0)
        source_detail_layout.setSpacing(6)

        lbl_snippet = QLabel("INGESTED TEXT SNIPPET & ABSTRACT")
        lbl_snippet.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        source_detail_layout.addWidget(lbl_snippet)

        self.txt_snippet = QTextEdit()
        self.txt_snippet.setReadOnly(True)
        self.txt_snippet.setStyleSheet("""
            QTextEdit {
                background-color: #0b0f19;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_snippet.setPlaceholderText("Select an ingested source above to inspect its text contents and abstract...")
        source_detail_layout.addWidget(self.txt_snippet)
        bottom_h_splitter.addWidget(source_detail_widget)

        # Bottom Right: Episodic Memories (Surprises & Breakthroughs)
        episodic_widget = QWidget()
        episodic_layout = QVBoxLayout(episodic_widget)
        episodic_layout.setContentsMargins(0, 0, 0, 0)
        episodic_layout.setSpacing(6)

        lbl_ep = QLabel("EPISODIC MEMORY (Salient Discoveries & Shocks)")
        lbl_ep.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        episodic_layout.addWidget(lbl_ep)

        self.table_episodic = QTableWidget(0, 4)
        self.table_episodic.setHorizontalHeaderLabels(["STEP", "TYPE", "IMPORTANCE", "DESCRIPTION"])
        self.table_episodic.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_episodic.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        episodic_layout.addWidget(self.table_episodic)
        bottom_h_splitter.addWidget(episodic_widget)

        bottom_h_splitter.setSizes([450, 450])
        v_splitter.addWidget(bottom_h_splitter)

        v_splitter.setSizes([380, 240])
        layout.addWidget(v_splitter)

        # Internal source records lookup: url -> dict
        self._sources_registry: Dict[str, Dict[str, Any]] = {}
        self._total_tokens_count = 0
        self._total_chars_count = 0

    def add_or_update_source(self, data: Dict[str, Any]) -> None:
        """Register or update an ingested internet resource or dataset."""
        url = data.get("url", "")
        title = data.get("title", "Unknown Source")
        source_type = data.get("source_type", "Web Stream")
        tokens = data.get("token_count", 0)
        chars = data.get("character_count", 0)
        snippet = data.get("text_snippet", data.get("full_text", ""))
        timestamp = data.get("timestamp", datetime.now().timestamp())
        status = data.get("status", "INGESTED")

        time_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

        is_new = url not in self._sources_registry
        self._sources_registry[url] = {
            "title": title,
            "url": url,
            "source_type": source_type,
            "token_count": tokens,
            "character_count": chars,
            "text_snippet": snippet,
            "timestamp": timestamp,
            "time_str": time_str,
            "status": status,
        }

        if is_new:
            self._total_tokens_count += tokens
            self._total_chars_count += chars

            # Insert at top of table
            row = 0
            self.table_sources.insertRow(row)

            item_status = QTableWidgetItem(status)
            item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            if status == "LEARNING" or status == "ACTIVE":
                item_status.setForeground(QColor(0, 229, 255))
            else:
                item_status.setForeground(QColor(100, 255, 218))

            item_type = QTableWidgetItem(source_type)
            item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_title = QTableWidgetItem(title)
            item_title.setToolTip(title)

            item_tokens = QTableWidgetItem(f"{tokens:,}")
            item_tokens.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            item_chars = QTableWidgetItem(f"{chars:,}")
            item_chars.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            item_url = QTableWidgetItem(url)
            item_url.setToolTip(url)
            item_url.setForeground(QColor(140, 180, 240))

            item_time = QTableWidgetItem(time_str)
            item_time.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            self.table_sources.setItem(row, 0, item_status)
            self.table_sources.setItem(row, 1, item_type)
            self.table_sources.setItem(row, 2, item_title)
            self.table_sources.setItem(row, 3, item_tokens)
            self.table_sources.setItem(row, 4, item_chars)
            self.table_sources.setItem(row, 5, item_url)
            self.table_sources.setItem(row, 6, item_time)

        self._update_header_cards()

    def set_active_learning_source(self, title: str, url: Optional[str] = None) -> None:
        """Mark currently active learning webpage/source in the table."""
        self.lbl_active_source.setText(f"ACTIVE: {title[:40]}")
        for row in range(self.table_sources.rowCount()):
            item_title = self.table_sources.item(row, 2)
            item_status = self.table_sources.item(row, 0)
            if item_title and item_status:
                if title in item_title.text() or (url and url in self.table_sources.item(row, 5).text()):
                    item_status.setText("ACTIVE")
                    item_status.setForeground(QColor(0, 229, 255))
                elif item_status.text() == "ACTIVE":
                    item_status.setText("MASTERED")
                    item_status.setForeground(QColor(100, 255, 218))

    def _update_header_cards(self) -> None:
        count = len(self._sources_registry)
        self.lbl_total_sources.setText(f"TOTAL SOURCES: {count:,}")
        self.lbl_total_tokens.setText(f"TOKENS INGESTED: {self._total_tokens_count:,}")
        self.lbl_total_chars.setText(f"CHARACTERS: {self._total_chars_count:,}")

    def _on_source_selected(self) -> None:
        selected_rows = self.table_sources.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        item_url = self.table_sources.item(row, 5)
        if item_url:
            url = item_url.text()
            data = self._sources_registry.get(url, {})
            snippet = data.get("text_snippet", "")
            title = data.get("title", "")
            tokens = data.get("token_count", 0)
            chars = data.get("character_count", 0)
            src_type = data.get("source_type", "")
            status = data.get("status", "INGESTED")

            full_info = (
                f"SOURCE: {title}\n"
                f"ORIGIN URL: {url}\n"
                f"TYPE: {src_type} | STATUS: {status} | TOKENS: {tokens:,} | CHARACTERS: {chars:,}\n"
                f"{'=' * 70}\n\n"
                f"{snippet}"
            )
            self.txt_snippet.setText(full_info)

    def _filter_sources_table(self, query: str) -> None:
        query = query.lower().strip()
        for row in range(self.table_sources.rowCount()):
            match = False
            if not query:
                match = True
            else:
                for col in (1, 2, 5):
                    item = self.table_sources.item(row, col)
                    if item and query in item.text().lower():
                        match = True
                        break
            self.table_sources.setRowHidden(row, not match)

    def populate_episodic(self, episodes: list) -> None:
        self.table_episodic.setRowCount(0)
        for ep in episodes:
            row = self.table_episodic.rowCount()
            self.table_episodic.insertRow(row)

            self.table_episodic.setItem(row, 0, QTableWidgetItem(str(ep.get("step", 0))))
            self.table_episodic.setItem(row, 1, QTableWidgetItem(ep.get("event_type", "")))
            self.table_episodic.setItem(row, 2, QTableWidgetItem(f"{ep.get('importance', 1.0):.2f}"))
            self.table_episodic.setItem(row, 3, QTableWidgetItem(ep.get("description", "")))
