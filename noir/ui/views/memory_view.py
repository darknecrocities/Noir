"""Cognitive memory, real-time current learning source, upcoming queue, and finished knowledge view."""

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
    QProgressBar,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class MemoryView(QWidget):
    """View inspecting Current Active Learning, Upcoming Batch Queue, Finished Knowledge, and Episodic Memories."""

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

        self.lbl_total_sources = QLabel("TOTAL INGESTED: 0")
        self.lbl_total_sources.setStyleSheet("color: #00e5ff; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_sources)

        self.lbl_total_finished = QLabel("FINISHED / MASTERED: 0")
        self.lbl_total_finished.setStyleSheet("color: #00e676; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_finished)

        self.lbl_total_queue = QLabel("IN QUEUE: 0")
        self.lbl_total_queue.setStyleSheet("color: #ffb300; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_queue)

        self.lbl_total_tokens = QLabel("TOKENS: 0")
        self.lbl_total_tokens.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_total_tokens)

        self.lbl_active_header = QLabel("CURRENT: None")
        self.lbl_active_header.setStyleSheet("color: #ff9100; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        top_cards_layout.addWidget(self.lbl_active_header, stretch=1)

        layout.addWidget(top_cards_frame)

        # 2. Main Tabbed Navigation
        self.tabs = QTabWidget(self)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1a233a;
                background-color: #070a12;
            }
            QTabBar::tab {
                background-color: #0c101c;
                color: #8da2c0;
                padding: 7px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background-color: #121927;
                color: #00e5ff;
                border-bottom: 2px solid #00e5ff;
            }
            QTabBar::tab:hover {
                background-color: #162032;
                color: #ffffff;
            }
        """)

        # Tab 1: CURRENT ACTIVE SOURCE & LIVE QUEUE
        tab_live = QWidget()
        tab_live_layout = QVBoxLayout(tab_live)
        tab_live_layout.setContentsMargins(8, 8, 8, 8)
        tab_live_layout.setSpacing(8)

        # Current Learning Hero Card
        self.hero_card = QFrame()
        self.hero_card.setStyleSheet("""
            QFrame {
                background-color: #0d1322;
                border: 1px solid #00e5ff;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        hero_layout = QVBoxLayout(self.hero_card)
        hero_layout.setContentsMargins(8, 6, 8, 6)
        hero_layout.setSpacing(4)

        hero_top_bar = QHBoxLayout()
        self.lbl_current_badge = QLabel("CURRENTLY TRAINING ON")
        self.lbl_current_badge.setStyleSheet("background-color: #00e5ff; color: #070a12; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 3px;")
        hero_top_bar.addWidget(self.lbl_current_badge)

        self.lbl_current_type = QLabel("SOURCE: Wikipedia Encyclopedia")
        self.lbl_current_type.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        hero_top_bar.addWidget(self.lbl_current_type)
        hero_top_bar.addStretch()

        self.lbl_current_meta = QLabel("TOKENS: 0 | CHARS: 0")
        self.lbl_current_meta.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 11px;")
        hero_top_bar.addWidget(self.lbl_current_meta)
        hero_layout.addLayout(hero_top_bar)

        self.lbl_current_title = QLabel("No active article loaded yet")
        self.lbl_current_title.setStyleSheet("color: #ffffff; font-size: 14px; font-weight: bold; margin-top: 2px;")
        hero_layout.addWidget(self.lbl_current_title)

        self.lbl_current_url = QLabel("https://en.wikipedia.org")
        self.lbl_current_url.setStyleSheet("color: #8cb4f0; font-family: 'Consolas', monospace; font-size: 10px;")
        hero_layout.addWidget(self.lbl_current_url)

        self.txt_current_snippet = QTextEdit()
        self.txt_current_snippet.setReadOnly(True)
        self.txt_current_snippet.setMaximumHeight(85)
        self.txt_current_snippet.setStyleSheet("""
            QTextEdit {
                background-color: #080c14;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_current_snippet.setPlaceholderText("Current document text will stream here as the model steps through it...")
        hero_layout.addWidget(self.txt_current_snippet)

        tab_live_layout.addWidget(self.hero_card)

        # Queue Section
        lbl_queue_header = QLabel("UPCOMING QUEUE (Next Websites in Active 20-Batch)")
        lbl_queue_header.setStyleSheet("color: #ffb300; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold; margin-top: 4px;")
        tab_live_layout.addWidget(lbl_queue_header)

        self.table_queue = QTableWidget(0, 5)
        self.table_queue.setHorizontalHeaderLabels(["QUEUE POS", "SOURCE TYPE", "TITLE", "TOKENS", "ORIGIN URL"])
        self.table_queue.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_queue.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_queue.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table_queue.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tab_live_layout.addWidget(self.table_queue)

        self.tabs.addTab(tab_live, "LIVE INGESTION (CURRENT & QUEUE)")

        # Tab 2: FINISHED & MASTERED SOURCES
        tab_finished = QWidget()
        tab_fin_layout = QVBoxLayout(tab_finished)
        tab_fin_layout.setContentsMargins(8, 8, 8, 8)
        tab_fin_layout.setSpacing(6)

        fin_header_layout = QHBoxLayout()
        lbl_fin_title = QLabel("FINISHED & MASTERED KNOWLEDGE CORPUS (Single-Pass Completed)")
        lbl_fin_title.setStyleSheet("color: #00e676; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        fin_header_layout.addWidget(lbl_fin_title)

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("Filter finished websites by title, URL, or type...")
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
        self.search_box.textChanged.connect(self._filter_finished_table)
        fin_header_layout.addWidget(self.search_box)
        tab_fin_layout.addLayout(fin_header_layout)

        fin_splitter = QSplitter(Qt.Orientation.Vertical)
        self.table_finished = QTableWidget(0, 6)
        self.table_finished.setHorizontalHeaderLabels(["COMPLETED #", "SOURCE TYPE", "TITLE", "TOKENS", "CHARS", "ORIGIN URL"])
        self.table_finished.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_finished.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_finished.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_finished.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        self.table_finished.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        self.table_finished.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.table_finished.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table_finished.itemSelectionChanged.connect(self._on_finished_selected)
        fin_splitter.addWidget(self.table_finished)

        # Finished Snippet Inspector
        fin_detail_box = QWidget()
        fin_detail_layout = QVBoxLayout(fin_detail_box)
        fin_detail_layout.setContentsMargins(0, 4, 0, 0)
        lbl_fin_snip = QLabel("MASTERED DOCUMENT SNIPPET & ABSTRACT")
        lbl_fin_snip.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 10px; font-weight: bold;")
        fin_detail_layout.addWidget(lbl_fin_snip)
        self.txt_finished_snippet = QTextEdit()
        self.txt_finished_snippet.setReadOnly(True)
        self.txt_finished_snippet.setMaximumHeight(110)
        self.txt_finished_snippet.setStyleSheet("""
            QTextEdit {
                background-color: #0b0f19;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_finished_snippet.setPlaceholderText("Select any completed source above to read its text abstract...")
        fin_detail_layout.addWidget(self.txt_finished_snippet)
        fin_splitter.addWidget(fin_detail_box)
        fin_splitter.setSizes([320, 120])
        tab_fin_layout.addWidget(fin_splitter)

        self.tabs.addTab(tab_finished, "FINISHED / MASTERED CORPUS")

        # Tab 3: EPISODIC & SEMANTIC MEMORIES
        tab_mem = QWidget()
        tab_mem_layout = QVBoxLayout(tab_mem)
        tab_mem_layout.setContentsMargins(8, 8, 8, 8)
        tab_mem_layout.setSpacing(6)

        lbl_ep = QLabel("EPISODIC DISCOVERIES & SURPRISE MOMENTS")
        lbl_ep.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px; font-weight: bold;")
        tab_mem_layout.addWidget(lbl_ep)

        self.table_episodic = QTableWidget(0, 4)
        self.table_episodic.setHorizontalHeaderLabels(["STEP", "TYPE", "IMPORTANCE", "DESCRIPTION"])
        self.table_episodic.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_episodic.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_episodic.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        tab_mem_layout.addWidget(self.table_episodic)

        self.tabs.addTab(tab_mem, "EPISODIC & SEMANTIC MEMORY")

        layout.addWidget(self.tabs)

        # Registries
        self._sources_registry: Dict[str, Dict[str, Any]] = {}
        self._finished_urls: List[str] = []
        self._queue_urls: List[str] = []
        self._current_url: str = ""
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

        is_new = url not in self._sources_registry
        self._sources_registry[url] = {
            "title": title,
            "url": url,
            "source_type": source_type,
            "token_count": tokens,
            "character_count": chars,
            "text_snippet": snippet,
            "timestamp": timestamp,
            "status": status,
        }

        if is_new:
            self._total_tokens_count += tokens
            self._total_chars_count += chars
            if url not in self._queue_urls and url not in self._finished_urls and url != self._current_url:
                self._queue_urls.append(url)

        self._refresh_tables()

    def set_active_learning_source(self, title: str, url: Optional[str] = None) -> None:
        """Update the currently active learning source, moving the previous active to finished."""
        # If we had a previous active URL and it changed, mark previous as finished
        if self._current_url and url and self._current_url != url:
            if self._current_url not in self._finished_urls:
                self._finished_urls.append(self._current_url)
                if self._current_url in self._sources_registry:
                    self._sources_registry[self._current_url]["status"] = "FINISHED"

        if url:
            self._current_url = url
            if url in self._queue_urls:
                self._queue_urls.remove(url)

        # Lookup data for the current active source
        current_data = self._sources_registry.get(self._current_url, {})
        c_title = current_data.get("title", title)
        c_url = current_data.get("url", url or "")
        c_type = current_data.get("source_type", "Web Stream")
        c_tokens = current_data.get("token_count", 0)
        c_chars = current_data.get("character_count", 0)
        c_snippet = current_data.get("text_snippet", "")

        self.lbl_active_header.setText(f"CURRENT: {c_title[:32]}")
        self.lbl_current_title.setText(c_title)
        self.lbl_current_url.setText(c_url)
        self.lbl_current_type.setText(f"SOURCE: {c_type}")
        self.lbl_current_meta.setText(f"TOKENS: {c_tokens:,} | CHARS: {c_chars:,}")
        if c_snippet:
            self.txt_current_snippet.setText(c_snippet)

        self._refresh_tables()

    def _refresh_tables(self) -> None:
        """Re-render Queue and Finished tables with updated counts."""
        # 1. Update Queue Table
        self.table_queue.setRowCount(0)
        for pos, q_url in enumerate(self._queue_urls, 1):
            data = self._sources_registry.get(q_url, {})
            row = self.table_queue.rowCount()
            self.table_queue.insertRow(row)

            item_pos = QTableWidgetItem(f"#{pos}")
            item_pos.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_pos.setForeground(QColor(255, 179, 0))

            item_type = QTableWidgetItem(data.get("source_type", "Web"))
            item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_title = QTableWidgetItem(data.get("title", "Queued Source"))
            item_title.setToolTip(data.get("title", ""))

            item_tokens = QTableWidgetItem(f"{data.get('token_count', 0):,}")
            item_tokens.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            item_url = QTableWidgetItem(q_url)
            item_url.setToolTip(q_url)
            item_url.setForeground(QColor(140, 180, 240))

            self.table_queue.setItem(row, 0, item_pos)
            self.table_queue.setItem(row, 1, item_type)
            self.table_queue.setItem(row, 2, item_title)
            self.table_queue.setItem(row, 3, item_tokens)
            self.table_queue.setItem(row, 4, item_url)

        # 2. Update Finished Table
        self.table_finished.setRowCount(0)
        for num, f_url in enumerate(self._finished_urls, 1):
            data = self._sources_registry.get(f_url, {})
            row = self.table_finished.rowCount()
            self.table_finished.insertRow(row)

            item_num = QTableWidgetItem(f"#{num}")
            item_num.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            item_num.setForeground(QColor(0, 230, 118))

            item_type = QTableWidgetItem(data.get("source_type", "Web"))
            item_type.setTextAlignment(Qt.AlignmentFlag.AlignCenter)

            item_title = QTableWidgetItem(data.get("title", "Finished Source"))
            item_title.setToolTip(data.get("title", ""))

            item_tokens = QTableWidgetItem(f"{data.get('token_count', 0):,}")
            item_tokens.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            item_chars = QTableWidgetItem(f"{data.get('character_count', 0):,}")
            item_chars.setTextAlignment(Qt.AlignmentFlag.AlignRight)

            item_url = QTableWidgetItem(f_url)
            item_url.setToolTip(f_url)
            item_url.setForeground(QColor(140, 180, 240))

            self.table_finished.setItem(row, 0, item_num)
            self.table_finished.setItem(row, 1, item_type)
            self.table_finished.setItem(row, 2, item_title)
            self.table_finished.setItem(row, 3, item_tokens)
            self.table_finished.setItem(row, 4, item_chars)
            self.table_finished.setItem(row, 5, item_url)

        # 3. Update Top Header Cards
        total_count = len(self._sources_registry)
        fin_count = len(self._finished_urls)
        queue_count = len(self._queue_urls)

        self.lbl_total_sources.setText(f"TOTAL INGESTED: {total_count:,}")
        self.lbl_total_finished.setText(f"FINISHED / MASTERED: {fin_count:,}")
        self.lbl_total_queue.setText(f"IN QUEUE: {queue_count:,}")
        self.lbl_total_tokens.setText(f"TOKENS: {self._total_tokens_count:,}")

    def _on_finished_selected(self) -> None:
        selected_rows = self.table_finished.selectionModel().selectedRows()
        if not selected_rows:
            return
        row = selected_rows[0].row()
        item_url = self.table_finished.item(row, 5)
        if item_url:
            url = item_url.text()
            data = self._sources_registry.get(url, {})
            snippet = data.get("text_snippet", "")
            title = data.get("title", "")
            tokens = data.get("token_count", 0)
            chars = data.get("character_count", 0)
            src_type = data.get("source_type", "")

            full_info = (
                f"SOURCE: {title}\n"
                f"ORIGIN URL: {url}\n"
                f"TYPE: {src_type} | STATUS: MASTERED (SINGLE PASS) | TOKENS: {tokens:,} | CHARS: {chars:,}\n"
                f"{'=' * 70}\n\n"
                f"{snippet}"
            )
            self.txt_finished_snippet.setText(full_info)

    def _filter_finished_table(self, query: str) -> None:
        query = query.lower().strip()
        for row in range(self.table_finished.rowCount()):
            match = False
            if not query:
                match = True
            else:
                for col in (1, 2, 5):
                    item = self.table_finished.item(row, col)
                    if item and query in item.text().lower():
                        match = True
                        break
            self.table_finished.setRowHidden(row, not match)

    def populate_episodic(self, episodes: list) -> None:
        self.table_episodic.setRowCount(0)
        for ep in episodes:
            row = self.table_episodic.rowCount()
            self.table_episodic.insertRow(row)

            self.table_episodic.setItem(row, 0, QTableWidgetItem(str(ep.get("step", 0))))
            self.table_episodic.setItem(row, 1, QTableWidgetItem(ep.get("event_type", "")))
            self.table_episodic.setItem(row, 2, QTableWidgetItem(f"{ep.get('importance', 1.0):.2f}"))
            self.table_episodic.setItem(row, 3, QTableWidgetItem(ep.get("description", "")))
