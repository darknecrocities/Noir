"""Strategist intelligence and MCP integration view."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
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


class StrategistView(QWidget):
    """View displaying LLM Strategist hypotheses, reasoning logs, and MCP tooling status."""

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Column: Strategic Hypotheses History
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        lbl_hyp = QLabel("AI STRATEGIST HYPOTHESES STREAM")
        lbl_hyp.setStyleSheet("color: #ffd700; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_hyp)

        self.table_hypotheses = QTableWidget(0, 3)
        self.table_hypotheses.setHorizontalHeaderLabels(["STEP", "CONFIDENCE", "HYPOTHESIS"])
        self.table_hypotheses.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_hypotheses.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_hypotheses.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table_hypotheses.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.table_hypotheses)

        splitter.addWidget(left_widget)

        # Right Column: Detailed Diagnosis & MCP Tools
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Detailed Reasoning Card
        grp_reasoning = QGroupBox("STRATEGIC REASONING & ACTIONS")
        grp_reasoning.setStyleSheet("""
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                color: #00e5ff;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        reasoning_layout = QVBoxLayout(grp_reasoning)
        self.txt_reasoning = QTextEdit()
        self.txt_reasoning.setReadOnly(True)
        self.txt_reasoning.setStyleSheet("""
            QTextEdit {
                background-color: #0b0f19;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_reasoning.setPlaceholderText("Select a hypothesis on the left to inspect strategic actions...")
        reasoning_layout.addWidget(self.txt_reasoning)
        right_layout.addWidget(grp_reasoning)

        # MCP Status Box
        grp_mcp = QGroupBox("MODEL CONTEXT PROTOCOL (MCP) INTERFACE")
        grp_mcp.setStyleSheet("""
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                color: #64ffda;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        mcp_layout = QVBoxLayout(grp_mcp)
        self.lbl_mcp_status = QLabel("SERVER: Running at http://127.0.0.1:8765 | STATUS: ACTIVE")
        self.lbl_mcp_status.setStyleSheet("color: #64ffda; font-family: 'Consolas', monospace; font-size: 11px;")
        mcp_layout.addWidget(self.lbl_mcp_status)

        mcp_tools_desc = (
            "Available MCP Tools:\n"
            "  • get_training_status    • get_latest_metrics   • inspect_model\n"
            "  • inspect_layer          • get_emotion_state    • get_memory\n"
            "  • create_checkpoint      • load_checkpoint      • branch_experiment\n"
            "  • pause_training         • resume_training      • stop_training"
        )
        lbl_tools = QLabel(mcp_tools_desc)
        lbl_tools.setStyleSheet("color: #8da2c0; font-family: 'Consolas', monospace; font-size: 10px;")
        mcp_layout.addWidget(lbl_tools)

        right_layout.addWidget(grp_mcp)

        splitter.addWidget(right_widget)
        splitter.setSizes([450, 450])

        layout.addWidget(splitter)
        self.table_hypotheses.itemClicked.connect(self._on_hyp_click)
        self._hypotheses_data = []

    def add_hypothesis(self, step: int, conf: float, hyp_text: str, explanation: str, actions: list) -> None:
        row = 0
        self.table_hypotheses.insertRow(row)
        self.table_hypotheses.setItem(row, 0, QTableWidgetItem(str(step)))
        self.table_hypotheses.setItem(row, 1, QTableWidgetItem(f"{conf:.2f}"))
        self.table_hypotheses.setItem(row, 2, QTableWidgetItem(hyp_text))

        self._hypotheses_data.insert(0, {
            "step": step,
            "confidence": conf,
            "hypothesis": hyp_text,
            "explanation": explanation,
            "actions": actions,
        })

    def _on_hyp_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        if 0 <= row < len(self._hypotheses_data):
            h = self._hypotheses_data[row]
            detail = (
                f"HYPOTHESIS (Step {h['step']}):\n{h['hypothesis']}\n\n"
                f"EXPLANATION:\n{h['explanation']}\n\n"
                f"PROPOSED ACTIONS:\n"
            )
            for a in h.get("actions", []):
                detail += f"  • Param: {a.get('parameter')}, Action: {a.get('action')}, Value/Factor: {a.get('value') or a.get('factor')}\n"

            self.txt_reasoning.setText(detail)
