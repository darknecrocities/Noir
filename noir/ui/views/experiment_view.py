"""Experiment management, tree branching, and comparison view."""

import json
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)


class ExperimentView(QWidget):
    """View for managing experiments, creating branches, and comparing runs."""

    branch_requested = Signal(str, str, dict)  # (parent_id, new_name, overrides)

    def __init__(self, parent: QWidget = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left Column: Experiment Tree List
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        lbl_tree = QLabel("EXPERIMENT ARCHIVE & BRANCHES")
        lbl_tree.setStyleSheet("color: #00e5ff; font-family: 'Consolas', monospace; font-size: 12px; font-weight: bold;")
        left_layout.addWidget(lbl_tree)

        self.table_experiments = QTableWidget(0, 5)
        self.table_experiments.setHorizontalHeaderLabels(["ID", "NAME", "STATUS", "STEPS", "BEST METRIC"])
        self.table_experiments.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table_experiments.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_experiments.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        left_layout.addWidget(self.table_experiments)

        splitter.addWidget(left_widget)

        # Right Column: Branching Controls & Comparison
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # 1. Branching Box
        grp_branch = QGroupBox("BRANCH FROM EXPERIMENT")
        grp_branch.setStyleSheet("""
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
        branch_layout = QVBoxLayout(grp_branch)
        branch_layout.setSpacing(8)

        branch_layout.addWidget(QLabel("Parent Experiment ID:"))
        self.txt_parent_id = QLineEdit()
        self.txt_parent_id.setPlaceholderText("Select an experiment from the left table or enter ID")
        branch_layout.addWidget(self.txt_parent_id)

        branch_layout.addWidget(QLabel("New Branch Name:"))
        self.txt_branch_name = QLineEdit("Exploration Branch (LR Tuning)")
        branch_layout.addWidget(self.txt_branch_name)

        branch_layout.addWidget(QLabel("Learning Rate Override:"))
        self.spin_branch_lr = QDoubleSpinBox()
        self.spin_branch_lr.setRange(0.00001, 0.1)
        self.spin_branch_lr.setDecimals(5)
        self.spin_branch_lr.setValue(0.0001)
        branch_layout.addWidget(self.spin_branch_lr)

        self.btn_branch = QPushButton("Create Branch & Initialize")
        self.btn_branch.setStyleSheet("""
            QPushButton {
                background-color: #00875a;
                color: #ffffff;
                border: 1px solid #00b875;
                border-radius: 4px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00a86b;
            }
        """)
        self.btn_branch.clicked.connect(self._on_branch)
        branch_layout.addWidget(self.btn_branch)

        right_layout.addWidget(grp_branch)

        # 2. Experiment Comparison Box
        grp_compare = QGroupBox("EXPERIMENT COMPARISON & DIFF")
        grp_compare.setStyleSheet("""
            QGroupBox {
                border: 1px solid #1e293b;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 12px;
                color: #ffd700;
                font-family: 'Consolas', monospace;
                font-size: 11px;
                font-weight: bold;
            }
        """)
        comp_layout = QVBoxLayout(grp_compare)
        comp_layout.setSpacing(6)

        self.txt_comparison = QTextEdit()
        self.txt_comparison.setReadOnly(True)
        self.txt_comparison.setStyleSheet("""
            QTextEdit {
                background-color: #0b0f19;
                border: 1px solid #162032;
                color: #cbd5e1;
                font-family: 'Consolas', monospace;
                font-size: 11px;
            }
        """)
        self.txt_comparison.setPlaceholderText("Select experiments to view hyperparameter diffs...")
        comp_layout.addWidget(self.txt_comparison)

        right_layout.addWidget(grp_compare)

        splitter.addWidget(right_widget)
        splitter.setSizes([500, 400])

        layout.addWidget(splitter)
        self.table_experiments.itemClicked.connect(self._on_table_click)

    def populate_experiments(self, experiments: list) -> None:
        """Refresh experiment records in table."""
        self.table_experiments.setRowCount(0)
        for e in experiments:
            row = self.table_experiments.rowCount()
            self.table_experiments.insertRow(row)

            self.table_experiments.setItem(row, 0, QTableWidgetItem(e.get("id", "")))
            self.table_experiments.setItem(row, 1, QTableWidgetItem(e.get("name", "")))
            self.table_experiments.setItem(row, 2, QTableWidgetItem(e.get("status", "")))
            self.table_experiments.setItem(row, 3, QTableWidgetItem(str(e.get("total_steps", 0))))

            best = e.get("best_metric")
            best_str = f"{best:.4f}" if best is not None else "--"
            self.table_experiments.setItem(row, 4, QTableWidgetItem(best_str))

    def _on_table_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        exp_id_item = self.table_experiments.item(row, 0)
        if exp_id_item:
            self.txt_parent_id.setText(exp_id_item.text())

    def _on_branch(self) -> None:
        parent_id = self.txt_parent_id.text().strip()
        new_name = self.txt_branch_name.text().strip()
        if parent_id and new_name:
            overrides = {
                "training": {
                    "learning_rate": self.spin_branch_lr.value(),
                }
            }
            self.branch_requested.emit(parent_id, new_name, overrides)
