"""Master Desktop Window for Project NOIR."""

import sys
from typing import Optional
from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from noir.core.engine import NoirEngine
from noir.events.event import NoirEvent
from noir.events.event_types import EventType
from noir.ui.views.dashboard import DashboardView
from noir.ui.views.experiment_view import ExperimentView
from noir.ui.views.memory_view import MemoryView
from noir.ui.views.strategist_view import StrategistView
from noir.ui.widgets.recovery_dialog import RecoveryDialog
from noir.ui.widgets.system_monitor import SystemMonitorBar
from noir.visualization.training_monitor import TrainingMonitor


class UIEventBridge(QObject):
    """Thread-safe bridge receiving events from EventBus and forwarding to Qt signals."""

    event_received = Signal(object)

    def dispatch(self, event: NoirEvent) -> None:
        self.event_received.emit(event)


class NoirMainWindow(QMainWindow):
    """Master native desktop interface for Project NOIR research laboratory."""

    def __init__(self, engine: NoirEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.setWindowTitle("PROJECT NOIR — Real-Time AI & Machine Learning Research Environment")
        self.resize(1360, 860)
        self.setMinimumSize(1024, 680)

        self._apply_theme()

        # Event Bridge
        self.bridge = UIEventBridge()
        self.bridge.event_received.connect(self._on_event_received)
        self.engine.event_bus.subscribe(None, self.bridge.dispatch)

        # Central Widget & Tab Navigation
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # 1. Dashboard View
        self.dashboard_view = DashboardView(self)
        self.tabs.addTab(self.dashboard_view, "DASHBOARD")

        # 2. Experiments & Branching View
        self.experiment_view = ExperimentView(self)
        self.tabs.addTab(self.experiment_view, "EXPERIMENTS & BRANCHING")

        # 3. Strategist & MCP View
        self.strategist_view = StrategistView(self)
        self.tabs.addTab(self.strategist_view, "AI STRATEGIST & MCP")

        # 4. Memory View
        self.memory_view = MemoryView(self)
        self.tabs.addTab(self.memory_view, "MEMORY & KNOWLEDGE")

        main_layout.addWidget(self.tabs)

        # System Monitor Status Bar
        self.system_bar = SystemMonitorBar(self)
        main_layout.addWidget(self.system_bar)

        # Attach Training Monitor for 3D View
        self.training_monitor = TrainingMonitor(
            visualizer=self.dashboard_view.visualizer_3d,
            engine=self.engine,
            event_bus=self.engine.event_bus,
        )

        # Connect UI Controls
        self._connect_signals()

        # Refresh initial lists
        self._refresh_experiment_list()

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow {
                background-color: #06080e;
            }
            QTabWidget::pane {
                border-top: 1px solid #1a233a;
                background-color: #06080e;
            }
            QTabBar::tab {
                background-color: #0d121f;
                color: #8da2c0;
                padding: 10px 20px;
                border: 1px solid #1a233a;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                font-family: 'Segoe UI', sans-serif;
                font-size: 11px;
                font-weight: bold;
                letter-spacing: 0.5px;
            }
            QTabBar::tab:selected {
                background-color: #121927;
                color: #00e5ff;
                border-color: #00e5ff;
            }
            QTabBar::tab:hover {
                background-color: #162032;
                color: #ffffff;
            }
        """)

    def _connect_signals(self) -> None:
        controls = self.dashboard_view.controls
        controls.start_clicked.connect(self._on_start_training)
        controls.pause_clicked.connect(self.engine.pause_training)
        controls.resume_clicked.connect(self.engine.resume_training)
        controls.stop_clicked.connect(self.engine.stop_training)
        controls.checkpoint_clicked.connect(self._on_manual_checkpoint)

        self.experiment_view.branch_requested.connect(self._on_branch_experiment)

    def check_startup_recovery(self) -> None:
        """Inspect storage on startup and prompt user if recovery is available."""
        recovery_info = self.engine.recovery_manager.check_for_recovery()
        if recovery_info:
            dlg = RecoveryDialog(recovery_info, self)
            if dlg.exec():
                action = dlg.selected_action
                self.engine.recover_from_previous_session(action=action)
                self.system_bar.update_status("RECOVERED")

    def _on_start_training(self, selection: str, lr: float) -> None:
        self.dashboard_view.metrics_panel.clear()
        if selection == "llm:open_web" or selection.startswith("llm"):
            self.engine.start_open_web_llm_experiment(
                name="Open Web Live Internet LLM",
                learning_rate=lr,
            )
        elif selection.startswith("supervised"):
            parts = selection.split(":")
            dataset_name = parts[1] if len(parts) > 1 else "digits"
            self.engine.start_supervised_experiment(
                dataset_name=dataset_name,
                learning_rate=lr,
            )
        else:
            self.engine.start_rl_experiment(
                name="PPO GridWorld Exploration",
                learning_rate=lr,
            )
        self.system_bar.update_status("RUNNING")
        self._refresh_experiment_list()

    def _on_manual_checkpoint(self) -> None:
        try:
            path = self.engine.save_checkpoint(tag="manual")
            QMessageBox.information(self, "Checkpoint Saved", f"Saved atomic checkpoint:\n{path.name}")
        except Exception as e:
            QMessageBox.warning(self, "Checkpoint Error", str(e))

    def _on_branch_experiment(self, parent_id: str, new_name: str, overrides: dict) -> None:
        try:
            new_id = self.engine.experiment_repo.branch_experiment(
                parent_id=parent_id,
                new_name=new_name,
                config_overrides=overrides,
            )
            QMessageBox.information(self, "Branch Created", f"Successfully created branch:\n{new_name} ({new_id})")
            self._refresh_experiment_list()
        except Exception as e:
            QMessageBox.warning(self, "Branch Error", str(e))

    def _refresh_experiment_list(self) -> None:
        exps = self.engine.experiment_repo.list_experiments()
        self.experiment_view.populate_experiments(exps)
        self.memory_view.populate_episodic(self.engine.memory_manager.episodic.get_salient_experiences())

    @Slot(object)
    def _on_event_received(self, event: NoirEvent) -> None:
        """Handle incoming typed events on the main Qt GUI thread with responsive throttling."""
        # 1. Update Timeline
        self.dashboard_view.event_timeline.add_event(event)

        # 2. Update Metrics & Emotion Panels
        if event.event_type == EventType.WEIGHTS_UPDATED:
            loss = event.payload.get("loss")
            metrics = event.payload.get("metrics", {})
            if loss is not None:
                self.dashboard_view.metrics_panel.add_loss_point(event.training_step, float(loss))
            if "train_acc" in metrics:
                self.dashboard_view.metrics_panel.add_reward_point(event.training_step, metrics["train_acc"])
            elif "perplexity" in metrics:
                # For LLM training: display log(perplexity) as metric curve
                self.dashboard_view.metrics_panel.add_reward_point(event.training_step, min(100.0, float(metrics["perplexity"])))

            if "article" in metrics:
                self.dashboard_view.set_hypothesis_text(f"Reading Open Web Source: {metrics['article']}")

            self.system_bar.update_training_metrics(
                step=event.training_step,
                epoch=event.epoch,
                loss=loss,
            )

        elif event.event_type == EventType.REWARD_RECEIVED:
            reward = event.payload.get("reward")
            if reward is not None:
                self.system_bar.update_training_metrics(
                    step=event.training_step,
                    epoch=event.epoch,
                    reward=reward,
                )

        elif event.event_type == EventType.EPISODE_COMPLETED:
            ep_reward = event.payload.get("episode_reward")
            if ep_reward is not None:
                self.dashboard_view.metrics_panel.add_reward_point(event.epoch, float(ep_reward))

        elif event.event_type == EventType.EMOTION_UPDATED:
            state = event.payload.get("emotion_state", {})
            self.dashboard_view.emotion_panel.update_state(state)

        elif event.event_type == EventType.SYSTEM_METRICS_UPDATED:
            self.system_bar.update_hardware(
                cpu=event.payload.get("cpu_percent", 0),
                ram=event.payload.get("ram_percent", 0),
                gpu=event.payload.get("gpu_percent", 0),
                vram=event.payload.get("gpu_memory_mb", 0),
                cuda=event.payload.get("cuda_available", False),
            )

        elif event.event_type == EventType.HYPOTHESIS_GENERATED:
            hyp = event.payload.get("hypothesis", "")
            conf = event.payload.get("confidence", 0.0)
            exp = event.payload.get("explanation", "")
            actions = event.payload.get("proposed_actions", [])
            self.dashboard_view.set_hypothesis_text(hyp)
            self.strategist_view.add_hypothesis(event.training_step, conf, hyp, exp, actions)

        elif event.event_type == EventType.TRAINING_PAUSED:
            self.system_bar.update_status("PAUSED")
        elif event.event_type == EventType.TRAINING_RESUMED:
            self.system_bar.update_status("RUNNING")
        elif event.event_type == EventType.TRAINING_STOPPED:
            self.system_bar.update_status("STOPPED")

    def closeEvent(self, event) -> None:
        """Graceful shutdown on window close."""
        self.engine.shutdown()
        event.accept()
