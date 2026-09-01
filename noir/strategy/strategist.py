"""Strategic planner, evaluator, and asynchronous strategist coordinator."""

import threading
import time
from typing import Any, Dict, List, Optional

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import EventBus, get_event_bus
from noir.events.event_types import EventType
from noir.storage.database import DatabaseManager, StrategyModel
from noir.strategy.hypothesis_engine import HypothesisEngine
from noir.strategy.llm_provider import LLMProvider, create_llm_provider

logger = get_logger("strategy.strategist")


class Planner:
    """Translates hypotheses into concrete experiment branch configurations."""

    @staticmethod
    def create_branch_plan(hypothesis_data: Dict[str, Any], current_config: Dict[str, Any]) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for action in hypothesis_data.get("proposed_actions", []):
            param = action.get("parameter")
            if not param:
                continue

            act_type = action.get("action")
            factor = action.get("factor")
            val = action.get("value")

            keys = param.split(".")
            # Handle deep config update
            if val is not None:
                Planner._set_nested(overrides, keys, val)
            elif factor is not None and len(keys) == 2:
                sec, k = keys
                current_val = current_config.get(sec, {}).get(k, 0.001)
                if isinstance(current_val, (int, float)):
                    new_val = current_val * factor
                    Planner._set_nested(overrides, keys, new_val)

        return overrides

    @staticmethod
    def _set_nested(d: Dict[str, Any], keys: List[str], val: Any) -> None:
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = val


class Evaluator:
    """Evaluates hypothesis success against metric trajectories."""

    @staticmethod
    def evaluate_hypothesis(hypothesis: Dict[str, Any], before_metric: float, after_metric: float, higher_is_better: bool = False) -> bool:
        if higher_is_better:
            return after_metric > before_metric
        return after_metric < before_metric


class Strategist:
    """Asynchronous strategist running in a non-blocking slow loop."""

    def __init__(
        self,
        experiment_id: str,
        provider: Optional[LLMProvider] = None,
        db_manager: Optional[DatabaseManager] = None,
        event_bus: Optional[EventBus] = None,
        analysis_interval_steps: int = 250,
    ):
        self.experiment_id = experiment_id
        self.provider = provider or create_llm_provider("mock")
        self.hypothesis_engine = HypothesisEngine(self.provider)
        self.db = db_manager
        self.event_bus = event_bus or get_event_bus()
        self.analysis_interval_steps = analysis_interval_steps

        self.last_analysis_step = 0
        self.latest_hypothesis: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def analyze_async(
        self,
        step: int,
        metrics: Dict[str, float],
        emotion_state: Dict[str, float],
        config: Dict[str, Any],
    ) -> None:
        """Trigger strategic analysis in a separate background thread without blocking the training loop."""
        if step - self.last_analysis_step < self.analysis_interval_steps:
            return

        self.last_analysis_step = step
        t = threading.Thread(
            target=self._run_analysis,
            args=(step, metrics, emotion_state, config),
            daemon=True,
            name=f"StrategistAnalysis-{step}",
        )
        t.start()

    def _run_analysis(
        self,
        step: int,
        metrics: Dict[str, float],
        emotion_state: Dict[str, float],
        config: Dict[str, Any],
    ) -> None:
        try:
            hypothesis = self.hypothesis_engine.generate_hypothesis(
                experiment_id=self.experiment_id,
                step=step,
                metrics=metrics,
                emotion_state=emotion_state,
                config=config,
            )
            with self._lock:
                self.latest_hypothesis = hypothesis

            # Emit event
            self.event_bus.publish(
                NoirEvent.create(
                    EventType.HYPOTHESIS_GENERATED,
                    experiment_id=self.experiment_id,
                    training_step=step,
                    hypothesis=hypothesis.get("hypothesis", ""),
                    confidence=hypothesis.get("confidence", 0.0),
                    proposed_actions=hypothesis.get("proposed_actions", []),
                    explanation=hypothesis.get("explanation", ""),
                )
            )

            # Persist to database
            if self.db:
                import uuid
                with self.db.get_session() as session:
                    model = StrategyModel(
                        id=f"strat_{uuid.uuid4().hex[:8]}",
                        experiment_id=self.experiment_id,
                        step=step,
                        hypothesis=hypothesis.get("hypothesis", ""),
                        proposal=json.dumps(hypothesis.get("proposed_actions", [])),
                        status="PROPOSED",
                        metrics_summary=json.dumps(metrics),
                        timestamp=time.time(),
                    )
                    session.add(model)
                    session.commit()

            logger.info("Generated hypothesis at step %d: %s", step, hypothesis.get("hypothesis", ""))
        except Exception as e:
            logger.error("Failed to run strategist analysis: %s", e)
