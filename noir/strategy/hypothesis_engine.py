"""Hypothesis generation and heuristic reasoning engine."""

import json
from typing import Any, Dict, List, Optional

from noir.core.logging import get_logger
from noir.strategy.llm_provider import LLMProvider

logger = get_logger("strategy.hypothesis")


class HypothesisEngine:
    """Analyzes real metrics, affective vectors, and gradient statistics to form scientific hypotheses."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def generate_hypothesis(
        self,
        experiment_id: str,
        step: int,
        metrics: Dict[str, float],
        emotion_state: Dict[str, float],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Formulate a strategic diagnosis and proposed actions."""
        system_prompt = (
            "You are the Chief AI Strategist for Project NOIR, an advanced machine learning research platform. "
            "Analyze the numerical training metrics and mathematical affective state vector. "
            "Output your findings STRICTLY as a valid JSON object with keys: "
            "'hypothesis' (str), 'confidence' (float), 'proposed_actions' (list of dicts with 'parameter', 'action', 'value'/'factor'), 'explanation' (str)."
        )

        user_prompt = json.dumps({
            "experiment_id": experiment_id,
            "step": step,
            "metrics": metrics,
            "affective_state": emotion_state,
            "active_config": config,
        }, indent=2)

        try:
            raw_response = self.provider.generate_completion(system_prompt, user_prompt)
            # Clean JSON if wrapped in markdown backticks
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            parsed = json.loads(cleaned)
            parsed["timestamp_step"] = step
            return parsed
        except Exception as e:
            logger.warning("Error parsing hypothesis JSON: %s. Using structured fallback.", e)
            return {
                "hypothesis": f"Training step {step}: Loss is {metrics.get('loss', metrics.get('train_loss', 0.0)):.4f}, Frustration: {emotion_state.get('frustration', 0.0):.2f}.",
                "confidence": 0.75,
                "proposed_actions": [],
                "explanation": "Automatic fallback analysis based on current telemetry.",
                "timestamp_step": step,
            }
