"""LLM Provider abstraction supporting local models, OpenAI endpoints, and heuristic fallback."""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from noir.core.logging import get_logger

logger = get_logger("strategy.llm_provider")


class LLMProvider(ABC):
    """Abstract interface for LLM completion services."""

    @abstractmethod
    def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text completion from LLM."""
        pass


class MockLLMProvider(LLMProvider):
    """Heuristic rule-based fallback provider when no external LLM API is configured or reachable."""

    def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        # Heuristic analysis based on prompt keywords
        if "loss" in user_prompt.lower() and "high" in user_prompt.lower():
            return json.dumps({
                "hypothesis": "Gradient variance is high, indicating the learning rate may be too large for current batch dynamics.",
                "confidence": 0.82,
                "proposed_actions": [
                    {"parameter": "training.learning_rate", "action": "decrease", "factor": 0.5},
                    {"parameter": "training.gradient_clip_val", "action": "set", "value": 0.5},
                ],
                "explanation": "Reducing learning rate will stabilize parameter updates and prevent loss spikes."
            })
        elif "frustration" in user_prompt.lower() or "stagnat" in user_prompt.lower():
            return json.dumps({
                "hypothesis": "Policy is trapped in a local reward plateau. Exploration pressure is insufficient.",
                "confidence": 0.78,
                "proposed_actions": [
                    {"parameter": "reinforcement_learning.entropy_coef", "action": "increase", "factor": 2.0},
                    {"parameter": "emotion.curiosity_weight", "action": "increase", "factor": 1.5},
                ],
                "explanation": "Boosting curiosity and entropy bonus forces exploration of alternate navigation paths."
            })
        else:
            return json.dumps({
                "hypothesis": "Training dynamics are progressing steadily within expected bounds.",
                "confidence": 0.88,
                "proposed_actions": [
                    {"parameter": "training.autosave_interval_seconds", "action": "maintain", "value": 60}
                ],
                "explanation": "Current loss and affective metrics indicate smooth convergence."
            })


class OpenAICompatibleProvider(LLMProvider):
    """Integrates with Ollama, vLLM, LMStudio, or OpenAI API."""

    def __init__(self, api_base: str = "http://localhost:11434/v1", api_key: str = "", model: str = "llama3"):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model

    def generate_completion(self, system_prompt: str, user_prompt: str) -> str:
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 800,
        }

        try:
            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning("Remote LLM request failed (%s). Falling back to heuristic reasoning engine.", e)
            fallback = MockLLMProvider()
            return fallback.generate_completion(system_prompt, user_prompt)


def create_llm_provider(provider_type: str = "local", api_base: str = "", api_key: str = "", model: str = "") -> LLMProvider:
    """Factory function for instantiating the appropriate LLM provider."""
    if provider_type in ("openai", "local"):
        base = api_base or "http://localhost:11434/v1"
        return OpenAICompatibleProvider(api_base=base, api_key=api_key, model=model or "llama3")
    return MockLLMProvider()
