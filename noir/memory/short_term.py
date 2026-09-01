"""Short-term operational memory buffer."""

from collections import deque
import time
from typing import Any, Dict, List, Optional


class ShortTermMemory:
    """Rolling memory buffer storing recent actions, states, and perceptual context."""

    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.buffer: deque[Dict[str, Any]] = deque(maxlen=capacity)

    def add(self, item_type: str, data: Dict[str, Any], step: int = 0) -> None:
        """Append an observation or state item."""
        self.buffer.append({
            "type": item_type,
            "data": data,
            "step": step,
            "timestamp": time.time(),
        })

    def get_recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retrieve latest N memory entries."""
        return list(self.buffer)[-limit:]

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)
