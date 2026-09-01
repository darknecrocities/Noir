"""Live Open Web and Internet Text Ingestion and Tokenization Stream."""

import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import urllib.request
import urllib.parse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from noir.core.logging import get_logger

logger = get_logger("datasets.open_web")


class WebTextTokenizer:
    """Byte-level and Character-level Tokenizer for raw internet text streams."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.char_to_id: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

        # Initialize ASCII/UTF-8 base mapping
        for i in range(256):
            ch = chr(i)
            self.char_to_id[ch] = i
            self.id_to_char[i] = ch

    def encode(self, text: str) -> List[int]:
        """Convert text into token IDs (byte-level ASCII/UTF-8 safe)."""
        bytes_data = text.encode("utf-8", errors="ignore")
        return [b for b in bytes_data]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to human-readable string."""
        bytes_data = bytes([min(max(0, int(t)), 255) for t in token_ids])
        return bytes_data.decode("utf-8", errors="replace")


class OpenWebStreamer:
    """Live internet crawler fetching real-time text from open web repositories."""

    # Public web text APIs requiring zero API keys
    WIKIPEDIA_RANDOM_API = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query&format=json&prop=extracts&exintro=1&explaintext=1&generator=random&grnnamespace=0&grnlimit=5"
    )
    HACKERNEWS_TOP_API = "https://hacker-news.firebaseio.com/v0/topstories.json"
    HACKERNEWS_ITEM_API = "https://hacker-news.firebaseio.com/v0/item/{}.json"

    # Offline fallback corpus in case network is disconnected
    FALLBACK_CORPUS = [
        (
            "Artificial intelligence and deep reinforcement learning represent transformative paradigms "
            "in computer science, enabling autonomous agents to acquire complex behaviors through trial and error "
            "in continuous and discrete state spaces. Neural network architectures like Transformers leverage "
            "multi-head self-attention mechanisms to model long-range contextual dependencies across tokens."
        ),
        (
            "Cognitive affective architectures model mathematical representations of emotion, curiosity, and surprise "
            "by deriving continuous vector states from predictive uncertainty, Shannon entropy, and gradient variance. "
            "Such intrinsic motivational drives empower artificial systems to explore unvisited state spaces."
        ),
        (
            "Modern distributed training algorithms scale gradient descent optimization across high-performance GPUs "
            "leveraging tensor parallelism, automatic mixed precision, and high-bandwidth interconnects to accelerate "
            "convergence on massive multi-modal internet datasets."
        ),
    ]

    def __init__(self, cache_dir: str | Path = "data/web_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = WebTextTokenizer()
        self._cached_texts: List[str] = list(self.FALLBACK_CORPUS)

    def fetch_live_web_articles(self, limit: int = 5) -> List[Dict[str, str]]:
        """Fetch live, authentic text articles from Wikipedia & Open Web APIs."""
        articles = []
        try:
            req = urllib.request.Request(
                self.WIKIPEDIA_RANDOM_API,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (AI Research Application)"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page_data in pages.items():
                        title = page_data.get("title", "Untitled")
                        extract = page_data.get("extract", "").strip()
                        if len(extract) > 100:
                            # Clean text formatting
                            clean_text = re.sub(r"\s+", " ", extract)
                            articles.append({"title": title, "text": clean_text, "source": f"Wikipedia: {title}"})
                            self._cached_texts.append(clean_text)

            logger.info("Successfully fetched %d live open web articles from Wikipedia API", len(articles))
        except Exception as e:
            logger.warning("Live web fetch encountered network exception: %s. Using cached open web corpus.", e)

        # Ensure we return at least fallback/cached articles
        if not articles:
            for i, text in enumerate(self._cached_texts[-limit:]):
                articles.append({
                    "title": f"Open Web Knowledge Slice {i+1}",
                    "text": text,
                    "source": "Open Web Knowledge Stream",
                })

        return articles

    def create_batch_stream(
        self,
        batch_size: int = 16,
        block_size: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """Fetch real web text and format into autoregressive next-token prediction tensors."""
        articles = self.fetch_live_web_articles(limit=5)
        combined_text = "\n\n".join(a["text"] for a in articles)
        active_title = articles[0]["title"] if articles else "Open Web Article Stream"

        tokens = self.tokenizer.encode(combined_text)

        # Pad tokens if needed
        required_len = batch_size * (block_size + 1)
        if len(tokens) < required_len:
            tokens = (tokens * ((required_len // len(tokens)) + 1))[:required_len + 10]

        # Sample random chunks for the batch
        X_list, Y_list = [], []
        max_start = max(1, len(tokens) - block_size - 1)

        for _ in range(batch_size):
            idx = np.random.randint(0, max_start)
            chunk = tokens[idx : idx + block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            X_list.append(x)
            Y_list.append(y)

        X_tensor = torch.tensor(X_list, dtype=torch.long)
        Y_tensor = torch.tensor(Y_list, dtype=torch.long)

        return X_tensor, Y_tensor, active_title
