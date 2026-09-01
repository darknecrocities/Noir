"""Live Open Web and Internet Text Ingestion Engine with multi-source streaming and resource tracking."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import time
from typing import Any, Dict, Generator, List, Optional, Tuple
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import numpy as np
import torch

from noir.core.logging import get_logger

logger = get_logger("datasets.open_web")


@dataclass
class WebResource:
    """Metadata and content for an authentic internet resource."""
    title: str
    url: str
    source_type: str
    timestamp: float
    character_count: int
    token_count: int
    text_snippet: str
    full_text: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "source_type": self.source_type,
            "timestamp": self.timestamp,
            "time_str": datetime.fromtimestamp(self.timestamp).strftime("%H:%M:%S"),
            "character_count": self.character_count,
            "token_count": self.token_count,
            "snippet": self.text_snippet,
        }


class WebTextTokenizer:
    """Byte-level and Character-level Tokenizer for raw internet text streams."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Convert text into token IDs (byte-level UTF-8 safe)."""
        bytes_data = text.encode("utf-8", errors="ignore")
        return [int(b) for b in bytes_data]

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to human-readable string safe across all terminals."""
        bytes_data = bytes([min(max(0, int(t)), 255) for t in token_ids])
        decoded = bytes_data.decode("utf-8", errors="ignore")
        # Ensure ASCII/console safe representation
        return "".join(c if c.isprintable() or c in ("\n", " ") else " " for c in decoded)


class OpenWebStreamer:
    """Live internet crawler fetching real-time text from Wikipedia, arXiv, and Hacker News."""

    WIKIPEDIA_RANDOM_API = (
        "https://en.wikipedia.org/w/api.php?"
        "action=query&format=json&prop=extracts|info&inprop=url&exintro=1&explaintext=1&generator=random&grnnamespace=0&grnlimit=4"
    )
    ARXIV_API = "https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&start=0&max_results=4"
    HACKERNEWS_TOP_API = "https://hacker-news.firebaseio.com/v0/topstories.json"
    HACKERNEWS_ITEM_API = "https://hacker-news.firebaseio.com/v0/item/{}.json"

    def __init__(self, cache_dir: str | Path = "data/web_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer = WebTextTokenizer()

        self.resource_history: List[WebResource] = []
        self._train_text_buffer: str = ""
        self._val_text_buffer: str = ""
        self._last_fetch_time: float = 0.0

    def get_resource_history(self) -> List[Dict[str, Any]]:
        """Return all internet resources fetched in chronological order."""
        return [r.to_dict() for r in self.resource_history]

    def fetch_live_wikipedia(self, limit: int = 4) -> List[WebResource]:
        """Fetch live random articles from Wikipedia API."""
        resources = []
        try:
            req = urllib.request.Request(
                self.WIKIPEDIA_RANDOM_API,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for page_id, page_data in pages.items():
                        title = page_data.get("title", "Untitled")
                        full_url = page_data.get("fullurl", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}")
                        extract = page_data.get("extract", "").strip()

                        if len(extract) > 80:
                            clean_text = re.sub(r"\s+", " ", extract)
                            tokens = self.tokenizer.encode(clean_text)
                            safe_title = "".join(c if c.isascii() and c.isprintable() else " " for c in title).strip()
                            safe_snippet = "".join(c if c.isascii() and c.isprintable() else " " for c in clean_text[:120]).strip() + "..."
                            res = WebResource(
                                title=safe_title or title,
                                url=full_url,
                                source_type="Wikipedia",
                                timestamp=time.time(),
                                character_count=len(clean_text),
                                token_count=len(tokens),
                                text_snippet=safe_snippet,
                                full_text=clean_text,
                            )
                            resources.append(res)
                            self.resource_history.append(res)
                            logger.info("[LIVE WEB] Ingested Wikipedia: '%s' (%s) [%d chars]", safe_title, full_url, len(clean_text))
        except Exception as e:
            logger.debug("Wikipedia fetch notice: %s", e)

        return resources

    def fetch_live_arxiv(self, limit: int = 3) -> List[WebResource]:
        """Fetch live computer science research abstracts from arXiv API."""
        resources = []
        try:
            offset = np.random.randint(0, 50)
            url = f"https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG&start={offset}&max_results={limit}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=5.0) as response:
                if response.status == 200:
                    root = ET.fromstring(response.read())
                    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                    for entry in entries:
                        title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                        summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                        id_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "arXiv Research"
                        summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                        paper_url = id_elem.text.strip() if id_elem is not None else "https://arxiv.org"

                        if len(summary) > 80:
                            clean_text = f"{title}. {summary}"
                            tokens = self.tokenizer.encode(clean_text)
                            safe_title = "".join(c if c.isascii() and c.isprintable() else " " for c in title[:60]).strip()
                            safe_snippet = "".join(c if c.isascii() and c.isprintable() else " " for c in summary[:120]).strip() + "..."
                            res = WebResource(
                                title=safe_title or title[:60],
                                url=paper_url,
                                source_type="arXiv Research",
                                timestamp=time.time(),
                                character_count=len(clean_text),
                                token_count=len(tokens),
                                text_snippet=safe_snippet,
                                full_text=clean_text,
                            )
                            resources.append(res)
                            self.resource_history.append(res)
                            logger.info("[LIVE WEB] Ingested arXiv Paper: '%s' (%s) [%d chars]", safe_title, paper_url, len(clean_text))
        except Exception as e:
            logger.debug("arXiv fetch notice: %s", e)

        return resources

    def replenish_buffer(self) -> None:
        """Fetch fresh internet articles to expand training and validation corpora."""
        now = time.time()
        # Fetch fresh batches every 5 seconds or if buffer is low
        if len(self._train_text_buffer) > 20000 and (now - self._last_fetch_time) < 10.0:
            return

        self._last_fetch_time = now

        # Fetch from Wikipedia & arXiv in parallel streams
        wiki_resources = self.fetch_live_wikipedia(limit=4)
        arxiv_resources = self.fetch_live_arxiv(limit=3)
        all_new = wiki_resources + arxiv_resources

        if not all_new:
            # Fallback real-world science text if temporarily offline
            fallback_text = (
                "Deep learning models optimize loss functions via stochastic gradient descent over parameter spaces. "
                "Causal self-attention maps query, key, and value vectors into dynamic contextual representations, "
                "allowing transformers to model long-range sequential coherence across natural language tokens. "
                "Regularization techniques such as dropout, weight decay, and layer normalization prevent overfitting "
                "by constraining the curvature of the optimization manifold."
            )
            self._train_text_buffer += "\n\n" + fallback_text
            self._val_text_buffer += "\n\n" + fallback_text
            return

        # Split 80% train / 20% validation to continuously measure generalization without overfitting
        for i, res in enumerate(all_new):
            if i % 5 == 0:
                self._val_text_buffer += "\n\n" + res.full_text
            else:
                self._train_text_buffer += "\n\n" + res.full_text

        # Keep rolling buffer size manageable
        if len(self._train_text_buffer) > 100000:
            self._train_text_buffer = self._train_text_buffer[-80000:]
        if len(self._val_text_buffer) > 30000:
            self._val_text_buffer = self._val_text_buffer[-20000:]

    def create_batch(
        self,
        batch_size: int = 16,
        block_size: int = 64,
        is_validation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Create autoregressive training/validation tensors from real internet stream."""
        self.replenish_buffer()

        text_source = self._val_text_buffer if is_validation else self._train_text_buffer
        if len(text_source) < (batch_size * block_size):
            text_source = self._train_text_buffer

        tokens = self.tokenizer.encode(text_source)
        required_len = batch_size * (block_size + 1)
        if len(tokens) < required_len:
            tokens = (tokens * ((required_len // max(1, len(tokens))) + 1))[:required_len + 50]

        X_list, Y_list = [], []
        max_start = max(1, len(tokens) - block_size - 1)

        for _ in range(batch_size):
            idx = np.random.randint(0, max_start)
            chunk = tokens[idx : idx + block_size + 1]
            X_list.append(chunk[:-1])
            Y_list.append(chunk[1:])

        X_tensor = torch.tensor(X_list, dtype=torch.long)
        Y_tensor = torch.tensor(Y_list, dtype=torch.long)

        latest_resource = self.resource_history[-1] if self.resource_history else None
        active_title = latest_resource.title if latest_resource else "Live Open Web Stream"
        active_url = latest_resource.url if latest_resource else "https://en.wikipedia.org"

        return X_tensor, Y_tensor, active_title, active_url
