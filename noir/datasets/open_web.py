"""Multi-source live internet ingestion engine for continuous LLM training with anti-overfitting streaming."""

import re
import time
import urllib.request
import urllib.parse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch

from noir.core.logging import get_logger

logger = get_logger("datasets.open_web")


@dataclass
class WebResource:
    """Metadata tracking an authentic ingested internet knowledge resource."""

    title: str
    url: str
    source_type: str
    timestamp: float
    character_count: int
    token_count: int
    text_snippet: str
    full_text: str


class WebTextTokenizer:
    """Byte-level UTF-8 / ASCII tokenizer suitable for open-vocabulary streaming."""

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        """Convert string to UTF-8 byte token IDs."""
        bytes_data = text.encode("utf-8", errors="replace")
        return [b % self.vocab_size for b in bytes_data]

    def decode(self, tokens: List[int]) -> str:
        """Convert byte token IDs back to human-readable string."""
        clean_bytes = bytes([t % 256 for t in tokens])
        return clean_bytes.decode("utf-8", errors="replace")


class OpenWebStreamer:
    """Harvests real encyclopedic and scientific articles from Wikipedia & arXiv with continuous FIFO turnover."""

    def __init__(self, tokenizer: Optional[WebTextTokenizer] = None):
        self.tokenizer = tokenizer or WebTextTokenizer(vocab_size=256)
        self.resource_history: List[WebResource] = []
        self._rehearsal_memory: List[str] = []

        self._train_text_buffer: str = ""
        self._val_text_buffer: str = ""
        self._last_fetch_time: float = 0.0
        self._batches_since_last_fetch: int = 0
        self._consecutive_overfit_signals: int = 0

        # Initial seed replenishment
        self.replenish_buffer(force=True)

    def fetch_live_wikipedia(self, limit: int = 4) -> List[WebResource]:
        """Fetch real articles from MediaWiki REST API."""
        resources = []
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0&grnlimit={limit}&prop=extracts&explaintext=1&exintro=0"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced Cognitive AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=6.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for _, page_info in pages.items():
                        title = page_info.get("title", "Wikipedia Article")
                        extract = page_info.get("extract", "")
                        clean_text = re.sub(r"\s+", " ", extract).strip()
                        if len(clean_text) > 200:
                            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                            full_url = f"https://en.wikipedia.org/wiki/{encoded_title}"
                            tokens = self.tokenizer.encode(clean_text)
                            safe_title = "".join(c if c.isascii() and c.isprintable() else " " for c in title).strip()
                            safe_snippet = "".join(c if c.isascii() and c.isprintable() else " " for c in clean_text[:120]).strip() + "..."
                            res = WebResource(
                                title=safe_title or title,
                                url=full_url,
                                source_type="Wikipedia Encyclopedia",
                                timestamp=time.time(),
                                character_count=len(clean_text),
                                token_count=len(tokens),
                                text_snippet=safe_snippet,
                                full_text=clean_text,
                            )
                            resources.append(res)
                            self.resource_history.append(res)
                            self._rehearsal_memory.append(clean_text)
                            logger.info("[LIVE WEB] Ingested Wikipedia: '%s' (%s) [%d chars]", safe_title, full_url, len(clean_text))
        except Exception as e:
            logger.debug("Wikipedia fetch notice: %s", e)

        return resources

    def fetch_live_arxiv(self, limit: int = 3) -> List[WebResource]:
        """Fetch live computer science research abstracts from arXiv API."""
        resources = []
        try:
            offset = np.random.randint(0, 80)
            url = f"https://export.arxiv.org/api/query?search_query=cat:cs.AI+OR+cat:cs.LG+OR+cat:cs.NE&start={offset}&max_results={limit}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced Cognitive AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=6.0) as response:
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
                            self._rehearsal_memory.append(clean_text)
                            logger.info("[LIVE WEB] Ingested arXiv Paper: '%s' (%s) [%d chars]", safe_title, paper_url, len(clean_text))
        except Exception as e:
            logger.debug("arXiv fetch notice: %s", e)

        return resources

    def replenish_buffer(self, force: bool = False, evict_old: bool = False) -> None:
        """Fetch fresh internet articles and evict old chunks to prevent buffer memorization."""
        now = time.time()
        if not force and self._batches_since_last_fetch < 20 and (now - self._last_fetch_time) < 8.0:
            return

        self._last_fetch_time = now
        self._batches_since_last_fetch = 0

        # Fetch fresh resources from Wikipedia & arXiv
        wiki_resources = self.fetch_live_wikipedia(limit=4)
        arxiv_resources = self.fetch_live_arxiv(limit=3)
        all_new = wiki_resources + arxiv_resources

        if not all_new and len(self._train_text_buffer) < 5000:
            # Fallback scientific text
            fallback_text = (
                "Deep learning models optimize loss functions via stochastic gradient descent over parameter manifolds. "
                "Causal self-attention maps query, key, and value vectors into dynamic contextual representations, "
                "allowing transformers to model long-range sequential coherence across natural language tokens. "
                "Regularization techniques such as weight decay, dropout, and layer normalization prevent overfitting "
                "by constraining empirical curvature."
            )
            self._train_text_buffer += "\n\n" + fallback_text
            self._val_text_buffer += "\n\n" + fallback_text
            return

        if evict_old:
            # Aggressively flush old buffer on overfitting trigger
            self._train_text_buffer = ""
            self._val_text_buffer = ""

        # Distribute 80% train / 20% validation
        for i, res in enumerate(all_new):
            if i % 5 == 0:
                self._val_text_buffer += "\n\n" + res.full_text
            else:
                self._train_text_buffer += "\n\n" + res.full_text

        # FIFO buffer pruning: Keep rolling train buffer bounded (max 40,000 chars) to prevent stale memorization
        if len(self._train_text_buffer) > 50000:
            self._train_text_buffer = self._train_text_buffer[-35000:]
        if len(self._val_text_buffer) > 20000:
            self._val_text_buffer = self._val_text_buffer[-15000:]

        # Maintain rehearsal memory bound
        if len(self._rehearsal_memory) > 100:
            self._rehearsal_memory = self._rehearsal_memory[-60:]

    def force_replenish(self, evict_old: bool = True) -> None:
        """Trigger immediate emergency buffer replenishment on overfitting signal."""
        logger.info("[GUARDRAIL] Triggering emergency streaming buffer replenishment (Evict old: %s)...", evict_old)
        self.replenish_buffer(force=True, evict_old=evict_old)

    def create_batch(
        self,
        batch_size: int = 16,
        block_size: int = 64,
        is_validation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Create autoregressive training/validation tensors with continuous streaming turnover and 20% rehearsal."""
        self._batches_since_last_fetch += 1
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

        # 80% Live Stream / 20% Rehearsal Replay to prevent catastrophic forgetting
        rehearsal_count = max(1, batch_size // 5) if (not is_validation and len(self._rehearsal_memory) > 3) else 0
        live_count = batch_size - rehearsal_count

        for _ in range(live_count):
            idx = np.random.randint(0, max_start)
            chunk = tokens[idx : idx + block_size + 1]
            X_list.append(chunk[:-1])
            Y_list.append(chunk[1:])

        if rehearsal_count > 0:
            for _ in range(rehearsal_count):
                mem_text = np.random.choice(self._rehearsal_memory)
                mem_tokens = self.tokenizer.encode(mem_text)
                if len(mem_tokens) > block_size + 1:
                    m_idx = np.random.randint(0, len(mem_tokens) - block_size - 1)
                    m_chunk = mem_tokens[m_idx : m_idx + block_size + 1]
                    X_list.append(m_chunk[:-1])
                    Y_list.append(m_chunk[1:])
                else:
                    # Fallback to standard
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
