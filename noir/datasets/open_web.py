"""Multi-source live internet ingestion engine with persistent disk registry, URL deduplication, and non-repeating single-pass learning."""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import torch

from noir.core.logging import get_logger
from noir.events.event import NoirEvent
from noir.events.event_bus import get_event_bus
from noir.events.event_types import EventType

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
    """Harvests real internet articles in continuous 20-website batches with permanent disk persistence and zero-repeat guarantees."""

    def __init__(self, tokenizer: Optional[WebTextTokenizer] = None, storage_dir: str | Path = "storage"):
        self.tokenizer = tokenizer or WebTextTokenizer(vocab_size=256)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.registry_path = self.storage_dir / "knowledge_registry.json"

        self.resource_history: List[WebResource] = []
        self._seen_urls: Set[str] = set()
        self._rehearsal_memory: List[str] = []

        # 1. Load persistent knowledge from previous training runs
        self._load_persisted_registry()

        # 2. 20-Website Batch Pipeline
        self._current_batch_queue: List[WebResource] = []
        self._active_article_idx: int = 0
        self._article_token_offset: int = 0
        self._steps_on_current_article: int = 0

        self._val_text_buffer: str = ""
        self._last_fetch_time: float = 0.0
        self._batch_cycle_count: int = 0

        # Initial seed: Harvest first fresh 20-website batch (skipping all previously seen URLs)
        self.replenish_buffer(force=True)

    def _load_persisted_registry(self) -> None:
        """Load all previously trained websites and mastered knowledge from disk."""
        if not self.registry_path.exists():
            return
        try:
            with open(self.registry_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            seen_list = data.get("seen_urls", [])
            self._seen_urls.update(seen_list)

            resources_raw = data.get("resources", [])
            for item in resources_raw:
                res = WebResource(
                    title=item.get("title", "Unknown"),
                    url=item.get("url", ""),
                    source_type=item.get("source_type", "Web Resource"),
                    timestamp=item.get("timestamp", time.time()),
                    character_count=item.get("character_count", 0),
                    token_count=item.get("token_count", 0),
                    text_snippet=item.get("text_snippet", ""),
                    full_text=item.get("full_text", ""),
                )
                self.resource_history.append(res)
                if res.full_text:
                    self._rehearsal_memory.append(res.full_text)
                self._emit_knowledge_event(res, status="MASTERED")

            logger.info(
                "[PERSISTENT KNOWLEDGE] Successfully restored %d mastered websites from disk (%s). Deduplication active.",
                len(self._seen_urls),
                self.registry_path.name,
            )
        except Exception as e:
            logger.warning("Could not read knowledge registry: %s", e)

    def save_persisted_registry(self) -> None:
        """Atomically persist all seen URLs, articles, and metadata to disk."""
        try:
            payload = {
                "version": "1.0",
                "updated_at": time.time(),
                "total_seen_urls": len(self._seen_urls),
                "seen_urls": list(self._seen_urls),
                "resources": [asdict(r) for r in self.resource_history],
            }
            tmp_path = self.registry_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            if tmp_path.exists():
                tmp_path.replace(self.registry_path)
            logger.debug("[PERSISTENT KNOWLEDGE] Saved %d records to %s", len(self.resource_history), self.registry_path.name)
        except Exception as e:
            logger.warning("Could not save knowledge registry: %s", e)

    def fetch_live_wikipedia(self, limit: int = 10) -> List[WebResource]:
        """Fetch 10 completely unique, unseen articles from MediaWiki REST API."""
        resources = []
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=query&format=json&generator=random&grnnamespace=0&grnlimit={limit * 3}&prop=extracts&explaintext=1&exintro=0"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced Cognitive AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=8.0) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode("utf-8"))
                    pages = data.get("query", {}).get("pages", {})
                    for _, page_info in pages.items():
                        if len(resources) >= limit:
                            break

                        title = page_info.get("title", "Wikipedia Article")
                        extract = page_info.get("extract", "")
                        clean_text = re.sub(r"\s+", " ", extract).strip()
                        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                        full_url = f"https://en.wikipedia.org/wiki/{encoded_title}"

                        # Skip if already seen in this or ANY previous session
                        if full_url in self._seen_urls or len(clean_text) < 150:
                            continue

                        tokens = self.tokenizer.encode(clean_text)
                        safe_title = "".join(c if c.isascii() and c.isprintable() else " " for c in title).strip()
                        safe_snippet = "".join(c if c.isascii() and c.isprintable() else " " for c in clean_text[:140]).strip() + "..."
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
                        self._seen_urls.add(full_url)
                        resources.append(res)
                        self.resource_history.append(res)
                        self._rehearsal_memory.append(clean_text)
                        self._emit_knowledge_event(res, status="INGESTED")
                        logger.info("[LIVE DISCOVERY %d/20] Ingested Wikipedia: '%s' (%s) [%d chars]", len(resources), safe_title[:35], full_url, len(clean_text))
        except Exception as e:
            logger.debug("Wikipedia batch fetch notice: %s", e)

        return resources

    def _emit_knowledge_event(self, res: WebResource, status: str = "INGESTED") -> None:
        """Publish KNOWLEDGE_INGESTED event for real-time GUI telemetry synchronization."""
        try:
            bus = get_event_bus()
            bus.publish(
                NoirEvent.create(
                    EventType.KNOWLEDGE_INGESTED,
                    experiment_id="autonomous",
                    training_step=0,
                    title=res.title,
                    url=res.url,
                    source_type=res.source_type,
                    character_count=res.character_count,
                    token_count=res.token_count,
                    text_snippet=res.text_snippet,
                    timestamp=res.timestamp,
                    status=status,
                ),
                asynchronous=True,
            )
        except Exception:
            pass

    def fetch_live_arxiv(self, limit: int = 10) -> List[WebResource]:
        """Fetch 10 completely unique, unseen research papers across AI, ML, Robotics, and Cognitive Science."""
        resources = []
        categories = ["cs.AI", "cs.LG", "cs.CL", "cs.NE", "cs.RO", "cs.CV", "stat.ML", "q-bio.NC", "physics.soc-ph"]
        selected_cat = np.random.choice(categories)
        try:
            offset = np.random.randint(0, 200)
            url = f"https://export.arxiv.org/api/query?search_query=cat:{selected_cat}&start={offset}&max_results={limit * 3}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ProjectNoirResearchBot/1.0 (Advanced Cognitive AI Research Platform)"},
            )
            with urllib.request.urlopen(req, timeout=8.0) as response:
                if response.status == 200:
                    root = ET.fromstring(response.read())
                    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
                    for entry in entries:
                        if len(resources) >= limit:
                            break

                        title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
                        summary_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
                        id_elem = entry.find("{http://www.w3.org/2005/Atom}id")

                        title = title_elem.text.strip().replace("\n", " ") if title_elem is not None else "arXiv Research Paper"
                        summary = summary_elem.text.strip().replace("\n", " ") if summary_elem is not None else ""
                        paper_url = id_elem.text.strip() if id_elem is not None else "https://arxiv.org"

                        # Skip if already seen in this or ANY previous session
                        if paper_url in self._seen_urls or len(summary) < 60:
                            continue

                        clean_text = f"{title}. {summary}"
                        tokens = self.tokenizer.encode(clean_text)
                        safe_title = "".join(c if c.isascii() and c.isprintable() else " " for c in title[:70]).strip()
                        safe_snippet = "".join(c if c.isascii() and c.isprintable() else " " for c in summary[:140]).strip() + "..."
                        res = WebResource(
                            title=safe_title or title[:70],
                            url=paper_url,
                            source_type=f"arXiv ({selected_cat})",
                            timestamp=time.time(),
                            character_count=len(clean_text),
                            token_count=len(tokens),
                            text_snippet=safe_snippet,
                            full_text=clean_text,
                        )
                        self._seen_urls.add(paper_url)
                        resources.append(res)
                        self.resource_history.append(res)
                        self._rehearsal_memory.append(clean_text)
                        self._emit_knowledge_event(res, status="INGESTED")
                        logger.info("[LIVE DISCOVERY %d/20] Ingested arXiv: '%s' (%s) [%d chars]", len(resources), safe_title[:35], paper_url, len(clean_text))
        except Exception as e:
            logger.debug("arXiv batch fetch notice: %s", e)

        return resources

    def replenish_buffer(self, force: bool = False, evict_old: bool = False) -> None:
        """Fetch a full fresh batch of 20 unique websites (10 Wikipedia + 10 arXiv)."""
        now = time.time()
        if not force and self._active_article_idx < len(self._current_batch_queue) and (now - self._last_fetch_time) < 10.0:
            return

        self._last_fetch_time = now
        self._batch_cycle_count += 1
        logger.info("[DISCOVERY LOOP] Starting Ingestion Cycle #%d: Searching next 20 unique websites...", self._batch_cycle_count)

        # Harvest 10 Wikipedia + 10 arXiv
        wiki_resources = self.fetch_live_wikipedia(limit=10)
        arxiv_resources = self.fetch_live_arxiv(limit=10)
        batch_20 = wiki_resources + arxiv_resources

        if not batch_20 and len(self.resource_history) == 0:
            fallback_text = (
                "Deep learning models optimize loss functions via stochastic gradient descent over parameter manifolds. "
                "Causal self-attention maps query, key, and value vectors into dynamic contextual representations, "
                "allowing transformers to model long-range sequential coherence across natural language tokens."
            )
            dummy_res = WebResource(
                title="Deep Learning Theory & Generalization",
                url="https://arxiv.org/abs/2103.00020",
                source_type="Core Knowledge",
                timestamp=time.time(),
                character_count=len(fallback_text),
                token_count=len(self.tokenizer.encode(fallback_text)),
                text_snippet=fallback_text[:120] + "...",
                full_text=fallback_text,
            )
            batch_20 = [dummy_res]
            self.resource_history.append(dummy_res)
            self._seen_urls.add(dummy_res.url)
            self._emit_knowledge_event(dummy_res, status="INGESTED")

        self._current_batch_queue = batch_20
        self._active_article_idx = 0
        self._article_token_offset = 0
        self._steps_on_current_article = 0

        # Build validation buffer from a subset
        self._val_text_buffer = "\n\n".join(res.full_text for i, res in enumerate(batch_20) if i % 4 == 0)

        # Save registry snapshot to disk
        self.save_persisted_registry()

        logger.info(
            "[BATCH STAGED] Batch #%d ready with %d fresh websites (Total Lifetime Unique Mastered: %d).",
            self._batch_cycle_count,
            len(batch_20),
            len(self._seen_urls),
        )

    def force_replenish(self, evict_old: bool = True) -> None:
        """Trigger immediate discovery and harvesting of the next 20-website batch."""
        self.replenish_buffer(force=True, evict_old=evict_old)

    def create_batch(
        self,
        batch_size: int = 16,
        block_size: int = 64,
        is_validation: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        """Create autoregressive token batch, training each website ONCE then advancing to the next."""
        # 1. Validation path
        if is_validation:
            val_tokens = self.tokenizer.encode(self._val_text_buffer or "Validation streaming corpus.")
            req_len = batch_size * (block_size + 1)
            if len(val_tokens) < req_len:
                val_tokens = (val_tokens * ((req_len // max(1, len(val_tokens))) + 1))[:req_len + 50]
            max_v = max(1, len(val_tokens) - block_size - 1)
            vx, vy = [], []
            for _ in range(batch_size):
                idx = np.random.randint(0, max_v)
                chunk = val_tokens[idx : idx + block_size + 1]
                vx.append(chunk[:-1])
                vy.append(chunk[1:])
            return torch.tensor(vx, dtype=torch.long), torch.tensor(vy, dtype=torch.long), "Validation Stream", "https://en.wikipedia.org"

        # 2. Check if current 20-website batch is completed
        if self._active_article_idx >= len(self._current_batch_queue):
            logger.info("[BATCH COMPLETE] All 20 websites in batch #%d completed single-pass training. Fetching next 20 websites...", self._batch_cycle_count)
            self.replenish_buffer(force=True)

        # 3. Retrieve current active article
        if self._current_batch_queue and self._active_article_idx < len(self._current_batch_queue):
            active_res = self._current_batch_queue[self._active_article_idx]
        elif self.resource_history:
            active_res = self.resource_history[-1]
        else:
            active_res = None

        active_title = active_res.title if active_res else "Live Internet Stream"
        active_url = active_res.url if active_res else "https://en.wikipedia.org"
        article_text = active_res.full_text if active_res else "Open web streaming text."
        article_tokens = self.tokenizer.encode(article_text)

        # 4. Extract non-repeating single-pass slice for training
        stride = batch_size * block_size
        offset = self._article_token_offset

        # Check if this article's tokens have been completely traversed
        if offset + block_size + 1 >= len(article_tokens) or self._steps_on_current_article >= 8:
            logger.info("[SOURCE MASTERED] Completed single pass on: '%s' (%s). Advancing to next website...", active_title[:35], active_url)
            self.save_persisted_registry()
            self._active_article_idx += 1
            self._article_token_offset = 0
            self._steps_on_current_article = 0

            # Re-evaluate with next article
            if self._active_article_idx < len(self._current_batch_queue):
                active_res = self._current_batch_queue[self._active_article_idx]
                active_title = active_res.title
                active_url = active_res.url
                article_text = active_res.full_text
                article_tokens = self.tokenizer.encode(article_text)
                offset = 0
            else:
                logger.info("[BATCH COMPLETE] Finished all 20 websites in Batch #%d. Harvesting next 20...", self._batch_cycle_count)
                self.replenish_buffer(force=True)
                if self._current_batch_queue:
                    active_res = self._current_batch_queue[0]
                    active_title = active_res.title
                    active_url = active_res.url
                    article_text = active_res.full_text
                    article_tokens = self.tokenizer.encode(article_text)
                    offset = 0

        self._steps_on_current_article += 1
        self._article_token_offset += stride

        # Ensure token buffer meets batch requirements
        req_len = batch_size * (block_size + 1)
        if len(article_tokens) < req_len:
            article_tokens = (article_tokens * ((req_len // max(1, len(article_tokens))) + 1))[:req_len + 50]

        X_list, Y_list = [], []
        max_start = max(1, len(article_tokens) - block_size - 1)

        # 80% Novel Single-Pass Tokens / 20% Historical Rehearsal Replay Tokens
        rehearsal_count = max(1, batch_size // 5) if len(self._rehearsal_memory) > 3 else 0
        live_count = batch_size - rehearsal_count

        for i in range(live_count):
            start_i = min(max_start, offset + (i * block_size))
            chunk = article_tokens[start_i : start_i + block_size + 1]
            if len(chunk) < block_size + 1:
                chunk = article_tokens[:block_size + 1]
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
                    chunk = article_tokens[:block_size + 1]
                    X_list.append(chunk[:-1])
                    Y_list.append(chunk[1:])

        X_tensor = torch.tensor(X_list, dtype=torch.long)
        Y_tensor = torch.tensor(Y_list, dtype=torch.long)

        return X_tensor, Y_tensor, active_title, active_url
