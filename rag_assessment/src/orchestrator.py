from __future__ import annotations

import json
import logging
from pathlib import Path

from .config import config
from .interfaces import BaseEmbeddingModel, BaseQueryExpander, BaseVectorStore, RetrievalResult

logger = logging.getLogger(__name__)


class ContextAwareRetrievalEngine:
    """
    Orchestrates the full RAG pipeline using dependency-injected components.

    Accepts any concrete implementation of BaseEmbeddingModel, BaseVectorStore,
    and BaseQueryExpander, making it trivial to swap local mocks for production
    Vertex AI clients without changing orchestration logic.
    """

    def __init__(
        self,
        embedding_model: BaseEmbeddingModel,
        vector_store: BaseVectorStore,
        query_expander: BaseQueryExpander,
    ) -> None:
        self._embedder = embedding_model
        self._store = vector_store
        self._expander = query_expander
        self._ingested = False

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_dataset(self, filepath: str | Path) -> int:
        """Read JSON dataset, embed all chunks, and index in the vector store."""
        filepath = Path(filepath)
        with open(filepath, encoding="utf-8") as fh:
            records = json.load(fh)

        texts = [r["content"] for r in records]
        logger.info("Ingesting %d chunks from %s", len(texts), filepath)

        embeddings = self._embedder.embed(texts)
        self._store.add_texts(texts, embeddings)
        self._ingested = True

        logger.info("Successfully indexed %d chunks (dim=%d)", len(texts), embeddings.shape[1])
        return len(texts)

    # ------------------------------------------------------------------
    # Strategy A — Raw Vector Search
    # ------------------------------------------------------------------

    def strategy_a_search(
        self, query: str, k: int = config.top_k
    ) -> list[RetrievalResult]:
        """Embed the raw query and retrieve top-k results directly."""
        self._require_ingested()
        query_vec = self._embedder.embed([query])[0]
        results = self._store.search(query_vec, k=k)
        logger.debug("[Strategy A] query=%r → %d results", query[:60], len(results))
        return results

    # ------------------------------------------------------------------
    # Strategy B — AI-Enhanced Retrieval
    # ------------------------------------------------------------------

    def strategy_b_search(
        self, query: str, k: int = config.top_k
    ) -> tuple[str, list[RetrievalResult]]:
        """Expand query via GenerativeModel, embed expanded text, retrieve top-k."""
        self._require_ingested()
        expanded_query = self._expander.expand(query)
        logger.debug("[Strategy B] expanded=%r", expanded_query[:80])

        query_vec = self._embedder.embed([expanded_query])[0]
        results = self._store.search(query_vec, k=k)
        logger.debug("[Strategy B] query=%r → %d results", query[:60], len(results))
        return expanded_query, results

    # ------------------------------------------------------------------

    def _require_ingested(self) -> None:
        if not self._ingested:
            raise RuntimeError(
                "Dataset not ingested. Call ingest_dataset() before searching."
            )
