"""Shared engine factory for all Streamlit pages — with index persistence."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import config
from src.embeddings import SentenceTransformerEmbedder
from src.orchestrator import ContextAwareRetrievalEngine
from src.query_expansion import MockVertexGenerativeModel
from src.vector_store import FAISSVectorStore

INDEX_STORE = ROOT / "index_store"
_INDEX_FILE = INDEX_STORE / "index.faiss"


def _index_is_fresh() -> bool:
    """True when a saved index exists AND is newer than the corpus file."""
    if not _INDEX_FILE.exists():
        return False
    return _INDEX_FILE.stat().st_mtime >= config.data_path.stat().st_mtime


@st.cache_resource(show_spinner=False)
def get_engine() -> ContextAwareRetrievalEngine:
    embedder = SentenceTransformerEmbedder()
    expander = MockVertexGenerativeModel()

    if _index_is_fresh():
        with st.spinner("Loading pre-built FAISS index from disk…"):
            store = FAISSVectorStore.load_index(INDEX_STORE, dim=config.embedding_dim)
        engine = ContextAwareRetrievalEngine(embedder, store, expander)
        engine._ingested = True          # index already built — skip ingest_dataset()
    else:
        with st.spinner("First run: embedding 100 chunks and building FAISS index…"):
            store = FAISSVectorStore(dim=config.embedding_dim)
            engine = ContextAwareRetrievalEngine(embedder, store, expander)
            engine.ingest_dataset(config.data_path)
            store.save_index(INDEX_STORE)  # persist for all future restarts

    return engine
