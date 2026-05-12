from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import config
from src.embeddings import SentenceTransformerEmbedder
from src.orchestrator import ContextAwareRetrievalEngine
from src.query_expansion import MockVertexGenerativeModel
from src.vector_store import FAISSVectorStore

SAMPLE_TEXTS = [
    "The load balancer distributes traffic across multiple GKE pod replicas using HPA auto-scaling.",
    "Database failover to the standby replica is automated within 60 seconds of primary failure.",
    "User sessions are stateless and managed via JWT tokens stored in Memorystore for Redis.",
    "Cloud Monitoring tracks p99 latency SLOs and fires alerts when error budget burn rate exceeds threshold.",
    "Pub/Sub decouples producers from consumers enabling independent horizontal scaling of each service.",
]

SAMPLE_JSON_DATA = [
    {"id": i + 1, "topic": f"topic_{i}", "content": text}
    for i, text in enumerate(SAMPLE_TEXTS)
]


@pytest.fixture(scope="session")
def embedder() -> SentenceTransformerEmbedder:
    # Session-scoped: model load is slow (~2s), but the model is stateless.
    return SentenceTransformerEmbedder()


@pytest.fixture(scope="session")
def sample_embeddings(embedder: SentenceTransformerEmbedder):
    return embedder.embed(SAMPLE_TEXTS)


@pytest.fixture
def vector_store() -> FAISSVectorStore:
    return FAISSVectorStore(dim=config.embedding_dim)


@pytest.fixture
def expander() -> MockVertexGenerativeModel:
    return MockVertexGenerativeModel()


@pytest.fixture
def data_file(tmp_path: Path) -> Path:
    path = tmp_path / "test_data.json"
    path.write_text(json.dumps(SAMPLE_JSON_DATA), encoding="utf-8")
    return path


@pytest.fixture
def engine(embedder, vector_store, expander) -> ContextAwareRetrievalEngine:
    return ContextAwareRetrievalEngine(embedder, vector_store, expander)


@pytest.fixture
def ingested_engine(engine: ContextAwareRetrievalEngine, data_file: Path) -> ContextAwareRetrievalEngine:
    engine.ingest_dataset(data_file)
    return engine
