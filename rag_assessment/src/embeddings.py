from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import config
from .interfaces import BaseEmbeddingModel


# ---------------------------------------------------------------------------
# Vertex AI API-shape data classes (mirror the real SDK's contract)
# ---------------------------------------------------------------------------

@dataclass
class TextEmbeddingStatistics:
    token_count: int
    truncated: bool = False


@dataclass
class TextEmbedding:
    """Mirrors vertexai.language_models.TextEmbedding."""
    values: list[float]
    statistics: TextEmbeddingStatistics = field(default_factory=lambda: TextEmbeddingStatistics(token_count=0))


# ---------------------------------------------------------------------------
# Local embedding model (real implementation)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbeddingModel):
    """Wraps sentence-transformers and returns L2-normalized float32 vectors."""

    def __init__(self, model_name: str = config.embedding_model_name) -> None:
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # unit vectors → IndexFlatIP == cosine similarity
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)


# ---------------------------------------------------------------------------
# Mock that perfectly mirrors vertexai.language_models.TextEmbeddingModel
# ---------------------------------------------------------------------------

class MockVertexTextEmbeddingModel:
    """
    Drop-in mock for vertexai.language_models.TextEmbeddingModel.

    Delegates to SentenceTransformerEmbedder internally so embeddings are
    semantically meaningful while requiring zero GCP credentials.
    """

    def __init__(self) -> None:
        self._embedder = SentenceTransformerEmbedder()

    @classmethod
    def from_pretrained(cls, model_name: str) -> "MockVertexTextEmbeddingModel":  # noqa: ARG003
        return cls()

    def get_embeddings(self, texts: list[str]) -> list[TextEmbedding]:
        vectors = self._embedder.embed(texts)
        return [
            TextEmbedding(
                values=vec.tolist(),
                statistics=TextEmbeddingStatistics(
                    token_count=len(text.split()),
                    truncated=len(text.split()) > 512,
                ),
            )
            for text, vec in zip(texts, vectors)
        ]
