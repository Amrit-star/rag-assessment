from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RetrievalResult:
    text: str
    score: float
    rank: int
    chunk_id: int


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return L2-normalized embeddings of shape (len(texts), dim)."""
        ...


class BaseVectorStore(ABC):
    @abstractmethod
    def add_texts(self, texts: list[str], embeddings: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[RetrievalResult]:
        ...


class BaseQueryExpander(ABC):
    @abstractmethod
    def expand(self, query: str) -> str:
        """Return a keyword-enriched version of the query for semantic search."""
        ...
