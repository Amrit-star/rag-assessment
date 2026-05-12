from __future__ import annotations

import pickle
from pathlib import Path

import faiss
import numpy as np

from .interfaces import BaseVectorStore, RetrievalResult


class FAISSVectorStore(BaseVectorStore):
    """
    Vector store backed by FAISS IndexFlatIP on L2-normalized vectors.

    Using IndexFlatIP (inner product) on unit vectors is mathematically
    equivalent to cosine similarity:
        cos(A, B) = A·B  when ||A|| = ||B|| = 1

    This avoids the magnitude bias of Euclidean (L2) distance, which would
    penalise longer documents even when they are semantically identical to
    shorter ones — a critical correctness requirement for text retrieval.
    """

    def __init__(self, dim: int) -> None:
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._id_to_chunk: dict[int, str] = {}
        self._next_id: int = 0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_texts(self, texts: list[str], embeddings: np.ndarray) -> None:
        assert embeddings.shape[1] == self._dim, (
            f"Embedding dim mismatch: expected {self._dim}, got {embeddings.shape[1]}"
        )
        vecs = embeddings.copy().astype(np.float32)
        faiss.normalize_L2(vecs)  # guarantee unit vectors even if caller skipped it
        self._index.add(vecs)
        for i, text in enumerate(texts):
            self._id_to_chunk[self._next_id + i] = text
        self._next_id += len(texts)

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, k: int = 3) -> list[RetrievalResult]:
        effective_k = min(k, self._index.ntotal)
        if effective_k == 0:
            return []

        q = query_embedding.reshape(1, -1).copy().astype(np.float32)
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, effective_k)

        results: list[RetrievalResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx == -1:
                continue
            results.append(
                RetrievalResult(
                    text=self._id_to_chunk[int(idx)],
                    score=float(score),
                    rank=rank,
                    chunk_id=int(idx),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence (production-awareness signal)
    # ------------------------------------------------------------------

    def save_index(self, directory: Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "wb") as fh:
            pickle.dump({"id_to_chunk": self._id_to_chunk, "next_id": self._next_id}, fh)

    @classmethod
    def load_index(cls, directory: Path, dim: int) -> "FAISSVectorStore":
        directory = Path(directory)
        store = cls(dim)
        store._index = faiss.read_index(str(directory / "index.faiss"))
        with open(directory / "metadata.pkl", "rb") as fh:
            meta = pickle.load(fh)
        store._id_to_chunk = meta["id_to_chunk"]
        store._next_id = meta["next_id"]
        return store
