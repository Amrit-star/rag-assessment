from __future__ import annotations

import numpy as np
import pytest

from src.interfaces import RetrievalResult
from src.vector_store import FAISSVectorStore
from tests.conftest import SAMPLE_TEXTS


class TestFAISSVectorStore:
    def test_add_and_search_returns_k_results(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        results = vector_store.search(sample_embeddings[0], k=3)
        assert len(results) == 3

    def test_results_are_retrieval_result_instances(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        results = vector_store.search(sample_embeddings[0], k=2)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_scores_in_cosine_range(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        results = vector_store.search(sample_embeddings[0], k=3)
        for r in results:
            assert 0.0 <= r.score <= 1.0 + 1e-5

    def test_results_sorted_descending(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        results = vector_store.search(sample_embeddings[0], k=4)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_exact_match_is_rank_one(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        # Searching with the stored vector of index 2 should return it as rank 1
        results = vector_store.search(sample_embeddings[2], k=3)
        assert results[0].chunk_id == 2
        assert results[0].score > 0.99

    def test_k_greater_than_n_does_not_raise(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS[:3], sample_embeddings[:3])
        results = vector_store.search(sample_embeddings[0], k=10)
        assert len(results) == 3  # capped at n_total

    def test_search_on_empty_store_returns_empty(self, vector_store: FAISSVectorStore):
        dummy_vec = np.random.rand(384).astype(np.float32)
        results = vector_store.search(dummy_vec, k=3)
        assert results == []

    def test_rank_field_is_one_indexed(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        results = vector_store.search(sample_embeddings[0], k=3)
        assert [r.rank for r in results] == [1, 2, 3]

    def test_save_and_load_roundtrip(
        self, vector_store: FAISSVectorStore, sample_embeddings: np.ndarray, tmp_path
    ):
        vector_store.add_texts(SAMPLE_TEXTS, sample_embeddings)
        original_results = vector_store.search(sample_embeddings[0], k=3)

        vector_store.save_index(tmp_path / "index")
        loaded = FAISSVectorStore.load_index(tmp_path / "index", dim=384)
        loaded_results = loaded.search(sample_embeddings[0], k=3)

        assert [r.chunk_id for r in original_results] == [r.chunk_id for r in loaded_results]
        for orig, load in zip(original_results, loaded_results):
            assert abs(orig.score - load.score) < 1e-5
