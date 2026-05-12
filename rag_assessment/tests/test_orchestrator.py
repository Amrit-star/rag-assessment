from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.interfaces import RetrievalResult
from src.orchestrator import ContextAwareRetrievalEngine


class TestContextAwareRetrievalEngine:

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def test_ingest_returns_correct_count(self, engine: ContextAwareRetrievalEngine, data_file: Path):
        n = engine.ingest_dataset(data_file)
        assert n == 5  # SAMPLE_TEXTS has 5 items

    def test_search_before_ingest_raises(self, engine: ContextAwareRetrievalEngine):
        with pytest.raises(RuntimeError, match="not ingested"):
            engine.strategy_a_search("test query")

    # ------------------------------------------------------------------
    # Strategy A
    # ------------------------------------------------------------------

    def test_strategy_a_returns_list_of_retrieval_results(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        results = ingested_engine.strategy_a_search("load balancing traffic")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_strategy_a_returns_top_k(self, ingested_engine: ContextAwareRetrievalEngine):
        results = ingested_engine.strategy_a_search("load balancing", k=2)
        assert len(results) == 2

    def test_strategy_a_results_sorted_descending(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        results = ingested_engine.strategy_a_search("database failover replica")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_a_never_calls_generate_content(
        self, ingested_engine: ContextAwareRetrievalEngine, mocker
    ):
        spy = mocker.spy(ingested_engine._expander, "generate_content")
        ingested_engine.strategy_a_search("How does the system scale?")
        spy.assert_not_called()

    # ------------------------------------------------------------------
    # Strategy B
    # ------------------------------------------------------------------

    def test_strategy_b_returns_tuple(self, ingested_engine: ContextAwareRetrievalEngine):
        result = ingested_engine.strategy_b_search("How does the system handle peak load?")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_strategy_b_first_element_is_expanded_query(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        expanded_query, _ = ingested_engine.strategy_b_search(
            "How does the system handle peak load?"
        )
        assert isinstance(expanded_query, str)
        assert "peak load" in expanded_query  # original query preserved

    def test_strategy_b_second_element_is_retrieval_results(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        _, results = ingested_engine.strategy_b_search("database failure")
        assert isinstance(results, list)
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_strategy_b_calls_generate_content_exactly_once(
        self, ingested_engine: ContextAwareRetrievalEngine, mocker
    ):
        spy = mocker.spy(ingested_engine._expander, "generate_content")
        ingested_engine.strategy_b_search("How does the system handle peak load?")
        spy.assert_called_once()

    def test_strategy_b_results_sorted_descending(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        _, results = ingested_engine.strategy_b_search("user session JWT Redis")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    # ------------------------------------------------------------------
    # Cross-strategy consistency
    # ------------------------------------------------------------------

    def test_strategy_b_expanded_query_is_longer(
        self, ingested_engine: ContextAwareRetrievalEngine
    ):
        query = "How does the system handle peak load?"
        expanded, _ = ingested_engine.strategy_b_search(query)
        assert len(expanded) > len(query)
