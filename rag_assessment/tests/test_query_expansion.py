from __future__ import annotations

import pytest

from src.interfaces import BaseQueryExpander
from src.query_expansion import GenerateContentResponse, MockVertexGenerativeModel


class TestMockVertexGenerativeModel:
    def test_implements_base_query_expander(self, expander: MockVertexGenerativeModel):
        assert isinstance(expander, BaseQueryExpander)

    def test_expand_returns_string(self, expander: MockVertexGenerativeModel):
        result = expander.expand("How does the system handle peak load?")
        assert isinstance(result, str)

    def test_expand_longer_than_input(self, expander: MockVertexGenerativeModel):
        query = "How does the system handle peak load?"
        expanded = expander.expand(query)
        assert len(expanded) > len(query)

    def test_expand_contains_original_query(self, expander: MockVertexGenerativeModel):
        query = "How does the system handle peak load?"
        expanded = expander.expand(query)
        assert query in expanded

    def test_generate_content_returns_correct_type(self, expander: MockVertexGenerativeModel):
        response = expander.generate_content("Query: test query")
        assert isinstance(response, GenerateContentResponse)

    def test_generate_content_response_has_text_attribute(self, expander: MockVertexGenerativeModel):
        response = expander.generate_content("Query: test query")
        assert hasattr(response, "text")
        assert isinstance(response.text, str)

    def test_peak_load_expansion_includes_scaling_terms(self, expander: MockVertexGenerativeModel):
        expanded = expander.expand("How does the system handle peak load?")
        scaling_terms = {"scaling", "HPA", "load balancing", "auto-scaling", "traffic"}
        assert any(term.lower() in expanded.lower() for term in scaling_terms)

    def test_database_failure_expansion_includes_failover_terms(self, expander: MockVertexGenerativeModel):
        expanded = expander.expand("What happens when the database goes down?")
        failover_terms = {"failover", "replica", "recovery", "circuit breaker"}
        assert any(term.lower() in expanded.lower() for term in failover_terms)

    def test_session_expansion_includes_auth_terms(self, expander: MockVertexGenerativeModel):
        expanded = expander.expand("How do you keep users logged in?")
        auth_terms = {"JWT", "Redis", "stateless", "token", "Memorystore"}
        assert any(term.lower() in expanded.lower() for term in auth_terms)

    def test_unknown_query_still_returns_expanded_string(self, expander: MockVertexGenerativeModel):
        # Queries not matching any rule should still get generic GCP expansion
        expanded = expander.expand("Some completely unrelated topic xyz")
        assert len(expanded) > len("Some completely unrelated topic xyz")
