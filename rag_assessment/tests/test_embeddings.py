from __future__ import annotations

import numpy as np
import pytest

from src.embeddings import MockVertexTextEmbeddingModel, SentenceTransformerEmbedder
from src.interfaces import BaseEmbeddingModel
from tests.conftest import SAMPLE_TEXTS


class TestSentenceTransformerEmbedder:
    def test_output_shape(self, embedder: SentenceTransformerEmbedder):
        vecs = embedder.embed(SAMPLE_TEXTS)
        assert vecs.shape == (len(SAMPLE_TEXTS), 384)

    def test_vectors_are_float32(self, embedder: SentenceTransformerEmbedder):
        vecs = embedder.embed(SAMPLE_TEXTS)
        assert vecs.dtype == np.float32

    def test_vectors_are_unit_norm(self, embedder: SentenceTransformerEmbedder):
        vecs = embedder.embed(SAMPLE_TEXTS)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_single_text_embed(self, embedder: SentenceTransformerEmbedder):
        vecs = embedder.embed(["single sentence test"])
        assert vecs.shape == (1, 384)

    def test_implements_base_interface(self, embedder: SentenceTransformerEmbedder):
        assert isinstance(embedder, BaseEmbeddingModel)

    def test_similar_texts_closer_than_dissimilar(self, embedder: SentenceTransformerEmbedder):
        similar_a = embedder.embed(["How does load balancing work?"])[0]
        similar_b = embedder.embed(["What is a load balancer?"])[0]
        unrelated = embedder.embed(["The recipe calls for two cups of flour."])[0]

        sim_related = float(np.dot(similar_a, similar_b))
        sim_unrelated = float(np.dot(similar_a, unrelated))
        assert sim_related > sim_unrelated


class TestMockVertexTextEmbeddingModel:
    def test_from_pretrained_returns_instance(self):
        model = MockVertexTextEmbeddingModel.from_pretrained("textembedding-gecko@003")
        assert isinstance(model, MockVertexTextEmbeddingModel)

    def test_get_embeddings_returns_correct_count(self):
        model = MockVertexTextEmbeddingModel()
        results = model.get_embeddings(SAMPLE_TEXTS)
        assert len(results) == len(SAMPLE_TEXTS)

    def test_embedding_has_values_attribute(self):
        model = MockVertexTextEmbeddingModel()
        result = model.get_embeddings(["test sentence"])[0]
        assert hasattr(result, "values")
        assert isinstance(result.values, list)
        assert len(result.values) == 384

    def test_embedding_has_statistics_attribute(self):
        model = MockVertexTextEmbeddingModel()
        result = model.get_embeddings(["this is a test"])[0]
        assert hasattr(result, "statistics")
        assert result.statistics.token_count > 0

    def test_embedding_values_are_unit_norm(self):
        model = MockVertexTextEmbeddingModel()
        result = model.get_embeddings(["normalised embedding test"])[0]
        norm = np.linalg.norm(result.values)
        assert abs(norm - 1.0) < 1e-5
