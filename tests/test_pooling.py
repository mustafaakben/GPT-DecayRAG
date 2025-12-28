"""Tests for the pooling module."""
import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag.decayrag.pooling import (
    compute_tfidf_weights,
    compute_global_embedding,
    apply_neighbor_decay_scores,
    apply_neighbor_decay_embeddings,
    blend_embeddings,
)


class TestComputeTfidfWeights:
    def test_returns_weights_and_vectorizer(self):
        texts = ["hello world", "world of code", "code is great"]
        weights, vectorizer = compute_tfidf_weights(texts)
        assert weights.shape == (3,)
        assert weights.dtype == np.float32
        assert vectorizer is not None

    def test_weights_are_positive(self):
        texts = ["apple banana", "banana cherry", "cherry date"]
        weights, _ = compute_tfidf_weights(texts)
        assert np.all(weights > 0)

    def test_reusing_vectorizer(self):
        texts1 = ["cat dog", "dog bird"]
        weights1, vectorizer = compute_tfidf_weights(texts1)
        texts2 = ["cat bird", "dog cat"]
        weights2, _ = compute_tfidf_weights(texts2, vectorizer=vectorizer)
        assert weights2.shape == (2,)


class TestComputeGlobalEmbedding:
    def test_mean_method(self):
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="mean")
        expected = np.array([3.0, 4.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_min_method(self):
        embeddings = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="min")
        expected = np.array([1.0, 2.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_max_method(self):
        embeddings = np.array([[1.0, 5.0], [3.0, 2.0], [2.0, 4.0]], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="max")
        expected = np.array([3.0, 5.0], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_minmax_method(self):
        embeddings = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="minmax")
        expected = np.array([1.0, 2.0, 3.0, 5.0], dtype=np.float32)
        assert np.allclose(result, expected)
        assert result.shape == (4,)

    def test_tfidf_method(self):
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        weights = np.array([1.0, 3.0], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="tfidf", weights=weights)
        # Expected: (1*[1,2] + 3*[3,4]) / 4 = [10, 14] / 4 = [2.5, 3.5]
        expected = np.array([2.5, 3.5], dtype=np.float32)
        assert np.allclose(result, expected)

    def test_tfidf_method_requires_weights(self):
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="weights.*must be provided"):
            compute_global_embedding(embeddings, method="tfidf")

    def test_l2_normalize(self):
        embeddings = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)
        result = compute_global_embedding(embeddings, method="mean", l2_normalize=True)
        assert np.allclose(np.linalg.norm(result), 1.0)

    def test_invalid_method_raises(self):
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="Unknown pooling method"):
            compute_global_embedding(embeddings, method="invalid")

    def test_rejects_1d_input(self):
        embeddings = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="must have shape"):
            compute_global_embedding(embeddings)


class TestApplyNeighborDecayScores:
    def test_exponential_decay(self):
        scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = apply_neighbor_decay_scores(scores, decay_type="exp", param=1.0)
        assert result.shape == (3,)
        # First score should be highest, decay to neighbors
        assert result[0] > result[1] > result[2]

    def test_gaussian_decay(self):
        scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = apply_neighbor_decay_scores(scores, decay_type="gaussian", param=1.0)
        assert result.shape == (3,)
        assert result[0] > result[2]

    def test_power_decay(self):
        scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = apply_neighbor_decay_scores(scores, decay_type="power", param=1.0)
        assert result.shape == (3,)
        assert result[0] > result[1]

    def test_auto_param_scaling(self):
        scores = np.array([1.0] * 100, dtype=np.float32)
        result = apply_neighbor_decay_scores(scores, decay_type="exp")
        assert result.shape == (100,)
        # With 100 chunks, auto param should be 10

    def test_empty_scores(self):
        scores = np.array([], dtype=np.float32)
        result = apply_neighbor_decay_scores(scores)
        assert result.shape == (0,)

    def test_rejects_2d_input(self):
        scores = np.array([[1.0, 2.0]], dtype=np.float32)
        with pytest.raises(ValueError, match="must be 1"):
            apply_neighbor_decay_scores(scores)

    def test_exclude_self(self):
        scores = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = apply_neighbor_decay_scores(
            scores, decay_type="exp", param=1.0, include_self=False, normalize=False
        )
        # Without self, first position gets contribution only from neighbors
        assert result[0] < 1.0


class TestApplyNeighborDecayEmbeddings:
    def test_exponential_decay(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float32)
        result = apply_neighbor_decay_embeddings(embeddings, decay_type="exp", param=1.0)
        assert result.shape == embeddings.shape
        # Middle embedding should borrow from neighbors
        assert result[1, 0] > 0
        assert result[1, 1] > 0

    def test_gaussian_decay(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = apply_neighbor_decay_embeddings(embeddings, decay_type="gaussian", param=1.0)
        assert result.shape == (2, 2)

    def test_power_decay(self):
        embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        result = apply_neighbor_decay_embeddings(embeddings, decay_type="power", param=1.0)
        assert result.shape == (2, 2)

    def test_empty_embeddings(self):
        embeddings = np.empty((0, 3), dtype=np.float32)
        result = apply_neighbor_decay_embeddings(embeddings)
        assert result.shape == (0, 3)

    def test_rejects_1d_input(self):
        embeddings = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        with pytest.raises(ValueError, match="must be"):
            apply_neighbor_decay_embeddings(embeddings)


class TestBlendEmbeddings:
    def test_basic_blending(self):
        chunk = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        decayed = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        doc = np.array([0.3, 0.7], dtype=np.float32)

        result = blend_embeddings(chunk, decayed, doc, alpha=1.0, beta=1.0, gamma=1.0)
        expected = chunk + decayed + doc[None, :]
        assert np.allclose(result, expected)

    def test_alpha_only(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[3.0, 4.0]], dtype=np.float32)
        doc = np.array([5.0, 6.0], dtype=np.float32)

        result = blend_embeddings(chunk, decayed, doc, alpha=2.0, beta=0.0, gamma=0.0)
        expected = 2.0 * chunk
        assert np.allclose(result, expected)

    def test_beta_only(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[3.0, 4.0]], dtype=np.float32)
        doc = np.array([5.0, 6.0], dtype=np.float32)

        result = blend_embeddings(chunk, decayed, doc, alpha=0.0, beta=2.0, gamma=0.0)
        expected = 2.0 * decayed
        assert np.allclose(result, expected)

    def test_gamma_only(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[3.0, 4.0]], dtype=np.float32)
        doc = np.array([5.0, 6.0], dtype=np.float32)

        result = blend_embeddings(chunk, decayed, doc, alpha=0.0, beta=0.0, gamma=2.0)
        expected = 2.0 * doc[None, :]
        assert np.allclose(result, expected)

    def test_l2_normalize(self):
        chunk = np.array([[3.0, 4.0]], dtype=np.float32)
        decayed = np.array([[3.0, 4.0]], dtype=np.float32)
        doc = np.array([3.0, 4.0], dtype=np.float32)

        result = blend_embeddings(chunk, decayed, doc, l2_normalize=True)
        norms = np.linalg.norm(result, axis=1)
        assert np.allclose(norms, 1.0)

    def test_shape_mismatch_raises(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        doc = np.array([1.0, 2.0], dtype=np.float32)

        with pytest.raises(ValueError, match="same shape"):
            blend_embeddings(chunk, decayed, doc)

    def test_doc_dimension_mismatch_raises(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[1.0, 2.0]], dtype=np.float32)
        doc = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            blend_embeddings(chunk, decayed, doc)

    def test_doc_must_be_1d(self):
        chunk = np.array([[1.0, 2.0]], dtype=np.float32)
        decayed = np.array([[1.0, 2.0]], dtype=np.float32)
        doc = np.array([[1.0, 2.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="must be 1"):
            blend_embeddings(chunk, decayed, doc)
