"""embedding_pooling.py — Helpers for hierarchical RAG
====================================================
This module now supports:
* TF‑IDF weighting per chunk
* Global document pooling
* Distance‑decay smoothing for *scores* **and** for *embeddings*
* Linear blending of chunk, neighbour‑smoothed, and document embeddings
"""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple, Union, Literal

# -----------------------------------------------------------------------------
# Optional dependency — only needed for TF‑IDF weighting
# -----------------------------------------------------------------------------
try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
except ImportError:  # pragma: no cover
    TfidfVectorizer = None  # allows importing the file even if sklearn is absent.

# Public API re‑exported when using `from embedding_pooling import *`
__all__ = [
    "compute_tfidf_weights",
    "compute_global_embedding",
    "apply_neighbor_decay_scores",
    "apply_neighbor_decay_embeddings",
    "blend_embeddings",
]

# -----------------------------------------------------------------------------
# 1. TF‑IDF weighting per chunk
# -----------------------------------------------------------------------------

def compute_tfidf_weights(
    texts: List[str],
    *,
    vectorizer: Optional["TfidfVectorizer"] = None,
    norm: Literal["l1", "l2", None] = "l2",
) -> Tuple[np.ndarray, "TfidfVectorizer"]:
    """Compute a scalar TF‑IDF weight for each chunk string in *texts*.

    The weight is simply the sum of the TF‑IDF scores of that chunk’s tokens.
    A fitted vectorizer is returned so the same vocabulary can be reused on the
    next document.
    """
    if TfidfVectorizer is None:
        raise ImportError(
            "scikit‑learn is required for TF‑IDF weighting. Install it via `pip install scikit-learn`."
        )

    if vectorizer is None:
        vectorizer = TfidfVectorizer(norm=norm)
        tfidf = vectorizer.fit_transform(texts)
    else:
        tfidf = vectorizer.transform(texts)

    weights = tfidf.sum(axis=1).A1.astype(np.float32)
    return weights, vectorizer

# -----------------------------------------------------------------------------
# 2. Global document‑level pooling
# -----------------------------------------------------------------------------

def compute_global_embedding(
    embeddings: np.ndarray,
    method: Literal["mean", "min", "max", "minmax", "tfidf"] = "mean",
    *,
    weights: Optional[np.ndarray] = None,
    l2_normalize: bool = False,
) -> np.ndarray:
    """Aggregate chunk embeddings into a single document embedding."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (n_chunks, dim)")

    n_chunks, dim = embeddings.shape

    if method == "mean":
        vec = embeddings.mean(axis=0)
    elif method == "min":
        vec = embeddings.min(axis=0)
    elif method == "max":
        vec = embeddings.max(axis=0)
    elif method == "minmax":
        vec = np.concatenate((embeddings.min(axis=0), embeddings.max(axis=0)))
    elif method == "tfidf":
        if weights is None:
            raise ValueError("`weights` must be provided when method='tfidf'")
        if len(weights) != n_chunks:
            raise ValueError("weights length must equal n_chunks")
        w = np.asarray(weights, dtype=np.float32)
        w_sum = w.sum()
        if w_sum == 0:
            raise ValueError("Sum of TF‑IDF weights is zero; check your input.")
        vec = (w[:, None] * embeddings).sum(axis=0) / w_sum
    else:
        raise ValueError(f"Unknown pooling method: {method}")

    if l2_normalize:
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

    return vec.astype(np.float32)

# -----------------------------------------------------------------------------
# 3. Distance‑decay kernels (shared helpers)
# -----------------------------------------------------------------------------

def _auto_decay_param(n_chunks: int, decay_type: str) -> float:
    """Heuristic parameter if the user supplies *None*."""
    if decay_type == "exp":
        return max(1.0, n_chunks / 10)  # τ
    if decay_type == "gaussian":
        return max(1.0, n_chunks / 10)  # σ
    if decay_type == "power":
        return 1.0  # α
    raise ValueError(f"Unsupported decay_type: {decay_type}")


def _build_decay_weights(
    n: int,
    *,
    decay_type: Literal["exp", "gaussian", "power"],
    param: float,
    normalize: bool,
    include_self: bool,
) -> np.ndarray:
    """Return an (n, n) matrix of neighbour weights."""
    idx = np.arange(n, dtype=np.float32)
    dist = np.abs(idx[:, None] - idx[None, :])

    if decay_type == "exp":
        tau = param
        w = np.exp(-dist / tau)
    elif decay_type == "gaussian":
        sigma = param
        w = np.exp(-(dist ** 2) / (2.0 * sigma ** 2))
    elif decay_type == "power":
        alpha = param
        w = 1.0 / (1.0 + dist) ** alpha
    else:
        raise ValueError(decay_type)

    if not include_self:
        np.fill_diagonal(w, 0.0)
    if normalize:
        w /= w.sum(axis=1, keepdims=True) + 1e-12
    return w.astype(np.float32)

# -----------------------------------------------------------------------------
# 3a. Apply decay to *scores* (legacy helper)
# -----------------------------------------------------------------------------

def apply_neighbor_decay_scores(
    scores: np.ndarray,
    *,
    decay_type: Literal["exp", "gaussian", "power"] = "exp",
    param: Optional[float] = None,
    normalize: bool = True,
    include_self: bool = True,
) -> np.ndarray:
    """Smooth scalar similarity scores via distance‑decay."""
    if scores.ndim != 1:
        raise ValueError("scores must be 1‑D (n_chunks,)")
    n = len(scores)
    if n == 0:
        return scores
    if param is None:
        param = _auto_decay_param(n, decay_type)
    w = _build_decay_weights(n, decay_type=decay_type, param=param, normalize=normalize, include_self=include_self)
    return (w @ scores).astype(np.float32)

# -----------------------------------------------------------------------------
# 3b. Apply decay directly to *embeddings*
# -----------------------------------------------------------------------------

def apply_neighbor_decay_embeddings(
    embeddings: np.ndarray,
    *,
    decay_type: Literal["exp", "gaussian", "power"] = "exp",
    param: Optional[float] = None,
    normalize: bool = True,
    include_self: bool = True,
) -> np.ndarray:
    """Return neighbour‑smoothed embeddings.

    Each output row is a weighted sum of its own embedding plus decayed
    contributions from every other chunk.
    """
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be (n_chunks, dim)")
    n, dim = embeddings.shape
    if n == 0:
        return embeddings
    if param is None:
        param = _auto_decay_param(n, decay_type)
    w = _build_decay_weights(n, decay_type=decay_type, param=param, normalize=normalize, include_self=include_self)
    return (w @ embeddings).astype(np.float32)

# -----------------------------------------------------------------------------
# 4. Linear blending of chunk/decayed/document embeddings
# -----------------------------------------------------------------------------

def blend_embeddings(
    chunk_embeddings: np.ndarray,
    decayed_embeddings: np.ndarray,
    doc_embedding: np.ndarray,
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.0,
    l2_normalize: bool = False,
) -> np.ndarray:
    """Return a per‑chunk blended embedding.

    final_i = α·chunk_i  +  β·decayed_i  +  γ·doc_embed
    """
    if chunk_embeddings.shape != decayed_embeddings.shape:
        raise ValueError("chunk and decayed embeddings must have the same shape")
    n, dim = chunk_embeddings.shape

    if doc_embedding.ndim != 1:
        raise ValueError("doc_embedding must be 1‑D (dim,)")
    if doc_embedding.shape[0] != dim:
        raise ValueError("Dimension mismatch between chunk embeddings and doc_embedding")

    final = alpha * chunk_embeddings + beta * decayed_embeddings + gamma * doc_embedding[None, :]

    if l2_normalize:
        norms = np.linalg.norm(final, axis=1, keepdims=True)
        final = np.divide(final, norms, out=final, where=norms != 0)

    return final.astype(np.float32)

# -----------------------------------------------------------------------------
# Quick demo when run standalone
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Dummy embeddings
    chunks = rng.random((6, 4), dtype=np.float32)
    doc_vec = compute_global_embedding(chunks)
    decayed = apply_neighbor_decay_embeddings(chunks, decay_type="gaussian")
    blended = blend_embeddings(chunks, decayed, doc_vec, alpha=1.0, beta=0.5, gamma=0.2, l2_normalize=True)

    print("Chunk[0] →", np.round(chunks[0], 3))
    print("Decayed[0]→", np.round(decayed[0], 3))
    print("Blended[0]→", np.round(blended[0], 3))
