"""Query-time retrieval utilities for DecayRAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .pooling import apply_neighbor_decay_scores

__all__ = [
    "embed_query",
    "compute_chunk_similarities",
    "blend_scores",
    "top_k_chunks",
    "retrieve",
]


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed a query string using the chosen model or a fallback."""
    texts = [query]
    try:
        from .ingest import _api_embed  # type: ignore

        embeds = _api_embed(texts, model_name)
    except Exception:
        try:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model_name)
            embeds = st_model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
        except Exception:
            rng = np.random.default_rng(0)
            embeds = rng.standard_normal((1, 384)).astype(np.float32)

    norms = np.linalg.norm(embeds, axis=1, keepdims=True)
    np.divide(embeds, norms, out=embeds, where=norms != 0)
    return embeds[0]


def compute_chunk_similarities(query_vec: np.ndarray, chunk_embeds: np.ndarray) -> np.ndarray:
    """Return the dot-product similarity between query_vec and each chunk embedding."""
    if query_vec.ndim != 1:
        query_vec = query_vec.reshape(-1)
    return (chunk_embeds @ query_vec).astype(np.float32)


def blend_scores(raw: np.ndarray, decayed: np.ndarray, *, alpha: float = 0.5, beta: float = 0.5) -> np.ndarray:
    """Blend raw and neighbour-decayed scores."""
    if raw.shape != decayed.shape:
        raise ValueError("Score arrays must have the same shape")
    return (alpha * raw + beta * decayed).astype(np.float32)


def top_k_chunks(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the top-k scores in descending order."""
    if k <= 0 or len(scores) == 0:
        return np.array([], dtype=int)
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]
    return idx.astype(int)


def _load_index(index_path: str) -> faiss.Index:
    return faiss.read_index(str(Path(index_path)))


def _load_metadata(meta_path: str) -> List[dict]:
    with open(meta_path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def retrieve(
    query: str,
    index_path: str,
    *,
    model: str = "text-embedding-3-small",
    top_k: int = 5,
    decay: bool = True,
    blend: bool = True,
) -> List[dict]:
    """Search *index_path* for chunks relevant to *query*."""
    index = _load_index(index_path)
    meta = _load_metadata(index_path + ".meta")
    if index.ntotal != len(meta):
        raise ValueError("Metadata size does not match index")

    # reconstruct embeddings from the underlying flat index
    if hasattr(index, "index") and hasattr(index.index, "reconstruct_n"):
        embeds = index.index.reconstruct_n(0, index.ntotal)
    elif hasattr(index, "reconstruct_n"):
        embeds = index.reconstruct_n(0, index.ntotal)
    else:
        raise ValueError("Index type does not support reconstruct")

    qvec = embed_query(query, model)
    raw_scores = compute_chunk_similarities(qvec, embeds)

    final_scores = raw_scores
    if decay:
        decayed = apply_neighbor_decay_scores(raw_scores)
        final_scores = decayed
        if blend:
            final_scores = blend_scores(raw_scores, decayed)

    top_idx = top_k_chunks(final_scores, top_k)
    results = []
    for i in top_idx:
        item = dict(meta[i])
        item["score"] = float(final_scores[i])
        results.append(item)
    return results
