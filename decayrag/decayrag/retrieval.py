"""Query-time retrieval utilities for DecayRAG."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import faiss
import numpy as np

from .pooling import (
    apply_neighbor_decay_scores,
    apply_neighbor_decay_embeddings,
    compute_global_embedding,
    blend_embeddings,
)
from . import ingest
from .ingest import load_config

__all__ = [
    "load_config",
    "embed_query",
    "compute_chunk_similarities",
    "blend_scores",
    "top_k_chunks",
    "retrieve",
]


def embed_query(query: str, model_name: str) -> np.ndarray:
    """Embed a query string using the OpenAI embeddings API."""
    embeds = ingest._api_embed([query], model_name)
    vec = embeds[0]
    norm = np.linalg.norm(vec)
    if norm != 0:
        vec = vec / norm
    return vec


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


def _load_embeddings(index: faiss.Index, index_path: str) -> np.ndarray | None:
    embed_file = Path(index_path + ".npy")
    if embed_file.exists():
        embeds = np.load(embed_file)
        if embeds.shape[0] != index.ntotal:
            raise ValueError("Embedding store size does not match index")
        return embeds
    if hasattr(index, "index") and hasattr(index.index, "reconstruct_n"):
        return index.index.reconstruct_n(0, index.ntotal)
    if hasattr(index, "reconstruct_n"):
        return index.reconstruct_n(0, index.ntotal)
    return None


def retrieve(
    query: str,
    index_path: str,
    *,
    model: str = "text-embedding-3-small",
    top_k: int = 5,
    decay: bool = True,
    blend: bool = True,
    embedding_blend: bool = False,
) -> List[dict]:
    """Search *index_path* for chunks relevant to *query*."""
    index = _load_index(index_path)
    meta = _load_metadata(index_path + ".meta")
    if index.ntotal != len(meta):
        raise ValueError("Metadata size does not match index")

    embeds = _load_embeddings(index, index_path)
    if embedding_blend and embeds is None:
        raise ValueError(
            "Embedding blending requires stored embeddings; re-run ingestion to save "
            f"{index_path}.npy or disable embedding_blend."
        )

    qvec = embed_query(query, model)

    # Embedding-level pathway
    if embedding_blend and embeds is not None:
        final_embeds = np.empty_like(embeds)

        # group by document id
        doc_to_indices: dict[str, List[int]] = {}
        for idx, m in enumerate(meta):
            doc_to_indices.setdefault(m.get("doc_id", ""), []).append(idx)

        for doc_id, indices in doc_to_indices.items():
            doc_vectors = embeds[indices]
            decayed = (
                apply_neighbor_decay_embeddings(doc_vectors)
                if decay
                else doc_vectors
            )
            doc_embed = compute_global_embedding(doc_vectors)
            if blend:
                blended = blend_embeddings(doc_vectors, decayed, doc_embed)
            else:
                blended = decayed
            final_embeds[indices] = blended

        final_scores = compute_chunk_similarities(qvec, final_embeds)
        top_idx = top_k_chunks(final_scores, top_k)
        results = []
        for i in top_idx:
            item = dict(meta[i])
            item["score"] = float(final_scores[i])
            results.append(item)
        return results

    if embeds is not None:
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

    qvec = qvec.astype(np.float32, copy=False).reshape(1, -1)
    scores, ids = index.search(qvec, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        item = dict(meta[int(idx)])
        item["score"] = float(score)
        results.append(item)
    return results
