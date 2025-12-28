"""Hybrid retrieval baseline - combines BM25 and dense retrieval."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np

from .naive_retrieval import NaiveRetriever
from .bm25_retrieval import BM25Retriever


def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
) -> List[tuple]:
    """Combine multiple rankings using Reciprocal Rank Fusion.

    Parameters
    ----------
    rankings : List[List[int]]
        List of rankings, each ranking is a list of indices
    k : int
        RRF parameter (default 60)

    Returns
    -------
    List[tuple]
        List of (index, score) tuples sorted by fused score
    """
    scores: Dict[int, float] = {}

    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            if idx not in scores:
                scores[idx] = 0.0
            scores[idx] += 1.0 / (k + rank + 1)

    # Sort by score descending
    sorted_items = sorted(scores.items(), key=lambda x: -x[1])
    return sorted_items


class HybridRetriever:
    """Hybrid retrieval combining BM25 and dense (naive) retrieval.

    Uses Reciprocal Rank Fusion to combine sparse and dense results.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        self.model = model
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.dense_retriever = NaiveRetriever(model=model)
        self.sparse_retriever = BM25Retriever()
        self.chunks: List[dict] = []

    def index_chunks(self, chunks: List[dict]) -> None:
        """Build both dense and sparse indices.

        Parameters
        ----------
        chunks : List[dict]
            List of chunk dictionaries
        """
        self.chunks = chunks
        self.dense_retriever.index_chunks(chunks)
        self.sparse_retriever.index_chunks(chunks)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        fusion_method: str = "rrf",
    ) -> List[dict]:
        """Retrieve using hybrid approach.

        Parameters
        ----------
        query : str
            Query string
        top_k : int
            Number of chunks to retrieve
        fusion_method : str
            "rrf" for Reciprocal Rank Fusion, "weighted" for weighted sum

        Returns
        -------
        List[dict]
            Retrieved chunks with fused scores
        """
        if not self.chunks:
            return []

        # Get more candidates from each method
        n_candidates = min(top_k * 3, len(self.chunks))

        dense_results = self.dense_retriever.retrieve(query, top_k=n_candidates)
        sparse_results = self.sparse_retriever.retrieve(query, top_k=n_candidates)

        if fusion_method == "rrf":
            return self._fuse_rrf(dense_results, sparse_results, top_k)
        else:
            return self._fuse_weighted(dense_results, sparse_results, top_k)

    def _fuse_rrf(
        self,
        dense_results: List[dict],
        sparse_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        """Fuse using Reciprocal Rank Fusion."""
        # Build position-to-index maps
        dense_ranking = [r.get("position", i) for i, r in enumerate(dense_results)]
        sparse_ranking = [r.get("position", i) for i, r in enumerate(sparse_results)]

        # Create chunk lookup by position
        chunk_lookup = {c.get("position", i): c for i, c in enumerate(self.chunks)}

        # RRF fusion
        fused = reciprocal_rank_fusion([dense_ranking, sparse_ranking])

        results = []
        for pos, score in fused[:top_k]:
            if pos in chunk_lookup:
                chunk = dict(chunk_lookup[pos])
                chunk["score"] = score
                chunk["fusion_method"] = "rrf"
                results.append(chunk)

        return results

    def _fuse_weighted(
        self,
        dense_results: List[dict],
        sparse_results: List[dict],
        top_k: int,
    ) -> List[dict]:
        """Fuse using weighted score combination."""
        # Normalize scores
        dense_scores = {r.get("position", i): r.get("score", 0)
                       for i, r in enumerate(dense_results)}
        sparse_scores = {r.get("position", i): r.get("score", 0)
                        for i, r in enumerate(sparse_results)}

        # Normalize each to [0, 1]
        if dense_scores:
            max_dense = max(dense_scores.values())
            if max_dense > 0:
                dense_scores = {k: v / max_dense for k, v in dense_scores.items()}

        if sparse_scores:
            max_sparse = max(sparse_scores.values())
            if max_sparse > 0:
                sparse_scores = {k: v / max_sparse for k, v in sparse_scores.items()}

        # Combine scores
        all_positions = set(dense_scores.keys()) | set(sparse_scores.keys())
        combined_scores = {}

        for pos in all_positions:
            d_score = dense_scores.get(pos, 0)
            s_score = sparse_scores.get(pos, 0)
            combined_scores[pos] = (
                self.dense_weight * d_score + self.sparse_weight * s_score
            )

        # Sort and return top-k
        sorted_positions = sorted(combined_scores.items(), key=lambda x: -x[1])

        chunk_lookup = {c.get("position", i): c for i, c in enumerate(self.chunks)}
        results = []

        for pos, score in sorted_positions[:top_k]:
            if pos in chunk_lookup:
                chunk = dict(chunk_lookup[pos])
                chunk["score"] = score
                chunk["fusion_method"] = "weighted"
                results.append(chunk)

        return results

    def save(self, path: str) -> None:
        """Save indices."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.dense_retriever.save(str(path / "dense"))
        self.sparse_retriever.save(str(path / "sparse"))

    def load(self, path: str) -> None:
        """Load indices."""
        path = Path(path)

        self.dense_retriever.load(str(path / "dense"))
        self.sparse_retriever.load(str(path / "sparse"))
        self.chunks = self.dense_retriever.chunks


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    chunks = [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 1, "text": "Guido van Rossum created Python."},
        {"doc_id": "test", "position": 2, "text": "Machine learning uses neural networks."},
        {"doc_id": "test", "position": 3, "text": "Deep learning is a subset of ML."},
    ]

    retriever = HybridRetriever()
    retriever.index_chunks(chunks)

    # Test RRF fusion
    print("Hybrid (RRF) for 'Python programming language':")
    results = retriever.retrieve("Python programming language", top_k=2, fusion_method="rrf")
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:40]}")

    # Test weighted fusion
    print("\nHybrid (Weighted) for 'Who made Python':")
    results = retriever.retrieve("Who made Python", top_k=2, fusion_method="weighted")
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:40]}")
