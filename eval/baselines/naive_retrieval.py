"""Naive chunking retrieval baseline - no decay, no blending."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np

from decayrag import embed_chunks, embed_query
from decayrag.decayrag.ingest import _api_embed


class NaiveRetriever:
    """Naive retrieval without decay or blending.

    This baseline embeds chunks independently and retrieves
    based on pure cosine similarity without any neighbor awareness.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
    ):
        self.model = model
        self.index: Optional[faiss.Index] = None
        self.chunks: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    def index_chunks(self, chunks: List[dict]) -> None:
        """Build index from chunks.

        Parameters
        ----------
        chunks : List[dict]
            List of chunk dictionaries with 'text', 'position', 'doc_id' keys
        """
        if not chunks:
            return

        self.chunks = chunks

        # Embed chunks
        texts = [c.get("text", "") for c in chunks]
        self.embeddings = _api_embed(texts, self.model)

        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        np.divide(self.embeddings, norms, out=self.embeddings, where=norms != 0)

        # Build FAISS index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def retrieve(self, query: str, top_k: int = 10) -> List[dict]:
        """Retrieve top-k chunks for a query.

        Parameters
        ----------
        query : str
            Query string
        top_k : int
            Number of chunks to retrieve

        Returns
        -------
        List[dict]
            Retrieved chunks with scores
        """
        if self.index is None or not self.chunks:
            return []

        # Embed query
        query_vec = embed_query(query, self.model)
        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # Search
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            results.append(chunk)

        return results

    def save(self, path: str) -> None:
        """Save index and metadata."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(path / "index.faiss"))
        np.save(path / "embeddings.npy", self.embeddings)
        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)

    def load(self, path: str) -> None:
        """Load index and metadata."""
        path = Path(path)

        self.index = faiss.read_index(str(path / "index.faiss"))
        self.embeddings = np.load(path / "embeddings.npy")
        with open(path / "chunks.json", "r") as f:
            self.chunks = json.load(f)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    # Test
    chunks = [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 1, "text": "Machine learning uses algorithms."},
        {"doc_id": "test", "position": 2, "text": "Deep learning is a subset of ML."},
    ]

    retriever = NaiveRetriever()
    retriever.index_chunks(chunks)

    results = retriever.retrieve("What is deep learning?", top_k=2)
    print("Results:")
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:50]}")
