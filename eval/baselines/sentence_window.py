"""Sentence Window retrieval baseline - expand context at retrieval time."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Dict

import faiss
import numpy as np

from decayrag import embed_query
from decayrag.decayrag.ingest import _api_embed, _api_embed_async


class SentenceWindowRetriever:
    """Sentence Window retrieval baseline.

    This method retrieves chunks based on similarity, then expands
    the context by including neighboring chunks at retrieval time.
    Similar to LlamaIndex's SentenceWindowRetriever.

    Key difference from DecayRAG: expansion happens POST-retrieval,
    not during scoring. Neighbors don't influence which chunks are selected.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        window_size: int = 1,
    ):
        self.model = model
        self.window_size = window_size
        self.index: Optional[faiss.Index] = None
        self.chunks: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.doc_chunks: Dict[str, List[dict]] = {}  # doc_id -> sorted chunks

    def index_chunks(self, chunks: List[dict]) -> None:
        """Build index from chunks.

        Parameters
        ----------
        chunks : List[dict]
            List of chunk dictionaries
        """
        if not chunks:
            return

        self.chunks = chunks

        # Organize chunks by document
        self.doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            if doc_id not in self.doc_chunks:
                self.doc_chunks[doc_id] = []
            self.doc_chunks[doc_id].append(chunk)

        # Sort by position within each document
        for doc_id in self.doc_chunks:
            self.doc_chunks[doc_id].sort(key=lambda x: x.get("position", 0))

        # Embed chunks
        texts = [c.get("text", "") for c in chunks]
        self.embeddings = _api_embed(texts, self.model)

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        np.divide(self.embeddings, norms, out=self.embeddings, where=norms != 0)

        # Build index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    async def index_chunks_async(
        self,
        chunks: List[dict],
        concurrency_limit: int = 50,
    ) -> None:
        """Async version of index_chunks.

        Parameters
        ----------
        chunks : List[dict]
            List of chunk dictionaries
        concurrency_limit : int
            Maximum concurrent embedding requests
        """
        if not chunks:
            return

        self.chunks = chunks

        # Organize chunks by document
        self.doc_chunks = {}
        for chunk in chunks:
            doc_id = chunk.get("doc_id", "")
            if doc_id not in self.doc_chunks:
                self.doc_chunks[doc_id] = []
            self.doc_chunks[doc_id].append(chunk)

        # Sort by position within each document
        for doc_id in self.doc_chunks:
            self.doc_chunks[doc_id].sort(key=lambda x: x.get("position", 0))

        # Embed chunks asynchronously
        texts = [c.get("text", "") for c in chunks]
        self.embeddings = await _api_embed_async(
            texts, self.model, concurrency_limit=concurrency_limit
        )

        # Normalize
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        np.divide(self.embeddings, norms, out=self.embeddings, where=norms != 0)

        # Build index
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))

    def _expand_chunk(self, chunk: dict) -> List[dict]:
        """Expand a chunk to include its neighbors."""
        doc_id = chunk.get("doc_id", "")
        position = chunk.get("position", 0)

        if doc_id not in self.doc_chunks:
            return [chunk]

        doc_chunk_list = self.doc_chunks[doc_id]

        # Find chunks within window
        expanded = []
        for c in doc_chunk_list:
            c_pos = c.get("position", 0)
            if abs(c_pos - position) <= self.window_size:
                expanded.append(c)

        # Sort by position
        expanded.sort(key=lambda x: x.get("position", 0))
        return expanded

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        expand: bool = True,
    ) -> List[dict]:
        """Retrieve top-k chunks with optional window expansion.

        Parameters
        ----------
        query : str
            Query string
        top_k : int
            Number of base chunks to retrieve
        expand : bool
            Whether to expand with neighboring chunks

        Returns
        -------
        List[dict]
            Retrieved chunks (possibly expanded)
        """
        if self.index is None or not self.chunks:
            return []

        # Embed query
        query_vec = embed_query(query, self.model)
        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # Search for base chunks
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_vec, k)

        # Get base results
        base_results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(score)
            chunk["is_expanded"] = False
            base_results.append(chunk)

        if not expand:
            return base_results

        # Expand with neighbors (for context assembly, not retrieval scoring)
        # Note: We return base results for evaluation, but provide expanded context
        seen_positions = set()
        expanded_results = []

        for chunk in base_results:
            doc_id = chunk.get("doc_id", "")
            pos = chunk.get("position", 0)
            key = (doc_id, pos)

            if key in seen_positions:
                continue
            seen_positions.add(key)

            # Add base chunk
            expanded_results.append(chunk)

            # Add neighbors (marked as expanded)
            for neighbor in self._expand_chunk(chunk):
                n_pos = neighbor.get("position", 0)
                n_key = (doc_id, n_pos)
                if n_key not in seen_positions:
                    seen_positions.add(n_key)
                    neighbor_copy = dict(neighbor)
                    neighbor_copy["score"] = chunk["score"] * 0.9  # Slightly lower score
                    neighbor_copy["is_expanded"] = True
                    expanded_results.append(neighbor_copy)

        # Sort by doc_id and position
        expanded_results.sort(key=lambda x: (x.get("doc_id", ""), x.get("position", 0)))

        return expanded_results

    def retrieve_base_only(self, query: str, top_k: int = 10) -> List[dict]:
        """Retrieve without expansion - for fair comparison."""
        return self.retrieve(query, top_k=top_k, expand=False)

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

        # Rebuild doc_chunks
        self.doc_chunks = {}
        for chunk in self.chunks:
            doc_id = chunk.get("doc_id", "")
            if doc_id not in self.doc_chunks:
                self.doc_chunks[doc_id] = []
            self.doc_chunks[doc_id].append(chunk)
        for doc_id in self.doc_chunks:
            self.doc_chunks[doc_id].sort(key=lambda x: x.get("position", 0))


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    chunks = [
        {"doc_id": "test", "position": 0, "text": "Introduction to Python."},
        {"doc_id": "test", "position": 1, "text": "Python is a programming language."},
        {"doc_id": "test", "position": 2, "text": "It was created by Guido van Rossum."},
        {"doc_id": "test", "position": 3, "text": "Python is used in data science."},
        {"doc_id": "test", "position": 4, "text": "Machine learning is popular."},
    ]

    retriever = SentenceWindowRetriever(window_size=1)
    retriever.index_chunks(chunks)

    print("Base retrieval (no expansion):")
    results = retriever.retrieve_base_only("What is Python?", top_k=2)
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:40]}")

    print("\nWith window expansion:")
    results = retriever.retrieve("What is Python?", top_k=2, expand=True)
    for r in results:
        exp = " (expanded)" if r.get("is_expanded") else ""
        print(f"  [{r['position']}] {r['score']:.4f}{exp}: {r['text'][:40]}")
