"""BM25 sparse retrieval baseline."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> List[str]:
    """Simple tokenization for BM25."""
    # Lowercase and split on non-alphanumeric
    text = text.lower()
    tokens = re.findall(r'\b[a-z0-9]+\b', text)
    return tokens


class BM25Retriever:
    """BM25 sparse retrieval baseline.

    Uses TF-IDF style keyword matching without semantic understanding.
    Good baseline for testing if dense retrieval adds value.
    """

    def __init__(self):
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[dict] = []
        self.tokenized_corpus: List[List[str]] = []

    def index_chunks(self, chunks: List[dict]) -> None:
        """Build BM25 index from chunks.

        Parameters
        ----------
        chunks : List[dict]
            List of chunk dictionaries with 'text' key
        """
        if not chunks:
            return

        self.chunks = chunks

        # Tokenize corpus
        self.tokenized_corpus = [
            _tokenize(c.get("text", "")) for c in chunks
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)

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
        if self.bm25 is None or not self.chunks:
            return []

        # Tokenize query
        query_tokens = _tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        k = min(top_k, len(self.chunks))
        top_indices = scores.argsort()[-k:][::-1]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)

        return results

    def save(self, path: str) -> None:
        """Save chunks (BM25 is rebuilt on load)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "chunks.json", "w") as f:
            json.dump(self.chunks, f)

    def load(self, path: str) -> None:
        """Load chunks and rebuild BM25."""
        path = Path(path)

        with open(path / "chunks.json", "r") as f:
            chunks = json.load(f)

        self.index_chunks(chunks)


if __name__ == "__main__":
    # Test
    chunks = [
        {"doc_id": "test", "position": 0, "text": "Python is a programming language created by Guido."},
        {"doc_id": "test", "position": 1, "text": "Machine learning algorithms process data."},
        {"doc_id": "test", "position": 2, "text": "Deep learning neural networks learn patterns."},
        {"doc_id": "test", "position": 3, "text": "Guido van Rossum created Python in the 1990s."},
    ]

    retriever = BM25Retriever()
    retriever.index_chunks(chunks)

    # Test with keyword-matching query
    results = retriever.retrieve("Who created Python programming?", top_k=2)
    print("BM25 Results for 'Who created Python programming?':")
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:50]}")

    # Test with semantic query (BM25 should struggle)
    results = retriever.retrieve("What is artificial intelligence?", top_k=2)
    print("\nBM25 Results for 'What is artificial intelligence?':")
    for r in results:
        print(f"  [{r['position']}] {r['score']:.4f}: {r['text'][:50]}")
