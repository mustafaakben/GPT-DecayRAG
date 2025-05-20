import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag import upsert_embeddings, retrieve
import decayrag.decayrag.retrieval as r


def test_retrieve_basic(tmp_path, monkeypatch):
    index = tmp_path / "index.faiss"
    chunks = [
        {"doc_id": "doc1", "chunk_id": 0, "text": "apple", "level": [], "position": 0},
        {"doc_id": "doc2", "chunk_id": 0, "text": "banana", "level": [], "position": 0},
    ]
    embeds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    upsert_embeddings(str(index), chunks, embeds)

    def fake_embed_query(query: str, model_name: str = "") -> np.ndarray:
        if "apple" in query.lower():
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(r, "embed_query", fake_embed_query)

    results = retrieve("apple", str(index), top_k=1, decay=False, blend=False)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"

    results = retrieve("banana", str(index), top_k=1, decay=False, blend=False)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc2"
