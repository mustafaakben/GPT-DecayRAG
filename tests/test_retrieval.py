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

    def fake_api_embed(texts, model_name):
        return [
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if "apple" in texts[0].lower()
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ]

    monkeypatch.setattr(r.ingest, "_api_embed", fake_api_embed)

    results = retrieve("apple", str(index), top_k=1, decay=False, blend=False)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"

    results = retrieve("banana", str(index), top_k=1, decay=False, blend=False)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc2"


def test_retrieve_embedding_blend(tmp_path, monkeypatch):
    index = tmp_path / "index.faiss"
    chunks = [
        {"doc_id": "doc1", "chunk_id": 0, "text": "apple", "level": [], "position": 0},
        {"doc_id": "doc2", "chunk_id": 0, "text": "banana", "level": [], "position": 0},
    ]
    embeds = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    upsert_embeddings(str(index), chunks, embeds)

    def fake_api_embed(texts, model_name):
        return [
            np.array([1.0, 0.0, 0.0], dtype=np.float32)
            if "apple" in texts[0].lower()
            else np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ]

    calls = {}

    def fake_compute_global_embedding(vecs, *a, **kw):
        calls["global"] = calls.get("global", 0) + 1
        return np.zeros(vecs.shape[1], dtype=np.float32)

    def fake_apply_neighbor_decay_embeddings(vecs, *a, **kw):
        calls["decay"] = calls.get("decay", 0) + 1
        return vecs

    def fake_blend_embeddings(c, d, doc, *a, **kw):
        calls["blend"] = calls.get("blend", 0) + 1
        return c

    monkeypatch.setattr(r.ingest, "_api_embed", fake_api_embed)
    monkeypatch.setattr(r, "compute_global_embedding", fake_compute_global_embedding)
    monkeypatch.setattr(r, "apply_neighbor_decay_embeddings", fake_apply_neighbor_decay_embeddings)
    monkeypatch.setattr(r, "blend_embeddings", fake_blend_embeddings)

    results = retrieve("apple", str(index), top_k=1, embedding_blend=True)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc1"
    assert calls.get("global", 0) > 0
    assert calls.get("decay", 0) > 0
    assert calls.get("blend", 0) > 0


def test_retrieve_score_decay_and_blend(tmp_path, monkeypatch):
    index = tmp_path / "index.faiss"
    chunks = [
        {"doc_id": "doc1", "chunk_id": 0, "text": "alpha", "level": [], "position": 0},
        {"doc_id": "doc2", "chunk_id": 0, "text": "beta", "level": [], "position": 0},
        {"doc_id": "doc3", "chunk_id": 0, "text": "gamma", "level": [], "position": 0},
    ]
    embeds = np.array(
        [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
        dtype=np.float32,
    )
    upsert_embeddings(str(index), chunks, embeds)

    def fake_api_embed(texts, model_name):
        return [np.array([1.0, 0.0], dtype=np.float32)]

    calls = {}

    def fake_apply_neighbor_decay_scores(scores):
        calls["decay"] = scores.copy()
        return scores + np.array([0.0, 0.2, 0.1], dtype=np.float32)

    def fake_blend_scores(raw, decayed, alpha=0.5, beta=0.5):
        calls["blend"] = True
        return (alpha * raw + beta * decayed).astype(np.float32)

    monkeypatch.setattr(r.ingest, "_api_embed", fake_api_embed)
    monkeypatch.setattr(r, "apply_neighbor_decay_scores", fake_apply_neighbor_decay_scores)
    monkeypatch.setattr(r, "blend_scores", fake_blend_scores)

    results = retrieve("alpha", str(index), top_k=2, decay=True, blend=True)
    assert [r["doc_id"] for r in results] == ["doc1", "doc3"]
    assert "decay" in calls
    assert calls.get("blend") is True
