import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import decayrag.decayrag.retrieval as r
import decayrag.decayrag.ingest as ingest


def test_embed_query_mock_backend(monkeypatch):
    calls = {}

    def fake_api_embed(texts, model_name):
        calls['texts'] = texts
        calls['model'] = model_name
        return np.array([[3.0, 4.0]], dtype=np.float32)

    monkeypatch.setattr(ingest, "_api_embed", fake_api_embed)
    vec = r.embed_query("hello", "dummy-model")
    assert np.allclose(vec, np.array([0.6, 0.8], dtype=np.float32))
    assert calls['texts'] == ["hello"]
    assert calls['model'] == "dummy-model"


def test_compute_chunk_similarities_dot():
    q = np.array([1.0, 2.0], dtype=np.float32)
    chunks = np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    sims = r.compute_chunk_similarities(q, chunks)
    expected = (chunks @ q).astype(np.float32)
    assert sims.shape == (2,)
    assert np.allclose(sims, expected)


def test_blend_scores_weighted_sum():
    raw = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    decayed = np.array([0.5, 1.0, 1.5], dtype=np.float32)
    result = r.blend_scores(raw, decayed, alpha=0.2, beta=0.8)
    expected = (0.2 * raw + 0.8 * decayed).astype(np.float32)
    assert np.allclose(result, expected)


def test_top_k_chunks_indices_and_order():
    scores = np.array([0.1, 0.3, 0.2, 0.5], dtype=np.float32)
    idx = r.top_k_chunks(scores, 2)
    assert idx.tolist() == [3, 1]
