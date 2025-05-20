import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decayrag.decayrag.retrieval import top_k_chunks


def test_top_k_basic() -> None:
    scores = np.array([0.1, 0.5, 0.3, 0.9], dtype=np.float32)
    idx = top_k_chunks(scores, 2)
    assert list(idx) == [3, 1]


def test_top_k_ties() -> None:
    scores = np.array([0.8, 0.8, 0.5], dtype=np.float32)
    idx = top_k_chunks(scores, 2)
    assert list(idx) == [0, 1]


def test_top_k_overflow() -> None:
    scores = np.array([0.2, 0.1], dtype=np.float32)
    idx = top_k_chunks(scores, 5)
    assert list(idx) == [0, 1]
