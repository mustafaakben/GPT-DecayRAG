"""Query-time retrieval helpers for DecayRAG."""

from __future__ import annotations

import numpy as np

__all__ = ["top_k_chunks"]


def top_k_chunks(scores: np.ndarray, k: int) -> np.ndarray:
    """Return indices of the ``k`` highest *scores* in descending order.

    Parameters
    ----------
    scores : np.ndarray
        1-D array of similarity scores.
    k : int
        Number of top entries to return.

    Returns
    -------
    np.ndarray
        Indices of the top ``k`` scores sorted from highest to lowest.
    """
    if scores.ndim != 1:
        raise ValueError("scores must be 1-D")
    if k <= 0:
        return np.array([], dtype=int)

    k = min(k, len(scores))
    # argsort on the negated scores gives descending order. Using a stable sort
    # ensures that ties are resolved by the original index order.
    sorted_idx = np.argsort(-scores, kind="stable")
    return sorted_idx[:k]
