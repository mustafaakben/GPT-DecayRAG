"""Ranking metrics for retrieval evaluation."""

from __future__ import annotations

import math
from typing import List, Set, Union


def mrr(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
) -> float:
    """Compute Mean Reciprocal Rank.

    Returns 1/rank of the first relevant result.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions

    Returns
    -------
    float
        MRR score in (0, 1], or 0 if no target found
    """
    if not targets:
        return 0.0

    targets_set = set(targets)

    for rank, chunk in enumerate(retrieved, start=1):
        pos = chunk.get("position")
        if pos in targets_set:
            return 1.0 / rank

    return 0.0


def dcg_at_k(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
    k: int,
) -> float:
    """Compute Discounted Cumulative Gain at K.

    Uses binary relevance (1 if target, 0 otherwise).

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions
    k : int
        Number of results to consider

    Returns
    -------
    float
        DCG@K score
    """
    targets_set = set(targets)
    dcg = 0.0

    for i, chunk in enumerate(retrieved[:k]):
        pos = chunk.get("position")
        if pos in targets_set:
            # Binary relevance: rel = 1 for targets
            # DCG formula: rel_i / log2(i + 2)
            dcg += 1.0 / math.log2(i + 2)

    return dcg


def idcg_at_k(num_targets: int, k: int) -> float:
    """Compute Ideal DCG at K.

    Assumes all targets would be ranked at the top.

    Parameters
    ----------
    num_targets : int
        Number of target chunks
    k : int
        Number of results to consider

    Returns
    -------
    float
        Ideal DCG@K score
    """
    idcg = 0.0
    num_relevant = min(num_targets, k)

    for i in range(num_relevant):
        idcg += 1.0 / math.log2(i + 2)

    return idcg


def ndcg_at_k(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
    k: int,
) -> float:
    """Compute Normalized DCG at K.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions
    k : int
        Number of results to consider

    Returns
    -------
    float
        NDCG@K score in [0, 1]
    """
    if not targets:
        return 1.0  # No targets means perfect ranking

    targets_set = set(targets)
    dcg = dcg_at_k(retrieved, targets_set, k)
    idcg = idcg_at_k(len(targets_set), k)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def average_precision(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
) -> float:
    """Compute Average Precision.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions

    Returns
    -------
    float
        Average Precision score in [0, 1]
    """
    if not targets:
        return 0.0

    targets_set = set(targets)
    num_relevant = 0
    precision_sum = 0.0

    for i, chunk in enumerate(retrieved, start=1):
        pos = chunk.get("position")
        if pos in targets_set:
            num_relevant += 1
            precision_at_i = num_relevant / i
            precision_sum += precision_at_i

    if len(targets_set) == 0:
        return 0.0

    return precision_sum / len(targets_set)


def first_target_rank(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
) -> int:
    """Find rank of first target chunk.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions

    Returns
    -------
    int
        Rank (1-indexed) of first target, or -1 if not found
    """
    targets_set = set(targets)

    for rank, chunk in enumerate(retrieved, start=1):
        pos = chunk.get("position")
        if pos in targets_set:
            return rank

    return -1


def compute_ranking_metrics(
    retrieved: List[dict],
    target_positions: List[int],
    k_values: List[int] = [3, 5, 10, 20],
) -> dict:
    """Compute multiple ranking metrics.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks
    target_positions : List[int]
        Target chunk positions
    k_values : List[int]
        K values for NDCG

    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    metrics = {
        "mrr": mrr(retrieved, target_positions),
        "ap": average_precision(retrieved, target_positions),
        "first_rank": first_target_rank(retrieved, target_positions),
    }

    for k in k_values:
        metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved, target_positions, k)

    return metrics


if __name__ == "__main__":
    # Test metrics
    retrieved = [
        {"position": 5, "score": 0.9},   # Not target
        {"position": 2, "score": 0.8},   # Target A
        {"position": 7, "score": 0.7},   # Target B
        {"position": 1, "score": 0.6},   # Not target
        {"position": 10, "score": 0.5},  # Not target
    ]

    targets = [2, 7]

    print("Testing ranking metrics:")
    print(f"  Targets: {targets}")
    print(f"  Retrieved: {[r['position'] for r in retrieved]}")
    print()

    print(f"  MRR: {mrr(retrieved, targets):.4f}")
    print(f"  AP: {average_precision(retrieved, targets):.4f}")
    print(f"  First target rank: {first_target_rank(retrieved, targets)}")
    print()

    for k in [3, 5, 10]:
        print(f"  NDCG@{k}: {ndcg_at_k(retrieved, targets, k):.4f}")

    print()
    all_metrics = compute_ranking_metrics(retrieved, targets)
    print("All metrics:", {k: f"{v:.4f}" if isinstance(v, float) else v
                          for k, v in all_metrics.items()})
