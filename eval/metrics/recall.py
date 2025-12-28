"""Recall metrics for retrieval evaluation."""

from __future__ import annotations

from typing import List, Set, Union


def recall_at_k(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
    k: int,
) -> float:
    """Compute Recall@K.

    Measures the fraction of target chunks retrieved in the top-K results.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions
    k : int
        Number of top results to consider

    Returns
    -------
    float
        Recall@K score in [0, 1]
    """
    if not targets:
        return 1.0  # No targets means perfect recall (vacuously true)

    targets_set = set(targets)
    retrieved_positions = set()

    for i, chunk in enumerate(retrieved[:k]):
        pos = chunk.get("position")
        if pos is not None:
            retrieved_positions.add(pos)

    hits = len(retrieved_positions & targets_set)
    return hits / len(targets_set)


def pair_recall_at_k(
    retrieved: List[dict],
    target_a: int,
    target_b: int,
    k: int,
) -> float:
    """Compute Pair Recall@K.

    Returns 1.0 if BOTH target chunks are in top-K, 0.0 otherwise.
    This is the key metric for multi-hop reasoning evaluation.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    target_a : int
        First target chunk position
    target_b : int
        Second target chunk position
    k : int
        Number of top results to consider

    Returns
    -------
    float
        1.0 if both targets retrieved, 0.0 otherwise
    """
    retrieved_positions = set()

    for i, chunk in enumerate(retrieved[:k]):
        pos = chunk.get("position")
        if pos is not None:
            retrieved_positions.add(pos)

    if target_a in retrieved_positions and target_b in retrieved_positions:
        return 1.0
    return 0.0


def hit_at_k(
    retrieved: List[dict],
    targets: Union[List[int], Set[int]],
    k: int,
) -> float:
    """Compute Hit@K (binary).

    Returns 1.0 if ANY target is in top-K, 0.0 otherwise.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks with 'position' key
    targets : List[int] or Set[int]
        Target chunk positions
    k : int
        Number of top results to consider

    Returns
    -------
    float
        1.0 if any target retrieved, 0.0 otherwise
    """
    if not targets:
        return 1.0

    targets_set = set(targets)
    for i, chunk in enumerate(retrieved[:k]):
        pos = chunk.get("position")
        if pos in targets_set:
            return 1.0
    return 0.0


def compute_recall_metrics(
    retrieved: List[dict],
    target_positions: List[int],
    k_values: List[int] = [3, 5, 10, 20],
) -> dict:
    """Compute multiple recall metrics at different K values.

    Parameters
    ----------
    retrieved : List[dict]
        Retrieved chunks
    target_positions : List[int]
        Target chunk positions (typically 2 for multi-hop)
    k_values : List[int]
        K values to compute metrics for

    Returns
    -------
    dict
        Dictionary of metric names to values
    """
    metrics = {}

    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(retrieved, target_positions, k)
        metrics[f"hit@{k}"] = hit_at_k(retrieved, target_positions, k)

        if len(target_positions) == 2:
            metrics[f"pair_recall@{k}"] = pair_recall_at_k(
                retrieved, target_positions[0], target_positions[1], k
            )

    return metrics


if __name__ == "__main__":
    # Test metrics
    retrieved = [
        {"position": 5, "score": 0.9},
        {"position": 2, "score": 0.8},
        {"position": 7, "score": 0.7},
        {"position": 1, "score": 0.6},
        {"position": 10, "score": 0.5},
    ]

    targets = [2, 7]  # Multi-hop targets

    print("Testing recall metrics:")
    print(f"  Targets: {targets}")
    print(f"  Retrieved positions: {[r['position'] for r in retrieved]}")
    print()

    for k in [1, 2, 3, 5]:
        recall = recall_at_k(retrieved, targets, k)
        pair_recall = pair_recall_at_k(retrieved, targets[0], targets[1], k)
        print(f"  K={k}: Recall={recall:.2f}, Pair Recall={pair_recall:.2f}")

    print()
    all_metrics = compute_recall_metrics(retrieved, targets)
    print("All metrics:", all_metrics)
