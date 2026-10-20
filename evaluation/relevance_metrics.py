"""Metrics for evaluating rankings against independently judged relevance."""

import math


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    return sum(record_id in relevant_ids for record_id in ranked_ids[:k]) / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant_ids:
        raise ValueError("recall is undefined when no relevant records are judged")
    return sum(record_id in relevant_ids for record_id in ranked_ids[:k]) / len(relevant_ids)


def average_precision_at_k(
    ranked_ids: list[str], relevant_ids: set[str], k: int
) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not relevant_ids:
        raise ValueError("average precision is undefined when no relevant records are judged")

    hits = 0
    precision_sum = 0.0
    for rank, record_id in enumerate(ranked_ids[:k], start=1):
        if record_id in relevant_ids:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / min(len(relevant_ids), k)


def ndcg_at_k(ranked_ids: list[str], relevance: dict[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")

    dcg = sum(
        (2**relevance.get(record_id, 0) - 1) / math.log2(rank + 1)
        for rank, record_id in enumerate(ranked_ids[:k], start=1)
    )
    ideal_grades = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_grades, start=1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0
