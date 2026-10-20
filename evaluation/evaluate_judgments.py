"""Evaluate retrieval methods against human-labeled pooled relevance judgments.

Run after annotating evaluation/judgment_candidates.csv:
    python evaluation/evaluate_judgments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.bm25_baseline import load_bm25_baseline
from src.retrieval import load_retrieval_engine, retrieve_relationships
from evaluation.relevance_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

EVALUATION_DIR = Path(__file__).resolve().parent
POOL_FILE = EVALUATION_DIR / "judgment_candidates.csv"
RANK_CUTOFF = 20


def _rankings(query: str, G, index, bm25) -> dict[str, list[str]]:
    graph = retrieve_relationships(query, G, index, max_hops=1, top_k=RANK_CUTOFF)
    bm25_results = bm25.search(query, top_k=RANK_CUTOFF)
    return {
        "graph": [fact.record_id for fact in graph.facts],
        "bm25": [result.record_id for result in bm25_results],
    }


def main() -> None:
    if not POOL_FILE.exists():
        raise FileNotFoundError(
            f"{POOL_FILE} not found. Run `python evaluation/build_judgment_pool.py` first."
        )
    pool = pd.read_csv(
        POOL_FILE,
        keep_default_na=False,
        dtype={"query": str, "record_id": str},
    )
    required = {"query", "record_id", "relevance"}
    if not required.issubset(pool.columns):
        raise ValueError(f"{POOL_FILE} is missing columns: {sorted(required - set(pool.columns))}")
    if pool.duplicated(["query", "record_id"]).any():
        raise ValueError("judgment_candidates.csv contains duplicate (query, record_id) rows")
    if pool["relevance"].astype(str).str.strip().eq("").any():
        raise ValueError("Every pooled candidate must be judged 0, 1, or 2 before evaluation")

    try:
        numeric_relevance = pd.to_numeric(pool["relevance"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Relevance judgments must be integers: 0, 1, or 2") from exc
    if not numeric_relevance.isin([0, 1, 2]).all():
        raise ValueError("Relevance judgments must be 0 (irrelevant), 1 (related), or 2 (direct answer)")
    pool["relevance"] = numeric_relevance.astype(int)

    queries_file = EVALUATION_DIR / "judgment_queries.csv"
    expected_queries = set(
        pd.read_csv(queries_file, keep_default_na=False, dtype={"query": str})["query"]
    )
    judged_queries = set(pool["query"])
    if judged_queries != expected_queries:
        missing = sorted(expected_queries - judged_queries)
        unexpected = sorted(judged_queries - expected_queries)
        raise ValueError(
            f"Judged query set does not match {queries_file}. "
            f"Missing: {missing}; unexpected: {unexpected}"
        )

    G, index = load_retrieval_engine()
    bm25 = load_bm25_baseline()
    all_rows = []

    for query, judgments in pool.groupby("query", sort=False):
        relevance = dict(zip(judgments["record_id"], judgments["relevance"]))
        relevant_ids = {record_id for record_id, grade in relevance.items() if grade > 0}
        if not relevant_ids:
            raise ValueError(f"No relevant candidates (relevance > 0) were judged for query: {query!r}")

        rankings = _rankings(query, G, index, bm25)
        for method, ranked_ids in rankings.items():
            unjudged = set(ranked_ids) - set(relevance)
            if unjudged:
                sample = ", ".join(sorted(unjudged)[:5])
                raise ValueError(
                    f"{query!r}: {method} returned {len(unjudged)} unjudged records "
                    f"(e.g. {sample}). Rebuild the candidate pool and judge the new rows."
                )

            all_rows.append(
                {
                    "query": query,
                    "method": method,
                    "n_judged_relevant": len(relevant_ids),
                    "precision_at_5": precision_at_k(ranked_ids, relevant_ids, 5),
                    "precision_at_10": precision_at_k(ranked_ids, relevant_ids, 10),
                    "precision_at_20": precision_at_k(ranked_ids, relevant_ids, 20),
                    "recall_at_20": recall_at_k(ranked_ids, relevant_ids, 20),
                    "map_at_20": average_precision_at_k(ranked_ids, relevant_ids, 20),
                    "ndcg_at_5": ndcg_at_k(ranked_ids, relevance, 5),
                    "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
                }
            )

    detailed = pd.DataFrame(all_rows)
    summary = detailed.groupby("method").mean(numeric_only=True).round(4)
    detailed_path = EVALUATION_DIR / "results_judged_detailed.csv"
    summary_path = EVALUATION_DIR / "results_judged_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path)
    print(f"[JUDGED EVAL] Saved query-level scores to {detailed_path}")
    print(f"[JUDGED EVAL] Saved macro-averaged scores to {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
