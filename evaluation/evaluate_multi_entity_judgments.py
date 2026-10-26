"""Evaluate multi-entity and BM25 rankings against reviewed judgments.

Run after manually completing the pooled judgment file:
    python evaluation/evaluate_multi_entity_judgments.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from evaluation.relevance_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from src import config
from src.bm25_baseline import load_bm25_baseline
from src.retrieval import load_retrieval_engine, retrieve_multi_entity_relationships

EVALUATION_DIR = Path(__file__).resolve().parent
QUERIES_FILE = EVALUATION_DIR / "multi_entity_queries.csv"
POOL_FILE = EVALUATION_DIR / "multi_entity_judgment_candidates.csv"
RANK_CUTOFF = config.TOP_K_RESULTS


def _rankings(query: str, graph, index, bm25) -> dict[str, list[str]]:
    graph_result = retrieve_multi_entity_relationships(
        query, graph, index, top_k=RANK_CUTOFF
    )
    return {
        "multi_entity": [fact.record_id for fact in graph_result.facts],
        "bm25": [result.record_id for result in bm25.search(query, top_k=RANK_CUTOFF)],
    }


def main() -> None:
    if not POOL_FILE.exists():
        raise FileNotFoundError(
            f"{POOL_FILE} not found. Run `python evaluation/build_multi_entity_judgment_pool.py` first."
        )

    pool = pd.read_csv(
        POOL_FILE,
        keep_default_na=False,
        dtype={"query": str, "record_id": str},
    )
    required = {"query", "record_id", "relevance", "human_reviewed"}
    if not required.issubset(pool.columns):
        raise ValueError(f"{POOL_FILE} is missing columns: {sorted(required - set(pool.columns))}")
    if pool.duplicated(["query", "record_id"]).any():
        raise ValueError("Judgment pool contains duplicate (query, record_id) rows")
    if pool["relevance"].astype(str).str.strip().eq("").any():
        raise ValueError("Every pooled candidate must be graded 0, 1, or 2")
    if not pool["human_reviewed"].astype(str).str.casefold().eq("yes").all():
        raise ValueError("Every candidate must be manually reviewed and marked human_reviewed=yes")

    try:
        grades = pd.to_numeric(pool["relevance"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Relevance grades must be integers: 0, 1, or 2") from exc
    if not grades.isin([0, 1, 2]).all():
        raise ValueError("Relevance grades must be 0 (irrelevant), 1 (related), or 2 (direct answer)")
    pool["relevance"] = grades.astype(int)

    queries = pd.read_csv(QUERIES_FILE, keep_default_na=False, dtype={"query": str})["query"]
    if set(pool["query"]) != set(queries):
        raise ValueError("Judged queries must match evaluation/multi_entity_queries.csv exactly")

    graph, index = load_retrieval_engine()
    bm25 = load_bm25_baseline()
    rows = []
    for query, judgments in pool.groupby("query", sort=False):
        relevance = dict(zip(judgments["record_id"], judgments["relevance"]))
        relevant_ids = {record_id for record_id, grade in relevance.items() if grade > 0}
        if not relevant_ids:
            raise ValueError(f"No relevant candidates were judged for {query!r}")

        for method, ranked_ids in _rankings(query, graph, index, bm25).items():
            unjudged = set(ranked_ids) - set(relevance)
            if unjudged:
                raise ValueError(
                    f"{query!r}: {method} returned {len(unjudged)} unjudged records. "
                    "Rebuild and review the candidate pool."
                )
            rows.append(
                {
                    "query": query,
                    "method": method,
                    "n_judged_relevant": len(relevant_ids),
                    "precision_at_5": precision_at_k(ranked_ids, relevant_ids, 5),
                    "precision_at_10": precision_at_k(ranked_ids, relevant_ids, 10),
                    "precision_at_20": precision_at_k(ranked_ids, relevant_ids, 20),
                    "recall_at_20": recall_at_k(ranked_ids, relevant_ids, 20),
                    "map_at_20": average_precision_at_k(ranked_ids, relevant_ids, 20),
                    "ndcg_at_10": ndcg_at_k(ranked_ids, relevance, 10),
                }
            )

    detailed = pd.DataFrame(rows)
    summary = detailed.groupby("method").mean(numeric_only=True).round(4)
    detailed_path = EVALUATION_DIR / "results_multi_entity_judged_detailed.csv"
    summary_path = EVALUATION_DIR / "results_multi_entity_judged_summary.csv"
    detailed.to_csv(detailed_path, index=False)
    summary.to_csv(summary_path)
    print(f"[MULTI-ENTITY EVAL] Saved query-level scores to {detailed_path}")
    print(f"[MULTI-ENTITY EVAL] Saved macro-averaged scores to {summary_path}")
    print(summary.to_string())


if __name__ == "__main__":
    main()
