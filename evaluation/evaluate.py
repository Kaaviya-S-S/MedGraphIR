"""
Runs two retrieval methods (graph and BM25) against
evaluation/test_queries.csv and reports Precision / Recall / F1 /
Precision@K / retrieval time.
No numbers in this file's output are hand-typed - everything is computed
by running the real system against the real saved graph.

Comparison design decisions:
  - Graph retrieval is run with max_hops=1, matching how the ground truth
    was defined (direct/1-hop neighbors) - an apples-to-apples comparison.
  - Both methods' results are reduced to the same shape: a ranked list of
    "the other entity" in each returned fact/record, excluding the
    query's own matched entity name (taken from graph retrieval's own
    entity resolution, since the lexical baseline doesn't do entity
    resolution at all).
  - BM25 is the single lexical baseline, ranked over the same relation
    records the graph is built from.

Run with:
    python evaluation/evaluate.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.bm25_baseline import load_bm25_baseline
from src.retrieval import load_retrieval_engine, retrieve_relationships

TEST_QUERIES_FILE = Path(__file__).resolve().parent / "test_queries.csv"
PRECISION_AT_KS = (5, 10)


def parse_expected(cell: str) -> set[str]:
    if pd.isna(cell) or cell == "":
        return set()
    return set(cell.split("|"))


def ranked_other_entities(pairs: list[tuple[str, str]], exclude_name: str | None) -> list[str]:
    """Flatten (source, target) pairs into a ranked list of unique entity
    names, dropping the query's own entity so we're left with "what was
    retrieved as the answer", not the query subject echoed back.
    """
    seen, ranked = set(), []
    exclude_lower = exclude_name.lower() if exclude_name else None
    for source, target in pairs:
        for name in (source, target):
            if exclude_lower and name.lower() == exclude_lower:
                continue
            if name not in seen:
                seen.add(name)
                ranked.append(name)
    return ranked


def precision_recall_f1(retrieved: set[str], expected: set[str]) -> tuple[float, float, float]:
    """Standard set-based IR metrics, with explicit handling for the
    "nothing should be found" case (expected is empty): correct iff
    retrieved is also empty.
    """
    if not expected:
        precision = 1.0 if not retrieved else 0.0
        return precision, 1.0, precision
    if not retrieved:
        return 0.0, 0.0, 0.0
    tp = len(retrieved & expected)
    precision = tp / len(retrieved)
    recall = tp / len(expected)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def evaluate_query(query: str, expected: set[str], G, index, bm25_retriever) -> list[dict]:
    rows = []

    # --- Graph retrieval ---
    t0 = time.perf_counter()
    result = retrieve_relationships(query, G, index, max_hops=1, top_k=config.TOP_K_RESULTS)
    graph_time_ms = (time.perf_counter() - t0) * 1000
    matched_name = result.matched_entity.canonical_name if result.matched_entity else None

    graph_pairs = [(f.source_name, f.target_name) for f in result.facts]
    graph_ranked = ranked_other_entities(graph_pairs, matched_name)
    precision, recall, f1 = precision_recall_f1(set(graph_ranked), expected)
    row = {"query": query, "method": "graph", "precision": precision, "recall": recall,
           "f1": f1, "n_retrieved": len(graph_ranked), "n_expected": len(expected),
           "retrieval_time_ms": graph_time_ms}
    for k in PRECISION_AT_KS:
        row[f"precision_at_{k}"] = precision_recall_f1(set(graph_ranked[:k]), expected)[0]
    rows.append(row)

    # --- BM25 lexical baseline ---
    t0 = time.perf_counter()
    bm25_results = bm25_retriever.search(query, top_k=config.TOP_K_RESULTS)
    bm25_time_ms = (time.perf_counter() - t0) * 1000

    bm25_pairs = [(r.source_entity, r.target_entity) for r in bm25_results]
    bm25_ranked = ranked_other_entities(bm25_pairs, matched_name)
    precision, recall, f1 = precision_recall_f1(set(bm25_ranked), expected)
    row = {"query": query, "method": "bm25", "precision": precision, "recall": recall,
           "f1": f1, "n_retrieved": len(bm25_ranked), "n_expected": len(expected),
           "retrieval_time_ms": bm25_time_ms}
    for k in PRECISION_AT_KS:
        row[f"precision_at_{k}"] = precision_recall_f1(set(bm25_ranked[:k]), expected)[0]
    rows.append(row)

    return rows


def main():
    print("[EVAL] Loading retrieval engine and lexical baseline...")
    G, index = load_retrieval_engine()
    bm25_retriever = load_bm25_baseline()

    test_queries = pd.read_csv(TEST_QUERIES_FILE, keep_default_na=False)
    print(f"[EVAL] Loaded {len(test_queries)} test queries\n")

    all_rows = []
    for _, row in test_queries.iterrows():
        expected = parse_expected(row["expected_entities"])
        all_rows.extend(evaluate_query(row["query"], expected, G, index, bm25_retriever))

    detailed = pd.DataFrame(all_rows)
    detailed_path = Path(__file__).resolve().parent / "results_detailed.csv"
    detailed.to_csv(detailed_path, index=False)
    print(f"[EVAL] Saved per-query results to {detailed_path}")

    metric_cols = ["precision", "recall", "f1", "precision_at_5", "precision_at_10", "retrieval_time_ms"]
    summary = detailed.groupby("method")[metric_cols].mean().round(4)
    summary["retrieval_time_ms"] = summary["retrieval_time_ms"].round(2)

    print("\n" + "=" * 78)
    print("RESULTS (macro-averaged over", len(test_queries), "queries)")
    print("=" * 78)
    print(summary.to_string())

    summary_path = Path(__file__).resolve().parent / "results_summary.csv"
    summary.to_csv(summary_path)
    print(f"\n[EVAL] Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
