"""Create a pooled candidate file for independent human relevance judgments.

Run after building the graph and processed relation table:
    python evaluation/build_judgment_pool.py

Annotate relevance in the output CSV: 0 = irrelevant, 1 = related, 2 = directly
answers the query. The pool is deliberately not treated as ground truth until
those judgments have been supplied.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.bm25_baseline import load_bm25_baseline
from src.retrieval import load_retrieval_engine, retrieve_relationships

EVALUATION_DIR = Path(__file__).resolve().parent
QUERIES_FILE = EVALUATION_DIR / "judgment_queries.csv"
POOL_FILE = EVALUATION_DIR / "judgment_candidates.csv"


def main() -> None:
    G, index = load_retrieval_engine()
    bm25 = load_bm25_baseline()
    queries = pd.read_csv(QUERIES_FILE, keep_default_na=False)
    if "query" not in queries or queries["query"].str.strip().eq("").any():
        raise ValueError(f"{QUERIES_FILE} must contain non-empty query values")
    if queries["query"].duplicated().any():
        raise ValueError(f"{QUERIES_FILE} contains duplicate queries")

    previous_ratings: dict[tuple[str, str], str] = {}
    previous_reviews: dict[tuple[str, str], str] = {}
    if POOL_FILE.exists():
        previous = pd.read_csv(
            POOL_FILE,
            keep_default_na=False,
            dtype={"query": str, "record_id": str},
        )
        required = {"query", "record_id", "relevance"}
        if not required.issubset(previous.columns):
            raise ValueError(f"Existing {POOL_FILE} is missing columns: {sorted(required - set(previous.columns))}")
        previous_ratings = {
            (row.query, row.record_id): row.relevance
            for row in previous[["query", "record_id", "relevance"]].itertuples(index=False)
            if str(row.relevance).strip()
        }
        if "human_reviewed" in previous:
            previous_reviews = {
                (row.query, row.record_id): row.human_reviewed
                for row in previous[["query", "record_id", "human_reviewed"]].itertuples(index=False)
                if str(row.human_reviewed).strip()
            }

    output_rows = []
    for query in queries["query"].astype(str):
        candidate_details: dict[str, dict] = {}
        graph_result = retrieve_relationships(query, G, index, max_hops=1, top_k=config.TOP_K_RESULTS)
        bm25_results = bm25.search(query, top_k=config.TOP_K_RESULTS)
        result_lists = {
            "graph": graph_result.facts,
            "bm25": bm25_results,
        }

        for results in result_lists.values():
            for result in results:
                record_id = result.record_id
                if record_id not in candidate_details:
                    candidate_details[record_id] = {
                        "source_entity": result.source_name if hasattr(result, "source_name") else result.source_entity,
                        "relationship": result.relationship,
                        "target_entity": result.target_name if hasattr(result, "target_name") else result.target_entity,
                        "evidence": result.evidence,
                    }

        ordered_record_ids = sorted(
            candidate_details,
            key=lambda record_id: hashlib.sha256(f"{query}\0{record_id}".encode("utf-8")).digest(),
        )
        for record_id in ordered_record_ids:
            details = candidate_details[record_id]
            row = {
                "query": query,
                "record_id": record_id,
                **details,
                "relevance": previous_ratings.get((query, record_id), ""),
                "human_reviewed": previous_reviews.get((query, record_id), ""),
            }
            output_rows.append(row)

        if not candidate_details:
            raise ValueError(f"No retrieval method returned candidates for query: {query!r}")

        print(f"[POOL] {query!r}: {len(candidate_details)} unique candidates")

    pool = pd.DataFrame(output_rows)
    pool.to_csv(POOL_FILE, index=False)
    print(
        f"[POOL] Saved {len(pool):,} candidate rows to {POOL_FILE}; "
        f"preserved {len(previous_ratings):,} existing relevance ratings."
    )
    print("[POOL] Fill relevance with 0 (irrelevant), 1 (related), or 2 (directly answers).")


if __name__ == "__main__":
    main()
