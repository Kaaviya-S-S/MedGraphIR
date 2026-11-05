"""Pool multi-entity graph and BM25 candidates for manual relevance judgments.

Run after building the graph:
    python evaluation/build_multi_entity_judgment_pool.py

Review each candidate and set relevance to 0 (irrelevant), 1 (related), or
2 (direct answer), then set human_reviewed to yes. The script does not assign
relevance grades.
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.bm25_baseline import load_bm25_baseline
from src.retrieval import (
    QueryIntent,
    load_retrieval_engine,
    retrieve_multi_entity_relationships,
)

EVALUATION_DIR = Path(__file__).resolve().parent
QUERIES_FILE = EVALUATION_DIR / "multi_entity_queries.csv"
POOL_FILE = EVALUATION_DIR / "multi_entity_judgment_candidates.csv"


def main() -> None:
    queries = pd.read_csv(QUERIES_FILE, keep_default_na=False, dtype={"query": str})
    if "query" not in queries or queries["query"].str.strip().eq("").any():
        raise ValueError(f"{QUERIES_FILE} must contain non-empty query values")
    if queries["query"].duplicated().any():
        raise ValueError(f"{QUERIES_FILE} contains duplicate queries")

    graph, index = load_retrieval_engine()
    bm25 = load_bm25_baseline()

    previous_values: dict[tuple[str, str], tuple[str, str]] = {}
    if POOL_FILE.exists():
        previous = pd.read_csv(
            POOL_FILE,
            keep_default_na=False,
            dtype={"query": str, "record_id": str},
        )
        required = {"query", "record_id", "relevance", "human_reviewed"}
        if not required.issubset(previous.columns):
            raise ValueError(
                f"Existing {POOL_FILE} is missing columns: {sorted(required - set(previous.columns))}"
            )
        previous_values = {
            (row.query, row.record_id): (row.relevance, row.human_reviewed)
            for row in previous[["query", "record_id", "relevance", "human_reviewed"]]
            .itertuples(index=False)
        }

    output_rows = []
    for query in queries["query"]:
        candidates: dict[str, dict[str, str]] = {}
        graph_result = retrieve_multi_entity_relationships(
            query, graph, index, top_k=config.TOP_K_RESULTS
        )
        if graph_result.intent not in (
            QueryIntent.DIRECT_RELATIONSHIP,
            QueryIntent.AND,
            QueryIntent.OR,
        ):
            raise ValueError(
                f"Benchmark query {query!r} was classified as {graph_result.intent.value!r}"
            )
        for fact in graph_result.facts:
            candidates[fact.record_id] = {
                "source_entity": fact.source_name,
                "relationship": fact.relationship,
                "target_entity": fact.target_name,
                "evidence": fact.evidence,
            }

        for result in bm25.search(query, top_k=config.TOP_K_RESULTS):
            candidates.setdefault(
                result.record_id,
                {
                    "source_entity": result.source_entity,
                    "relationship": result.relationship,
                    "target_entity": result.target_entity,
                    "evidence": result.evidence,
                },
            )

        if not candidates:
            raise ValueError(f"No graph or BM25 candidates were found for {query!r}")

        ordered_ids = sorted(
            candidates,
            key=lambda record_id: hashlib.sha256(
                f"{query}\0{record_id}".encode("utf-8")
            ).digest(),
        )
        for record_id in ordered_ids:
            relevance, reviewed = previous_values.get((query, record_id), ("", ""))
            output_rows.append(
                {
                    "query": query,
                    "record_id": record_id,
                    **candidates[record_id],
                    "relevance": relevance,
                    "human_reviewed": reviewed,
                }
            )
        print(f"[MULTI-ENTITY POOL] {query!r}: {len(candidates)} unique candidates")

    pool = pd.DataFrame(output_rows)
    pool.to_csv(POOL_FILE, index=False)
    print(f"[MULTI-ENTITY POOL] Saved {len(pool):,} candidates to {POOL_FILE}")
    print("[MULTI-ENTITY POOL] Review every row and mark human_reviewed=yes.")


if __name__ == "__main__":
    main()
