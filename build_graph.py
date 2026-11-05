"""
Standalone script that builds the medical knowledge graph from scratch and
saves it to disk. Run this whenever the raw data or pipeline code changes.

    python build_graph.py                # full dataset (~222k DDInter rows)
    python build_graph.py --max-rows 2000 # quick end-to-end dry run

This is intentionally separate from app.py: the graph should be built once,
not on every Streamlit page load (see README "Performance").
"""

import argparse
import time

from src import config
from src.graph_pipeline import build_knowledge_graph


def main(max_rows_per_file: int | None) -> None:
    start = time.time()
    print("[BUILD] Loading data, linking entities, and building the graph")
    build_knowledge_graph(max_rows_per_file)
    elapsed = time.time() - start
    print("\n" + "=" * 70)
    print(f"[BUILD] Done in {elapsed:.1f}s. Graph saved to {config.GRAPH_FILE}")
    print("[BUILD] Run `streamlit run app.py` to query it.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the medical knowledge graph.")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Cap rows read per DDInter file (omit for the full ~222k-row dataset).",
    )
    args = parser.parse_args()
    main(args.max_rows)
