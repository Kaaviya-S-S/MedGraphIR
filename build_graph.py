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
from src.data_loader import load_raw_data
from src.entity_linking import link_entities, save_entities
from src.graph_builder import build_graph, save_graph
from src.preprocessing import clean_indications, clean_interactions
from src.relation_extraction import extract_structured_relations, relations_to_dataframe, save_relations


def main(max_rows_per_file: int | None) -> None:
    start = time.time()

    print("=" * 70)
    print("[BUILD] Stage 1/5: Loading raw data")
    print("=" * 70)
    raw_interactions, raw_indications = load_raw_data(max_rows_per_file)

    print("\n" + "=" * 70)
    print("[BUILD] Stage 2/5: Preprocessing")
    print("=" * 70)
    interactions = clean_interactions(raw_interactions)
    indications = clean_indications(raw_indications)
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(config.PROCESSED_INTERACTIONS_FILE, index=False)
    indications.to_csv(config.PROCESSED_INDICATIONS_FILE, index=False)

    print("\n" + "=" * 70)
    print("[BUILD] Stage 3/5: Entity extraction & normalization")
    print("=" * 70)
    entities, interactions, indications = link_entities(interactions, indications)
    save_entities(entities)

    print("\n" + "=" * 70)
    print("[BUILD] Stage 4/5: Relation extraction")
    print("=" * 70)
    relations = extract_structured_relations(interactions, indications)
    save_relations(relations_to_dataframe(relations))

    print("\n" + "=" * 70)
    print("[BUILD] Stage 5/5: Knowledge graph construction")
    print("=" * 70)
    G = build_graph(entities, relations)
    save_graph(G)

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
