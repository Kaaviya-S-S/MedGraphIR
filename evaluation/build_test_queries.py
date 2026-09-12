"""
Builds evaluation/test_queries.csv from the ACTUAL saved knowledge graph.

Per the project brief ("Do not fabricate evaluation results"), the ground
truth here is not hand-typed - it is computed directly from
data/graph/medical_kg.pkl using plain networkx calls, independent of
src/retrieval.py, so the evaluation isn't grading retrieval.py against
answers that retrieval.py itself produced (that would be circular). What
IS manually curated is the list of (query, entity, relationship type)
below: a diverse, deliberately chosen mix of real entities found by
inspecting the graph - high-degree and low-degree interactions, drug-side
and disease-side TREATS queries, unfiltered "ANY relationship" queries,
typo-recovery cases, and unknown-entity negative cases.

Run once to (re)generate the CSV:
    python evaluation/build_test_queries.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src import config
from src.graph_builder import load_graph

# (query text, entity_id or None, relationship type or "ANY" or None)
# entity_id=None marks a negative test case (entity should not be found).
QUERY_SPECS = [
    # --- INTERACTS_WITH, high-degree drugs (tests precision under top-K truncation) ---
    ("What drugs interact with Warfarin?", "drug::warfarin", "INTERACTS_WITH"),
    ("What drugs interact with Dexamethasone?", "drug::dexamethasone", "INTERACTS_WITH"),
    ("What drugs interact with Clarithromycin?", "drug::clarithromycin", "INTERACTS_WITH"),
    ("What drugs interact with Tacrolimus?", "drug::tacrolimus", "INTERACTS_WITH"),
    ("What drugs interact with Methotrexate?", "drug::methotrexate", "INTERACTS_WITH"),
    # --- INTERACTS_WITH, low-degree drugs (full recall should be achievable) ---
    ("What drugs interact with Doripenem?", "drug::doripenem", "INTERACTS_WITH"),
    ("What drugs interact with Agalsidase beta?", "drug::agalsidase_beta", "INTERACTS_WITH"),
    ("What drugs interact with Ferumoxides?", "drug::ferumoxides", "INTERACTS_WITH"),
    ("What drugs interact with Cedazuridine?", "drug::cedazuridine", "INTERACTS_WITH"),
    ("What drugs interact with Bismuth subcitrate potassium?", "drug::bismuth_subcitrate_potassium", "INTERACTS_WITH"),
    # --- TREATS, drug -> disease direction ---
    ("What diseases does Metformin treat?", "drug::metformin", "TREATS"),
    ("What diseases does Methotrexate treat?", "drug::methotrexate", "TREATS"),
    ("What diseases does Prednisone treat?", "drug::prednisone", "TREATS"),
    ("What diseases does Acetylsalicylic acid treat?", "drug::acetylsalicylic_acid", "TREATS"),
    ("What diseases does Omeprazole treat?", "drug::omeprazole", "TREATS"),
    ("What diseases does Cyclosporine treat?", "drug::cyclosporine", "TREATS"),
    # --- TREATS, disease -> drug direction ("what treats X") ---
    ("What drugs treat hypertension?", "disease::hypertension", "TREATS"),
    ("What drugs treat asthma?", "disease::asthma", "TREATS"),
    ("What drugs treat Kawasaki disease?", "disease::kawasaki_disease", "TREATS"),
    ("What drugs treat gestational diabetes?", "disease::gestational_diabetes", "TREATS"),
    ("What drugs treat Barrett's esophagus?", "disease::barrett_s_esophagus", "TREATS"),
    ("What drugs treat nephrolithiasis?", "disease::nephrolithiasis", "TREATS"),
    ("What drugs treat amyotrophic lateral sclerosis?", "disease::amyotrophic_lateral_sclerosis", "TREATS"),
    ("What drugs treat malignant glioma?", "disease::malignant_glioma", "TREATS"),
    # --- No relationship keyword: expected = union of ALL relation types ---
    ("Tell me about Metformin", "drug::metformin", "ANY"),
    ("Show connections for Cyclosporine", "drug::cyclosporine", "ANY"),
    # --- Typo recovery (fuzzy matching tier should still resolve these) ---
    ("What drugs interact with Worfarin?", "drug::warfarin", "INTERACTS_WITH"),
    ("What diseases does Metfromin treat?", "drug::metformin", "TREATS"),
    # --- Negative cases: entity should NOT be found (graceful failure) ---
    ("xyzabc123", None, None),
    ("What drugs interact with Zzznonexistentdrug?", None, None),
]


def expected_neighbors(G, entity_id: str, relationship: str | None) -> list[str]:
    """Ground truth: canonical names of every node directly connected to
    entity_id (checking BOTH edge directions, per the graph_builder design
    decision), optionally filtered to one relationship type.
    """
    names = set()
    for _, target, data in G.out_edges(entity_id, data=True):
        if relationship in (None, "ANY", data["relationship"]):
            names.add(G.nodes[target]["canonical_name"])
    for source, _, data in G.in_edges(entity_id, data=True):
        if relationship in (None, "ANY", data["relationship"]):
            names.add(G.nodes[source]["canonical_name"])
    return sorted(names)


def main():
    G = load_graph()
    rows = []

    for query, entity_id, relationship in QUERY_SPECS:
        if entity_id is None:
            rows.append({"query": query, "expected_entities": "", "expected_relationships": ""})
            continue

        if entity_id not in G:
            raise ValueError(f"'{entity_id}' (query: {query!r}) is not a node in the current graph")

        expected = expected_neighbors(G, entity_id, relationship)
        rows.append(
            {
                "query": query,
                "expected_entities": "|".join(expected),
                "expected_relationships": relationship or "",
            }
        )
        print(f"[EVAL-BUILD] {query!r} -> {len(expected)} expected entities ({relationship})")

    df = pd.DataFrame(rows)
    config.PROJECT_ROOT.joinpath("evaluation").mkdir(exist_ok=True)
    out_path = config.PROJECT_ROOT / "evaluation" / "test_queries.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[EVAL-BUILD] Saved {len(df)} queries to {out_path}")


if __name__ == "__main__":
    main()
