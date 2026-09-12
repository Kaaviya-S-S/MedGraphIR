"""
Builds and persists the NetworkX knowledge graph from the canonical entity
table (Milestone 2) and validated relation triples (Milestone 3).

Graph shape
-----------
A single `nx.DiGraph` is used for both relationship types:

- TREATS (drug -> disease) is genuinely directional, so one directed edge
  per relation is correct as-is.
- INTERACTS_WITH (drug <-> drug) is conceptually symmetric, but DDInter
  records each interaction as exactly one row in one direction (verified
  during Milestone 1: no pair appears in both directions). Rather than
  doubling the edge count by inserting an artificial reverse edge (which
  would need a fabricated record_id for a "fact" that isn't really a
  separate source record), a single directed edge is stored, matching the
  source data exactly. The retrieval layer (Milestone 5) is responsible
  for treating INTERACTS_WITH as bidirectional by checking both successors
  and predecessors - this keeps provenance 1:1 with the source rows.

Node attributes: canonical_name, entity_type, entity_id (duplicated as an
attribute for convenience when iterating), aliases (list), mention_count.

Edge attributes: relationship, record_id, evidence, plus whatever
source-specific fields came from Relation.attributes (e.g. severity,
indication_category) flattened directly onto the edge - so e.g.
G['drug::warfarin']['drug::aspirin']['severity'] works without an extra
lookup level.
"""

import json
import pickle

import networkx as nx
import pandas as pd

from src import config
from src.relation_extraction import Relation


def _entities_to_nodes(G: nx.DiGraph, entities: pd.DataFrame) -> None:
    for _, row in entities.iterrows():
        aliases = row["aliases"].split("|") if isinstance(row["aliases"], str) else []
        G.add_node(
            row["entity_id"],
            entity_id=row["entity_id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            aliases=aliases,
            mention_count=int(row["mention_count"]),
        )


def _relation_attrs(relation_attributes) -> dict:
    """Relation.attributes may already be a dict (in-memory Relation
    objects) or a JSON string (if relations were reloaded from
    relations.csv) - normalize to a dict either way.
    """
    if isinstance(relation_attributes, str):
        return json.loads(relation_attributes)
    return dict(relation_attributes)


def build_graph(entities: pd.DataFrame, relations: list[Relation] | pd.DataFrame) -> nx.DiGraph:
    """Construct the knowledge graph.

    Args:
        entities: canonical entity table (see entity_linking.build_entity_table).
        relations: either a list of Relation objects (preferred, from
            relation_extraction.extract_structured_relations) or a
            DataFrame read back from relations.csv.

    Returns:
        A directed graph with one node per entity and one edge per
        validated relation whose endpoints both resolved to a known
        entity.
    """
    G = nx.DiGraph()
    _entities_to_nodes(G, entities)

    if isinstance(relations, pd.DataFrame):
        relation_rows = relations.to_dict(orient="records")
    else:
        relation_rows = [r.model_dump() for r in relations]

    skipped = 0
    for rel in relation_rows:
        source_id, target_id = rel["source_entity_id"], rel["target_entity_id"]
        if source_id not in G or target_id not in G:
            skipped += 1
            continue

        edge_attrs = {
            "relationship": rel["relationship"],
            "record_id": rel["record_id"],
            "evidence": rel["evidence"],
            **_relation_attrs(rel["attributes"]),
        }
        G.add_edge(source_id, target_id, **edge_attrs)

    if skipped:
        print(f"[GRAPH] Skipped {skipped} relations with an unresolved entity endpoint")

    print(f"[GRAPH] Nodes: {G.number_of_nodes():,}")
    print(f"[GRAPH] Edges: {G.number_of_edges():,}")
    print(f"[GRAPH] Isolated nodes (no edges): {nx.number_of_isolates(G):,}")
    return G


def save_graph(G: nx.DiGraph, path=config.GRAPH_FILE) -> None:
    config.GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"[GRAPH] Saved graph to {path}")


def load_graph(path=config.GRAPH_FILE) -> nx.DiGraph:
    if not path.exists():
        raise FileNotFoundError(
            f"No saved graph at {path}. Run `python build_graph.py` first."
        )
    with open(path, "rb") as f:
        G = pickle.load(f)
    print(f"[GRAPH] Loaded graph from {path} ({G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges)")
    return G


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.graph_builder`
    # (uses whatever is currently in data/processed/, at MAX_ROWS_PER_FILE scale)
    from src.entity_linking import link_entities
    from src.relation_extraction import extract_structured_relations

    interactions_df = pd.read_csv(config.PROCESSED_INTERACTIONS_FILE)
    indications_df = pd.read_csv(config.PROCESSED_INDICATIONS_FILE)
    entities_df, interactions_df, indications_df = link_entities(interactions_df, indications_df)
    relations = extract_structured_relations(interactions_df, indications_df)

    G = build_graph(entities_df, relations)
    save_graph(G)

    print("\nMetformin's neighborhood (proof the graph is queryable):")
    node_id = "drug::metformin"
    print("Outgoing edges (successors):")
    for target in G.successors(node_id):
        edge = G[node_id][target]
        print(f"  {node_id} -[{edge['relationship']}]-> {target}")
    print("Incoming edges (predecessors):")
    for source in G.predecessors(node_id):
        edge = G[source][node_id]
        print(f"  {source} -[{edge['relationship']}]-> {node_id}")
