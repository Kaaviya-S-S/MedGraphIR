"""Validate source relationships, build the NetworkX graph, and save it."""

import json
import pickle

import networkx as nx
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from src import config
from src.data_pipeline import clean_indications, clean_interactions, link_entities, load_raw_data, save_entities


class Relation(BaseModel):
    """One source-backed relationship, with its provenance and evidence."""

    source_entity: str
    source_entity_id: str
    relationship: str
    target_entity: str
    target_entity_id: str
    record_id: str = Field(description="ID of the source dataset row, for provenance")
    evidence: str = Field(default="", description="Human-readable evidence shown in the UI")
    attributes: dict = Field(default_factory=dict, description="Source-specific structured evidence")

    @field_validator("relationship")
    @classmethod
    def relationship_must_be_controlled(cls, value: str) -> str:
        if value not in config.VALID_RELATIONS:
            raise ValueError(
                f"'{value}' is not in the controlled relationship vocabulary: "
                f"{sorted(config.VALID_RELATIONS)}"
            )
        return value


def _interaction_row_to_relation(row: pd.Series) -> Relation:
    return Relation(
        source_entity=row["source_entity"],
        source_entity_id=row["source_entity_id"],
        relationship=row["relationship"],
        target_entity=row["target_entity"],
        target_entity_id=row["target_entity_id"],
        record_id=row["record_id"],
        evidence=f"DDInter severity: {row['severity']} (ATC category: {row['atc_label']})",
        attributes={"severity": row["severity"], "atc_code": row["atc_code"]},
    )


def _indication_row_to_relation(row: pd.Series) -> Relation:
    category = {"DM": "disease-modifying", "SYM": "symptomatic relief"}.get(
        row["indication_category"], row["indication_category"]
    )
    return Relation(
        source_entity=row["source_entity"],
        source_entity_id=row["source_entity_id"],
        relationship=row["relationship"],
        target_entity=row["target_entity"],
        target_entity_id=row["target_entity_id"],
        record_id=row["record_id"],
        evidence=(
            f"PharmacotherapyDB indication ({category}), confirmed by "
            f"{row['n_curators']} curator(s) using {row['n_resources']} resource(s)"
        ),
        attributes={
            "indication_category": row["indication_category"],
            "n_curators": int(row["n_curators"]),
            "n_resources": int(row["n_resources"]),
        },
    )


def extract_structured_relations(
    interactions: pd.DataFrame,
    indications: pd.DataFrame,
) -> list[Relation]:
    """Convert already-linked source rows into validated relationships."""
    relations = []
    failed = 0
    for rows, converter in (
        (interactions, _interaction_row_to_relation),
        (indications, _indication_row_to_relation),
    ):
        for _, row in rows.iterrows():
            try:
                relations.append(converter(row))
            except ValidationError:
                failed += 1

    print(
        f"[RELATION] Built {len(relations):,} validated relations from structured data "
        f"({failed} rejected by validation)"
    )
    return relations


_LLM_PROMPT_TEMPLATE = """You are extracting a medical relationship from the text below.

Text: "{text}"

Entity A: {source_entity}
Entity B: {target_entity}

Classify the relationship FROM Entity A TO Entity B using EXACTLY one of these
labels: {allowed_labels}. Only use information stated in the text - do not
guess or use outside medical knowledge. If none of the labels are clearly
supported by the text, respond with "NONE".

Respond with ONLY a JSON object, no other text:
{{"relationship": "<LABEL or NONE>"}}
"""


def extract_relation_with_llm(
    text: str,
    source_entity: str,
    source_entity_id: str,
    target_entity: str,
    target_entity_id: str,
    record_id: str,
    model: str = config.OLLAMA_MODEL,
) -> Relation | None:
    """Optional Ollama extension for classifying a relationship from free text."""
    try:
        import ollama
    except ImportError:
        print("[RELATION] 'ollama' package not installed, skipping LLM extraction")
        return None

    prompt = _LLM_PROMPT_TEMPLATE.format(
        text=text,
        source_entity=source_entity,
        target_entity=target_entity,
        allowed_labels=", ".join(sorted(config.VALID_RELATIONS)),
    )
    try:
        response = ollama.generate(model=model, prompt=prompt, format="json")
        label = json.loads(response["response"]).get("relationship", "NONE")
    except Exception as exc:
        print(f"[RELATION] LLM extraction failed ({exc}), skipping this fact")
        return None

    if label == "NONE" or label not in config.VALID_RELATIONS:
        print(f"[RELATION] LLM returned no valid relationship ('{label}'), discarding")
        return None

    try:
        return Relation(
            source_entity=source_entity,
            source_entity_id=source_entity_id,
            relationship=label,
            target_entity=target_entity,
            target_entity_id=target_entity_id,
            record_id=record_id,
            evidence=f'LLM-inferred from text: "{text}"',
            attributes={"extraction_method": "llm", "model": model},
        )
    except ValidationError as exc:
        print(f"[RELATION] LLM output failed validation, discarding: {exc}")
        return None


def relations_to_dataframe(relations: list[Relation]) -> pd.DataFrame:
    """Serialize relations for CSV storage, with attributes stored as JSON."""
    rows = []
    for relation in relations:
        row = relation.model_dump()
        row["attributes"] = json.dumps(row["attributes"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_relations(
    relations_df: pd.DataFrame,
    path=config.PROCESSED_DATA_DIR / "relations.csv",
) -> None:
    """Save validated relationships."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    relations_df.to_csv(path, index=False)
    print(f"[RELATION] Saved {path}")


def _entities_to_nodes(graph: nx.DiGraph, entities: pd.DataFrame) -> None:
    for _, row in entities.iterrows():
        aliases = row["aliases"].split("|") if isinstance(row["aliases"], str) else []
        graph.add_node(
            row["entity_id"],
            entity_id=row["entity_id"],
            canonical_name=row["canonical_name"],
            entity_type=row["entity_type"],
            aliases=aliases,
            mention_count=int(row["mention_count"]),
        )


def _relation_attrs(attributes) -> dict:
    """Accept either in-memory dicts or JSON read back from the CSV."""
    if isinstance(attributes, str):
        return json.loads(attributes)
    return dict(attributes)


def build_graph(
    entities: pd.DataFrame,
    relations: list[Relation] | pd.DataFrame,
) -> nx.DiGraph:
    """Build a directed graph while preserving one edge per source record."""
    graph = nx.DiGraph()
    _entities_to_nodes(graph, entities)
    relation_rows = (
        relations.to_dict(orient="records")
        if isinstance(relations, pd.DataFrame)
        else [relation.model_dump() for relation in relations]
    )

    skipped = 0
    for relation in relation_rows:
        source_id = relation["source_entity_id"]
        target_id = relation["target_entity_id"]
        if source_id not in graph or target_id not in graph:
            skipped += 1
            continue
        edge_attributes = {
            "relationship": relation["relationship"],
            "record_id": relation["record_id"],
            "evidence": relation["evidence"],
            **_relation_attrs(relation["attributes"]),
        }
        graph.add_edge(source_id, target_id, **edge_attributes)

    if skipped:
        print(f"[GRAPH] Skipped {skipped} relations with an unresolved entity endpoint")
    print(f"[GRAPH] Nodes: {graph.number_of_nodes():,}")
    print(f"[GRAPH] Edges: {graph.number_of_edges():,}")
    print(f"[GRAPH] Isolated nodes (no edges): {nx.number_of_isolates(graph):,}")
    return graph


def save_graph(graph: nx.DiGraph, path=config.GRAPH_FILE) -> None:
    """Save the graph as a pickle file."""
    config.GRAPH_DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
        pickle.dump(graph, file)
    print(f"[GRAPH] Saved graph to {path}")


def load_graph(path=config.GRAPH_FILE) -> nx.DiGraph:
    """Load a previously built graph."""
    if not path.exists():
        raise FileNotFoundError(f"No saved graph at {path}. Run `python build_graph.py` first.")
    with open(path, "rb") as file:
        graph = pickle.load(file)
    print(
        f"[GRAPH] Loaded graph from {path} "
        f"({graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges)"
    )
    return graph


def build_knowledge_graph(max_rows_per_file: int | None = None) -> nx.DiGraph:
    """Run the complete load-to-graph pipeline and save its intermediate outputs."""
    raw_interactions, raw_indications = load_raw_data(max_rows_per_file)
    interactions = clean_interactions(raw_interactions)
    indications = clean_indications(raw_indications)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(config.PROCESSED_INTERACTIONS_FILE, index=False)
    indications.to_csv(config.PROCESSED_INDICATIONS_FILE, index=False)

    entities, interactions, indications = link_entities(interactions, indications)
    save_entities(entities)
    relations = extract_structured_relations(interactions, indications)
    save_relations(relations_to_dataframe(relations))

    graph = build_graph(entities, relations)
    save_graph(graph)
    return graph
