"""
Turns processed, entity-linked records into validated (source, relationship,
target) triples using a controlled relationship vocabulary.

Two extraction paths, per the project brief's priority rule ("if the
dataset already explicitly provides a relationship, prefer the dataset
relationship over LLM inference"):

1. STRUCTURED (primary, always used): both DDInter interactions and
   PharmacotherapyDB indications already carry an explicit, reliable
   relationship (INTERACTS_WITH / TREATS respectively - assigned back in
   src/preprocessing.py). This path just packages each row into a
   validated Relation object. No LLM involved.

2. LLM-ASSISTED (optional, extension point): for a free-text field where no
   structured relationship exists, extract_relation_with_llm() asks a local
   Ollama model to classify the relationship between two given entities
   from the given text, constrained to the same controlled vocabulary.
   The current pipeline has no free-text field to run this on (DDInter's
   bulk export has no mechanism/description column - see README
   Limitations), so this path is not invoked by default. It exists so the
   architecture matches the brief and can be exercised the moment a
   free-text source is added.

Every Relation is validated by Pydantic against config.VALID_RELATIONS
before it can enter the graph - an LLM (or a bug in the structured path)
can never inject an arbitrary relationship label.
"""

import json

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

from src import config
from src.entity_linking import link_entities


class Relation(BaseModel):
    """A single validated (source, relationship, target) fact with provenance."""

    source_entity: str
    source_entity_id: str
    relationship: str
    target_entity: str
    target_entity_id: str
    record_id: str = Field(description="ID of the source dataset row, for provenance")
    evidence: str = Field(default="", description="Human-readable evidence shown in the UI")
    attributes: dict = Field(default_factory=dict, description="Source-specific structured evidence, e.g. severity")

    @field_validator("relationship")
    @classmethod
    def relationship_must_be_controlled(cls, v: str) -> str:
        if v not in config.VALID_RELATIONS:
            raise ValueError(
                f"'{v}' is not in the controlled relationship vocabulary: {sorted(config.VALID_RELATIONS)}"
            )
        return v


# ---------------------------------------------------------------------------
# 1. Structured extraction (primary path)
# ---------------------------------------------------------------------------

def _interaction_row_to_relation(row: pd.Series) -> Relation:
    return Relation(
        source_entity=row["source_entity"],
        source_entity_id=row["source_entity_id"],
        relationship=row["relationship"],  # INTERACTS_WITH, set in preprocessing.py
        target_entity=row["target_entity"],
        target_entity_id=row["target_entity_id"],
        record_id=row["record_id"],
        evidence=f"DDInter severity: {row['severity']} (ATC category: {row['atc_label']})",
        attributes={"severity": row["severity"], "atc_code": row["atc_code"]},
    )


def _indication_row_to_relation(row: pd.Series) -> Relation:
    category_label = {"DM": "disease-modifying", "SYM": "symptomatic relief"}.get(
        row["indication_category"], row["indication_category"]
    )
    return Relation(
        source_entity=row["source_entity"],
        source_entity_id=row["source_entity_id"],
        relationship=row["relationship"],  # TREATS, set in preprocessing.py
        target_entity=row["target_entity"],
        target_entity_id=row["target_entity_id"],
        record_id=row["record_id"],
        evidence=(
            f"PharmacotherapyDB indication ({category_label}), confirmed by "
            f"{row['n_curators']} curator(s) using {row['n_resources']} resource(s)"
        ),
        attributes={
            "indication_category": row["indication_category"],
            "n_curators": int(row["n_curators"]),
            "n_resources": int(row["n_resources"]),
        },
    )


def extract_structured_relations(
    interactions: pd.DataFrame, indications: pd.DataFrame
) -> list[Relation]:
    """Build validated Relation objects directly from the (already entity-
    linked) processed tables. Rows must already carry source_entity_id /
    target_entity_id - see entity_linking.link_entities().
    """
    relations: list[Relation] = []
    failed = 0

    for _, row in interactions.iterrows():
        try:
            relations.append(_interaction_row_to_relation(row))
        except ValidationError:
            failed += 1

    for _, row in indications.iterrows():
        try:
            relations.append(_indication_row_to_relation(row))
        except ValidationError:
            failed += 1

    print(f"[RELATION] Built {len(relations):,} validated relations from structured data "
          f"({failed} rejected by validation)")
    return relations


# ---------------------------------------------------------------------------
# 2. LLM-assisted extraction (optional extension point, not used by default)
# ---------------------------------------------------------------------------

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
    """Ask a local Ollama model to classify the relationship expressed in
    `text` between two already-identified entities. Returns None (never
    raises) if Ollama is unreachable, the response isn't valid JSON, or the
    returned label isn't in the controlled vocabulary - the caller can
    always safely fall back to "no relation extracted" for that row.

    Not used by the current pipeline (no free-text source is available yet
    - see module docstring), but kept ready for when one is added.
    """
    try:
        import ollama  # local import: this optional path shouldn't make the
                        # `ollama` package a hard dependency for the rest of the app
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
        payload = json.loads(response["response"])
        label = payload.get("relationship", "NONE")
    except Exception as exc:  # Ollama not running, model missing, bad JSON, etc.
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
            evidence=f"LLM-inferred from text: \"{text}\"",
            attributes={"extraction_method": "llm", "model": model},
        )
    except ValidationError as exc:
        print(f"[RELATION] LLM output failed validation, discarding: {exc}")
        return None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def relations_to_dataframe(relations: list[Relation]) -> pd.DataFrame:
    """Serialize Relation objects to a DataFrame ready for CSV storage.

    `attributes` is JSON-encoded as a string (rather than left as a dict)
    because pandas writes dicts to CSV using Python repr (single-quoted),
    which json.loads() cannot parse back. Downstream readers should use
    json.loads() on this column.
    """
    rows = []
    for r in relations:
        row = r.model_dump()
        row["attributes"] = json.dumps(row["attributes"])
        rows.append(row)
    return pd.DataFrame(rows)


def save_relations(relations_df: pd.DataFrame, path=config.PROCESSED_DATA_DIR / "relations.csv") -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    relations_df.to_csv(path, index=False)
    print(f"[RELATION] Saved {path}")


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.relation_extraction`
    interactions_df = pd.read_csv(config.PROCESSED_INTERACTIONS_FILE)
    indications_df = pd.read_csv(config.PROCESSED_INDICATIONS_FILE)
    _entities_df, interactions_df, indications_df = link_entities(interactions_df, indications_df)

    relations = extract_structured_relations(interactions_df, indications_df)
    relations_df = relations_to_dataframe(relations)
    save_relations(relations_df)

    print("\nRelationship type counts:")
    print(relations_df["relationship"].value_counts())

    print("\nSample relation:")
    print(relations[0].model_dump_json(indent=2))

    print("\nProving the controlled vocabulary is actually enforced "
          "(attempting an invalid label 'CURES'):")
    try:
        Relation(
            source_entity="X", source_entity_id="drug::x",
            relationship="CURES",
            target_entity="Y", target_entity_id="disease::y",
            record_id="test",
        )
        print("ERROR: invalid label was NOT rejected!")
    except ValidationError as exc:
        print("Correctly rejected invalid label:", exc.errors()[0]["msg"])
