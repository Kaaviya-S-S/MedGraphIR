"""
Normalizes raw entity mentions into canonical graph nodes.
  - lowercase
  - strip leading/trailing whitespace
  - collapse internal whitespace runs to a single space

Punctuation is intentionally KEPT rather than stripped. Inspecting the
actual data showed that parenthetical qualifiers are semantically load-
bearing, not noise - e.g. "Amphotericin B (liposomal)" and "Amphotericin B
(cholesteryl sulfate)" are different drug formulations with different
interaction profiles. Stripping "(...)" would silently merge them into one
incorrect node. Apostrophes/hyphens ("Alzheimer's disease",
"anti-inhibitor coagulant complex") are similarly part of the name itself.

This module is the seam where a future UMLS/ScispaCy entity linker would
plug in: `normalize_text()` could be replaced/augmented with a call that
maps a raw name to a UMLS CUI, and `make_entity_id()` could use that CUI
instead of a text slug, without touching the rest of the pipeline.
"""

import re

import pandas as pd

from src import config
from src.entity_extraction import extract_mentions


def normalize_text(name: str) -> str:
    """Canonicalization key used to decide whether two raw names are the
    same entity. Case- and whitespace-insensitive, punctuation-preserving.
    """
    return re.sub(r"\s+", " ", name.strip()).lower()


def slugify(normalized_name: str) -> str:
    """Turn a normalized name into a plain identifier-safe string."""
    slug = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return slug or "unknown"


def make_entity_id(entity_type: str, normalized_name: str) -> str:
    """Stable graph node key, e.g. 'drug::metformin', 'disease::alzheimer_s_disease'."""
    return f"{entity_type.lower()}::{slugify(normalized_name)}"


def build_entity_table(mentions: pd.DataFrame) -> pd.DataFrame:
    """Collapse raw mentions into one row per canonical entity.

    Args:
        mentions: output of entity_extraction.extract_mentions()
            (columns: record_id, raw_name, entity_type).

    Returns:
        DataFrame with columns: entity_id, canonical_name, entity_type,
        aliases (pipe-joined raw variants seen), mention_count.
    """
    df = mentions.copy()
    df["normalized_name"] = df["raw_name"].map(normalize_text)

    rows = []
    grouped = df.groupby(["entity_type", "normalized_name"])
    for (entity_type, normalized_name), group in grouped:
        variant_counts = group["raw_name"].value_counts()
        canonical_name = variant_counts.index[0]  # most frequently seen casing/variant
        aliases = sorted(variant_counts.index.tolist())

        rows.append(
            {
                "entity_id": make_entity_id(entity_type, normalized_name),
                "canonical_name": canonical_name,
                "entity_type": entity_type,
                "normalized_name": normalized_name,
                "aliases": "|".join(aliases),
                "mention_count": len(group),
            }
        )

    entities = pd.DataFrame(rows).sort_values("entity_id").reset_index(drop=True)
    print(f"[ENTITY] Resolved {len(mentions):,} mentions into {len(entities):,} unique entities "
          f"({(entities['entity_type'] == config.ENTITY_DRUG).sum():,} drugs, "
          f"{(entities['entity_type'] == config.ENTITY_DISEASE).sum():,} diseases)")

    merged = int((entities["aliases"].str.count(r"\|") > 0).sum())
    if merged:
        print(f"[ENTITY] {merged} entities merged multiple raw name variants into one canonical node")

    return entities


def build_alias_lookup(entities: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map (normalized_alias, entity_type) -> entity_id, for resolving any
    raw name (from a data row or a user query) to its canonical node.
    """
    lookup: dict[tuple[str, str], str] = {}
    for _, row in entities.iterrows():
        entity_type = row["entity_type"]
        for alias in row["aliases"].split("|"):
            lookup[(normalize_text(alias), entity_type)] = row["entity_id"]
        lookup[(row["normalized_name"], entity_type)] = row["entity_id"]
    return lookup


def resolve_entity_id(raw_name: str, entity_type: str, alias_lookup: dict[tuple[str, str], str]) -> str | None:
    """Look up the canonical entity_id for a raw name, or None if unseen."""
    return alias_lookup.get((normalize_text(raw_name), entity_type))


def link_entities(
    interactions: pd.DataFrame, indications: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Full Milestone 2 pipeline: extract mentions, build the canonical
    entity table, and attach resolved entity_id columns to both record
    tables (used by graph_builder.py in Milestone 4 to create edges
    between canonical nodes instead of raw strings).

    Returns:
        (entities, interactions_with_ids, indications_with_ids)
    """
    mentions = extract_mentions(interactions, indications)
    entities = build_entity_table(mentions)
    alias_lookup = build_alias_lookup(entities)

    interactions = interactions.copy()
    interactions["source_entity_id"] = interactions["source_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, alias_lookup)
    )
    interactions["target_entity_id"] = interactions["target_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, alias_lookup)
    )

    indications = indications.copy()
    indications["source_entity_id"] = indications["source_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, alias_lookup)
    )
    indications["target_entity_id"] = indications["target_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DISEASE, alias_lookup)
    )

    return entities, interactions, indications


def save_entities(entities: pd.DataFrame) -> None:
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    entities.to_csv(config.PROCESSED_ENTITIES_FILE, index=False)
    print(f"[ENTITY] Saved {config.PROCESSED_ENTITIES_FILE}")


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.entity_linking`
    interactions_df = pd.read_csv(config.PROCESSED_INTERACTIONS_FILE)
    indications_df = pd.read_csv(config.PROCESSED_INDICATIONS_FILE)

    entities_df, interactions_df, indications_df = link_entities(interactions_df, indications_df)
    save_entities(entities_df)

    print("\nEntities with multiple merged aliases (proof normalization did something):")
    multi_alias = entities_df[entities_df["aliases"].str.contains(r"\|", regex=True)]
    print(multi_alias[["entity_id", "canonical_name", "aliases", "mention_count"]].head(10).to_string())

    print("\nAny unresolved entity_ids after linking (should be none)?")
    print(
        "interactions:", interactions_df[["source_entity_id", "target_entity_id"]].isnull().sum().sum(),
        "| indications:", indications_df[["source_entity_id", "target_entity_id"]].isnull().sum().sum(),
    )
