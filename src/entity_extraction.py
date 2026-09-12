"""
Identifies entity mentions (drug/disease names) inside the processed
record tables.

The DDInter and PharmacotherapyDB sources already give us structured,
labeled drug/disease names in dedicated columns, so extraction here is a
lookup, not NLP: every row in `interactions_clean.csv` yields two DRUG
mentions, every row in `indications_clean.csv` yields one DRUG mention and
one DISEASE mention. Per the project brief, running an LLM/NER model over
already-structured fields would add cost and risk for no benefit.

`extract_entities_from_text()` is kept as a documented extension point so
that a future milestone can plug in ScispaCy/UMLS NER over free-text
fields (e.g. scraped DDInter mechanism text) without changing how the rest
of the pipeline consumes mentions - it just needs to return the same
(raw_name, entity_type) shape as the structured path below.
"""

import pandas as pd

from src import config


def extract_mentions(interactions: pd.DataFrame, indications: pd.DataFrame) -> pd.DataFrame:
    """Flatten both processed tables into one long table of entity mentions.

    Returns:
        DataFrame with columns: record_id, raw_name, entity_type.
        One row per (record, entity) occurrence - a drug pair record
        contributes 2 rows, an indication record contributes 2 rows.
    """
    mentions = []

    for _, row in interactions.iterrows():
        mentions.append((row["record_id"], row["source_entity"], config.ENTITY_DRUG))
        mentions.append((row["record_id"], row["target_entity"], config.ENTITY_DRUG))

    for _, row in indications.iterrows():
        mentions.append((row["record_id"], row["source_entity"], config.ENTITY_DRUG))
        mentions.append((row["record_id"], row["target_entity"], config.ENTITY_DISEASE))

    mentions_df = pd.DataFrame(mentions, columns=["record_id", "raw_name", "entity_type"])
    print(f"[ENTITY] Extracted {len(mentions_df):,} entity mentions "
          f"({(mentions_df['entity_type'] == config.ENTITY_DRUG).sum():,} drug, "
          f"{(mentions_df['entity_type'] == config.ENTITY_DISEASE).sum():,} disease)")
    return mentions_df


def extract_entities_from_text(text: str) -> list[tuple[str, str]]:
    """Extension point for NLP-based extraction over unstructured text.

    Not used by the current pipeline (both data sources are already
    structured). A future milestone could implement this with ScispaCy's
    biomedical NER model and feed the result into the same
    `build_entity_table()` step used for structured mentions, keeping the
    rest of the pipeline unchanged.
    """
    raise NotImplementedError(
        "No unstructured text fields are used in the current pipeline. "
        "Implement with ScispaCy/UMLS if free-text sources are added later."
    )


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.entity_extraction`
    interactions_df = pd.read_csv(config.PROCESSED_INTERACTIONS_FILE)
    indications_df = pd.read_csv(config.PROCESSED_INDICATIONS_FILE)
    mentions_df = extract_mentions(interactions_df, indications_df)
    print(mentions_df.head(10))
