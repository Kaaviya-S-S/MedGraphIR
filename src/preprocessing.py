"""
Cleans the raw DDInter interactions and PharmacotherapyDB indications into
two flat, record-level tables ready for entity extraction (Milestone 2).

This module does ROW-LEVEL cleaning only (trimming, dropping invalid/
duplicate rows, mapping raw fields onto the controlled relationship
vocabulary). Canonical entity normalization (e.g. deciding that "Metformin"
and " metformin " are the same node) happens later in entity_extraction.py.

Design decisions (see README data dictionary for full detail):
  - DDInter rows are inherently symmetric drug-drug interactions with a
    severity label (Major/Moderate/Minor/Unknown). They map to a single
    controlled relationship: INTERACTS_WITH, with severity kept as an edge
    attribute.
  - The same (DDInterID_A, DDInterID_B) pair sometimes appears in more than
    one ATC-code file (~28% of raw rows). These are exact duplicates of the
    same fact and are dropped, keeping the first occurrence.
  - PharmacotherapyDB rows with category "NOT" are physician-reviewed
    NEGATIVE indications (confirmed the drug does NOT treat the disease).
    They are dropped rather than forced into the TREATS vocabulary, since
    keeping them would assert the opposite of what the source says.
  - PharmacotherapyDB categories DM (disease-modifying) and SYM (symptomatic)
    both map to relationship=TREATS; the original category is preserved as
    an `indication_category` attribute so the distinction isn't lost.
"""

import pandas as pd

from src import config
from src.data_loader import load_raw_data


def clean_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning of DDInter drug-drug interaction records."""
    df = raw.copy()

    for col in ("Drug_A", "Drug_B"):
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[(df["Drug_A"] != "") & (df["Drug_B"] != "")]
    dropped_empty = before - len(df)

    before = len(df)
    df = df.drop_duplicates(subset=["DDInterID_A", "DDInterID_B"], keep="first")
    dropped_dupes = before - len(df)

    df["record_id"] = "ddinter_" + df["DDInterID_A"] + "_" + df["DDInterID_B"]
    df["relationship"] = config.REL_INTERACTS_WITH
    df["severity"] = df["Level"]

    clean = df[
        [
            "record_id",
            "Drug_A",
            "Drug_B",
            "relationship",
            "severity",
            "atc_code",
            "atc_label",
        ]
    ].rename(columns={"Drug_A": "source_entity", "Drug_B": "target_entity"})

    print(f"[PREPROCESSING] Interactions: dropped {dropped_empty} empty-name rows")
    print(f"[PREPROCESSING] Interactions: dropped {dropped_dupes} duplicate (drug pair) rows")
    print(f"[PREPROCESSING] Interactions: {len(clean):,} usable records")
    return clean.reset_index(drop=True)


def clean_indications(raw: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning of PharmacotherapyDB drug-disease indication records."""
    df = raw.copy()

    for col in ("drug", "disease"):
        df[col] = df[col].astype(str).str.strip()

    before = len(df)
    df = df[df["category"] != "NOT"]
    dropped_negative = before - len(df)

    df["record_id"] = "pharmdb_" + df["drugbank_id"] + "_" + df["doid_id"]
    df["relationship"] = config.REL_TREATS
    df["indication_category"] = df["category"]

    clean = df[
        [
            "record_id",
            "drug",
            "disease",
            "relationship",
            "indication_category",
            "n_curators",
            "n_resources",
        ]
    ].rename(columns={"drug": "source_entity", "disease": "target_entity"})

    print(f"[PREPROCESSING] Indications: dropped {dropped_negative} confirmed non-indication ('NOT') rows")
    print(f"[PREPROCESSING] Indications: {len(clean):,} usable records")
    return clean.reset_index(drop=True)


def preprocess_all(max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full load -> clean -> save pipeline for both data sources."""
    raw_interactions, raw_indications = load_raw_data(max_rows_per_file)

    interactions = clean_interactions(raw_interactions)
    indications = clean_indications(raw_indications)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(config.PROCESSED_INTERACTIONS_FILE, index=False)
    indications.to_csv(config.PROCESSED_INDICATIONS_FILE, index=False)

    print(f"[PREPROCESSING] Saved {config.PROCESSED_INTERACTIONS_FILE}")
    print(f"[PREPROCESSING] Saved {config.PROCESSED_INDICATIONS_FILE}")

    return interactions, indications


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.preprocessing`
    interactions_df, indications_df = preprocess_all()
    print("\nProcessed interactions sample:")
    print(interactions_df.head())
    print("\nProcessed indications sample:")
    print(indications_df.head())
