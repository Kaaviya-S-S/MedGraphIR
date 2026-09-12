"""
Loads the raw medical datasets from disk.

Two independent, openly downloadable sources are used:

1. DDInter 2.0 (https://ddinter2.scbdd.com/) - bulk drug-drug interaction
   export, split into 8 CSV files by top-level ATC drug category. Each file
   has the SAME 5 columns: DDInterID_A, Drug_A, DDInterID_B, Drug_B, Level.
   There is no disease or free-text mechanism data in this bulk export.

2. PharmacotherapyDB 1.0 (https://github.com/dhimmel/indications) - a small,
   physician-curated catalog of drug-disease indications. Used to supply the
   drug-disease side of the knowledge graph, which DDInter does not provide.

This module only loads and lightly tags raw data - all cleaning/normalization
happens in preprocessing.py.
"""

import pandas as pd

from src import config


def load_ddinter_interactions(max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE) -> pd.DataFrame:
    """Load and concatenate all 8 DDInter ATC-code interaction CSVs.

    Args:
        max_rows_per_file: cap on rows read per CSV file (for fast dev
            iteration). Pass None to load every row.

    Returns:
        A single DataFrame with an added `atc_code` and `atc_label` column
        identifying which source file each row came from (provenance).
    """
    frames = []
    for code in config.DDINTER_ATC_CODES:
        path = config.RAW_DATA_DIR / config.DDINTER_FILE_TEMPLATE.format(code=code)
        if not path.exists():
            print(f"[DATA] WARNING: missing DDInter file {path.name}, skipping")
            continue

        df = pd.read_csv(path, nrows=max_rows_per_file)
        df["atc_code"] = code
        df["atc_label"] = config.ATC_CODE_LABELS.get(code, "Unknown")
        frames.append(df)
        print(f"[DATA] Loaded {len(df):,} rows from {path.name} ({config.ATC_CODE_LABELS.get(code)})")

    if not frames:
        raise FileNotFoundError(
            f"No DDInter interaction files found in {config.RAW_DATA_DIR}. "
            "Did you download the CSVs from https://ddinter2.scbdd.com/download/ ?"
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"[DATA] Total DDInter interaction rows loaded: {len(combined):,}")
    return combined


def load_indications() -> pd.DataFrame:
    """Load the PharmacotherapyDB drug-disease indications table.

    Returns:
        Raw DataFrame with columns: doid_id, drugbank_id, disease, drug,
        category (DM/SYM/NOT), n_curators, n_resources.
    """
    path = config.INDICATIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Indications file not found at {path}. Download it from "
            "https://raw.githubusercontent.com/dhimmel/indications/gh-pages/catalog/indications.tsv"
        )

    df = pd.read_csv(path, sep="\t")
    print(f"[DATA] Loaded {len(df):,} drug-disease indication rows from {path.name}")
    return df


def load_raw_data(max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load both raw sources at once."""
    interactions = load_ddinter_interactions(max_rows_per_file)
    indications = load_indications()
    return interactions, indications


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.data_loader`
    interactions_df, indications_df = load_raw_data()
    print("\nInteractions sample:")
    print(interactions_df.head())
    print("\nIndications sample:")
    print(indications_df.head())
