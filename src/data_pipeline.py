"""Load, clean, and link the source data used by MedGraphIR."""

import re

import pandas as pd

from src import config


def load_ddinter_interactions(
    max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE,
) -> pd.DataFrame:
    """Load the DDInter CSV files and keep their source-category labels."""
    frames = []
    for code in config.DDINTER_ATC_CODES:
        path = config.RAW_DATA_DIR / config.DDINTER_FILE_TEMPLATE.format(code=code)
        if not path.exists():
            print(f"[DATA] WARNING: missing DDInter file {path.name}, skipping")
            continue

        frame = pd.read_csv(path, nrows=max_rows_per_file)
        frame["atc_code"] = code
        frame["atc_label"] = config.ATC_CODE_LABELS.get(code, "Unknown")
        frames.append(frame)
        print(f"[DATA] Loaded {len(frame):,} rows from {path.name} ({config.ATC_CODE_LABELS.get(code)})")

    if not frames:
        raise FileNotFoundError(
            f"No DDInter interaction files found in {config.RAW_DATA_DIR}. "
            "Did you download the CSVs from https://ddinter2.scbdd.com/download/ ?"
        )

    combined = pd.concat(frames, ignore_index=True)
    print(f"[DATA] Total DDInter interaction rows loaded: {len(combined):,}")
    return combined


def load_indications() -> pd.DataFrame:
    """Load the PharmacotherapyDB drug-disease indication table."""
    path = config.INDICATIONS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Indications file not found at {path}. Download it from "
            "https://raw.githubusercontent.com/dhimmel/indications/gh-pages/catalog/indications.tsv"
        )

    frame = pd.read_csv(path, sep="\t")
    print(f"[DATA] Loaded {len(frame):,} drug-disease indication rows from {path.name}")
    return frame


def load_raw_data(
    max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both source datasets."""
    return load_ddinter_interactions(max_rows_per_file), load_indications()


def clean_interactions(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove empty/duplicate DDInter records and label their relationship."""
    frame = raw.copy()
    for column in ("Drug_A", "Drug_B"):
        frame[column] = frame[column].astype(str).str.strip()

    before = len(frame)
    frame = frame[(frame["Drug_A"] != "") & (frame["Drug_B"] != "")]
    dropped_empty = before - len(frame)

    before = len(frame)
    frame = frame.drop_duplicates(subset=["DDInterID_A", "DDInterID_B"], keep="first")
    dropped_duplicates = before - len(frame)

    frame["record_id"] = "ddinter_" + frame["DDInterID_A"] + "_" + frame["DDInterID_B"]
    frame["relationship"] = config.REL_INTERACTS_WITH
    frame["severity"] = frame["Level"]
    clean = frame[
        ["record_id", "Drug_A", "Drug_B", "relationship", "severity", "atc_code", "atc_label"]
    ].rename(columns={"Drug_A": "source_entity", "Drug_B": "target_entity"})

    print(f"[PREPROCESSING] Interactions: dropped {dropped_empty} empty-name rows")
    print(f"[PREPROCESSING] Interactions: dropped {dropped_duplicates} duplicate (drug pair) rows")
    print(f"[PREPROCESSING] Interactions: {len(clean):,} usable records")
    return clean.reset_index(drop=True)


def clean_indications(raw: pd.DataFrame) -> pd.DataFrame:
    """Remove negative indications and keep the evidence for positive ones."""
    frame = raw.copy()
    for column in ("drug", "disease"):
        frame[column] = frame[column].astype(str).str.strip()

    before = len(frame)
    frame = frame[frame["category"] != "NOT"]
    dropped_negative = before - len(frame)

    frame["record_id"] = "pharmdb_" + frame["drugbank_id"] + "_" + frame["doid_id"]
    frame["relationship"] = config.REL_TREATS
    frame["indication_category"] = frame["category"]
    clean = frame[
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


def preprocess_all(
    max_rows_per_file: int | None = config.MAX_ROWS_PER_FILE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load, clean, and save the two record tables."""
    raw_interactions, raw_indications = load_raw_data(max_rows_per_file)
    interactions = clean_interactions(raw_interactions)
    indications = clean_indications(raw_indications)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(config.PROCESSED_INTERACTIONS_FILE, index=False)
    indications.to_csv(config.PROCESSED_INDICATIONS_FILE, index=False)
    print(f"[PREPROCESSING] Saved {config.PROCESSED_INTERACTIONS_FILE}")
    print(f"[PREPROCESSING] Saved {config.PROCESSED_INDICATIONS_FILE}")
    return interactions, indications


def normalize_text(name: str) -> str:
    """Normalize case and whitespace while preserving name punctuation."""
    return re.sub(r"\s+", " ", name.strip()).lower()


def slugify(normalized_name: str) -> str:
    """Convert a normalized name to a stable identifier part."""
    slug = re.sub(r"[^a-z0-9]+", "_", normalized_name).strip("_")
    return slug or "unknown"


def make_entity_id(entity_type: str, normalized_name: str) -> str:
    """Build a graph key such as ``drug::metformin``."""
    return f"{entity_type.lower()}::{slugify(normalized_name)}"


def extract_mentions(interactions: pd.DataFrame, indications: pd.DataFrame) -> pd.DataFrame:
    """List each drug/disease mention found in the two structured tables."""
    mentions = []
    for _, row in interactions.iterrows():
        mentions.extend(
            [
                (row["record_id"], row["source_entity"], config.ENTITY_DRUG),
                (row["record_id"], row["target_entity"], config.ENTITY_DRUG),
            ]
        )
    for _, row in indications.iterrows():
        mentions.extend(
            [
                (row["record_id"], row["source_entity"], config.ENTITY_DRUG),
                (row["record_id"], row["target_entity"], config.ENTITY_DISEASE),
            ]
        )

    columns = ["record_id", "raw_name", "entity_type"]
    mentions_df = pd.DataFrame(mentions, columns=columns)
    print(
        f"[ENTITY] Extracted {len(mentions_df):,} entity mentions "
        f"({(mentions_df['entity_type'] == config.ENTITY_DRUG).sum():,} drug, "
        f"{(mentions_df['entity_type'] == config.ENTITY_DISEASE).sum():,} disease)"
    )
    return mentions_df


def extract_entities_from_text(text: str) -> list[tuple[str, str]]:
    """Reserved extension point for a future biomedical text extractor."""
    raise NotImplementedError(
        "No unstructured text fields are used in the current pipeline. "
        "Implement with ScispaCy/UMLS if free-text sources are added later."
    )


def build_entity_table(mentions: pd.DataFrame) -> pd.DataFrame:
    """Merge case/whitespace variants into one canonical entity per name."""
    frame = mentions.copy()
    frame["normalized_name"] = frame["raw_name"].map(normalize_text)

    rows = []
    for (entity_type, normalized_name), group in frame.groupby(["entity_type", "normalized_name"]):
        variant_counts = group["raw_name"].value_counts()
        rows.append(
            {
                "entity_id": make_entity_id(entity_type, normalized_name),
                "canonical_name": variant_counts.index[0],
                "entity_type": entity_type,
                "normalized_name": normalized_name,
                "aliases": "|".join(sorted(variant_counts.index.tolist())),
                "mention_count": len(group),
            }
        )

    entities = pd.DataFrame(rows).sort_values("entity_id").reset_index(drop=True)
    print(
        f"[ENTITY] Resolved {len(mentions):,} mentions into {len(entities):,} unique entities "
        f"({(entities['entity_type'] == config.ENTITY_DRUG).sum():,} drugs, "
        f"{(entities['entity_type'] == config.ENTITY_DISEASE).sum():,} diseases)"
    )
    merged = int((entities["aliases"].str.count(r"\|") > 0).sum())
    if merged:
        print(f"[ENTITY] {merged} entities merged multiple raw name variants into one canonical node")
    return entities


def build_alias_lookup(entities: pd.DataFrame) -> dict[tuple[str, str], str]:
    """Map normalized names and aliases to their canonical entity IDs."""
    lookup = {}
    for _, row in entities.iterrows():
        entity_type = row["entity_type"]
        for alias in row["aliases"].split("|"):
            lookup[(normalize_text(alias), entity_type)] = row["entity_id"]
        lookup[(row["normalized_name"], entity_type)] = row["entity_id"]
    return lookup


def resolve_entity_id(
    raw_name: str,
    entity_type: str,
    alias_lookup: dict[tuple[str, str], str],
) -> str | None:
    """Return a canonical ID for a known name, or ``None`` if it is unseen."""
    return alias_lookup.get((normalize_text(raw_name), entity_type))


def link_entities(
    interactions: pd.DataFrame,
    indications: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the entity table and add canonical IDs to both record tables."""
    entities = build_entity_table(extract_mentions(interactions, indications))
    aliases = build_alias_lookup(entities)

    interactions = interactions.copy()
    interactions["source_entity_id"] = interactions["source_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, aliases)
    )
    interactions["target_entity_id"] = interactions["target_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, aliases)
    )

    indications = indications.copy()
    indications["source_entity_id"] = indications["source_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DRUG, aliases)
    )
    indications["target_entity_id"] = indications["target_entity"].map(
        lambda name: resolve_entity_id(name, config.ENTITY_DISEASE, aliases)
    )
    return entities, interactions, indications


def save_entities(entities: pd.DataFrame) -> None:
    """Save the canonical entity table."""
    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    entities.to_csv(config.PROCESSED_ENTITIES_FILE, index=False)
    print(f"[ENTITY] Saved {config.PROCESSED_ENTITIES_FILE}")
