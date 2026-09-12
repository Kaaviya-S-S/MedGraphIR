"""
Central configuration for the Medical Knowledge Graph IR System.

Every path and tunable constant used across the project lives here so that
nothing is scattered/hardcoded inside individual modules (per project
requirement: "Do not scatter these values throughout the code").
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
GRAPH_DATA_DIR = DATA_DIR / "graph"

# DDInter 2.0 bulk drug-drug interaction files (one CSV per ATC top-level code).
# Downloaded from https://ddinter2.scbdd.com/download/
DDINTER_ATC_CODES = ["A", "B", "D", "H", "L", "P", "R", "V"]
DDINTER_FILE_TEMPLATE = "ddinter_downloads_code_{code}.csv"

# ATC top-level code -> human readable label (used for logging/provenance only).
ATC_CODE_LABELS = {
    "A": "Alimentary tract and metabolism",
    "B": "Blood and blood forming organs",
    "D": "Dermatologicals",
    "H": "Systemic hormonal preparations (excl. sex hormones/insulins)",
    "L": "Antineoplastic and immunomodulating agents",
    "P": "Antiparasitic products, insecticides and repellents",
    "R": "Respiratory system",
    "V": "Various",
}

# PharmacotherapyDB 1.0 drug-disease indications (physician curated).
# Downloaded from https://github.com/dhimmel/indications
INDICATIONS_FILE = RAW_DATA_DIR / "pharmacotherapydb_indications.tsv"

# Processed (cleaned + normalized) output tables.
PROCESSED_INTERACTIONS_FILE = PROCESSED_DATA_DIR / "interactions_clean.csv"
PROCESSED_INDICATIONS_FILE = PROCESSED_DATA_DIR / "indications_clean.csv"
PROCESSED_ENTITIES_FILE = PROCESSED_DATA_DIR / "entities.csv"

# Saved knowledge graph (built once by build_graph.py, loaded by app.py).
# Plain Python pickle - networkx dropped its own write_gpickle/read_gpickle
# helpers in 3.0, and a NetworkX graph pickles safely on its own.
GRAPH_FILE = GRAPH_DATA_DIR / "medical_kg.pkl"

# ---------------------------------------------------------------------------
# Development sampling
# ---------------------------------------------------------------------------
# Cap on how many raw interaction rows to load per DDInter file while
# iterating on the pipeline. Set to None to use the FULL dataset (recommended
# for the final build; the full dataset is only ~222k rows and NetworkX
# handles that easily).
MAX_ROWS_PER_FILE: int | None = 2000

# ---------------------------------------------------------------------------
# Relation vocabulary (controlled — see Milestone 3)
# ---------------------------------------------------------------------------
REL_INTERACTS_WITH = "INTERACTS_WITH"
REL_TREATS = "TREATS"
REL_CONTRAINDICATED_IN = "CONTRAINDICATED_IN"
REL_ASSOCIATED_WITH = "ASSOCIATED_WITH"
REL_CAUTION_IN = "CAUTION_IN"

VALID_RELATIONS = {
    REL_INTERACTS_WITH,
    REL_TREATS,
    REL_CONTRAINDICATED_IN,
    REL_ASSOCIATED_WITH,
    REL_CAUTION_IN,
}

# Entity types
ENTITY_DRUG = "DRUG"
ENTITY_DISEASE = "DISEASE"

# ---------------------------------------------------------------------------
# Graph retrieval
# ---------------------------------------------------------------------------
MAX_HOPS = 2
TOP_K_RESULTS = 20

# Caps how many 1-hop neighbors are used as "seeds" for 2-hop expansion.
# Without this, a high-degree node (e.g. Warfarin has 800+ neighbors) would
# make 2-hop expansion touch a huge fraction of the graph. Seeds are chosen
# by highest 1-hop rank_score first, so expansion favors the clinically
# most relevant neighbors.
TWO_HOP_MAX_SEED_NODES = 10

# difflib.SequenceMatcher ratio threshold for the fuzzy-match fallback tier.
FUZZY_MATCH_CUTOFF = 0.72
# Cap on n-gram length (in words) tried during fuzzy matching, to keep it fast.
FUZZY_NGRAM_MAX_WORDS = 4

# ---------------------------------------------------------------------------
# Hybrid retrieval (TF-IDF lexical relevance + graph structural relevance)
# ---------------------------------------------------------------------------
# HybridScore = HYBRID_ALPHA * normalized_tfidf_score + HYBRID_BETA * normalized_graph_score
HYBRID_ALPHA_DEFAULT = 0.5
HYBRID_BETA_DEFAULT = 0.5
# Each underlying method (TF-IDF, graph) is queried for this many candidates
# BEFORE merging/re-ranking - wider than TOP_K_RESULTS so a candidate that
# ranks modestly in one method but well in the other still has a chance to
# surface after the two scores are combined, rather than being cut off by
# either method's own top-K truncation first.
HYBRID_CANDIDATE_POOL_K = 50

# ---------------------------------------------------------------------------
# Ollama (optional local LLM, used only in later milestones)
# ---------------------------------------------------------------------------
OLLAMA_MODEL = "llama3.2"
OLLAMA_BASE_URL = "http://localhost:11434"
