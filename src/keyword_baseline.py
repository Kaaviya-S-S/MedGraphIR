"""
Traditional (non-graph) IR baseline: TF-IDF + cosine similarity keyword
search over the same relation records the knowledge graph was built from.

This exists specifically to give Milestone 7 something meaningful to
compare graph retrieval against - the entire point of this being an IR
project rather than just "a graph with a search box" (see README section
6, Milestone 6). It deliberately ignores the graph structure entirely:
each relation record (one drug-drug interaction or drug-disease indication
row) is treated as an independent bag-of-words document. A query can only
surface a record if that record's OWN text contains matching/related
terms - there is no notion of "connected" entities and no multi-hop
reasoning, which is exactly the capability graph retrieval is expected to
win on.

TF-IDF was chosen (over naive substring matching) because it's the
standard classical-IR keyword baseline: it weights rare/distinctive terms
higher than common ones and produces a ranked result via cosine
similarity, so it's a fair, ranked comparison point rather than a
strawman.
"""

import json

import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src import config


class KeywordResult(BaseModel):
    record_id: str
    source_entity: str
    source_entity_id: str
    relationship: str
    target_entity: str
    target_entity_id: str
    evidence: str
    attributes: dict = {}
    score: float


def _parse_attributes(value) -> dict:
    """relations.csv stores `attributes` as a JSON string (see
    relation_extraction.relations_to_dataframe) - parse it back to a dict.
    """
    if isinstance(value, str):
        return json.loads(value) if value else {}
    return dict(value) if value else {}


def _record_to_document(row: pd.Series) -> str:
    """Flatten one relation record into a single bag-of-words document.
    Underscores in the relationship label are split into words so
    "INTERACTS_WITH" contributes the term "interacts", not one opaque
    token.
    """
    parts = [
        row["source_entity"],
        str(row["relationship"]).replace("_", " "),
        row["target_entity"],
        row["evidence"],
    ]
    return " ".join(str(p) for p in parts)


class KeywordRetriever:
    """Fit once on the relations table, then answer many queries cheaply -
    mirrors retrieval.load_retrieval_engine()'s "build once, query many"
    shape so the two approaches can be timed and compared fairly in
    evaluate.py.
    """

    def __init__(self, relations: pd.DataFrame):
        self.relations = relations.reset_index(drop=True)
        documents = [_record_to_document(row) for _, row in self.relations.iterrows()]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(documents)
        print(
            f"[KEYWORD] Indexed {len(documents):,} records into a "
            f"{self.doc_matrix.shape[1]:,}-term TF-IDF matrix"
        )

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> list[KeywordResult]:
        """Rank all records by cosine similarity to the query's TF-IDF
        vector; return the top_k with a non-zero score (score 0 means no
        query term appeared in the document at all).
        """
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix).flatten()

        results = []
        for idx in scores.argsort()[::-1]:
            if scores[idx] <= 0:
                break
            row = self.relations.iloc[idx]
            results.append(
                KeywordResult(
                    record_id=row["record_id"],
                    source_entity=row["source_entity"],
                    source_entity_id=row["source_entity_id"],
                    relationship=row["relationship"],
                    target_entity=row["target_entity"],
                    target_entity_id=row["target_entity_id"],
                    evidence=row["evidence"],
                    attributes=_parse_attributes(row.get("attributes", {})),
                    score=round(float(scores[idx]), 4),
                )
            )
            if len(results) >= top_k:
                break
        return results


def load_keyword_baseline(path=config.PROCESSED_DATA_DIR / "relations.csv") -> KeywordRetriever:
    """Convenience loader, parity with retrieval.load_retrieval_engine()."""
    relations = pd.read_csv(path)
    return KeywordRetriever(relations)


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.keyword_baseline`
    retriever = load_keyword_baseline()

    demo_queries = [
        "What drugs interact with Warfarin?",
        "What diseases are associated with Metformin?",
        "What drugs are related to Diabetes?",
        "xyzabc123",
    ]
    for q in demo_queries:
        results = retriever.search(q, top_k=8)
        print(f'\n=== "{q}" ===')
        if not results:
            print("No keyword matches found.")
            continue
        for r in results:
            print(f"  ({r.score}) {r.source_entity} -[{r.relationship}]-> {r.target_entity}")
