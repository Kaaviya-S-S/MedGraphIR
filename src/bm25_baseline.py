"""Okapi BM25 lexical retrieval over the same relation records as the graph.

The lexical (non-graph) baseline: each relation record is treated as an
independent bag-of-words document and ranked by BM25, which weights rare
terms higher than common ones and normalizes for document length. It has no
notion of connected entities or relationship type - which is exactly the
capability graph retrieval is expected to win on.
"""

import json
import math
import re
from collections import Counter, defaultdict

import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src import config
from src.retrieval import EntityIndex, RetrievedFact

_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
_STOP_WORDS = frozenset(ENGLISH_STOP_WORDS)


class LexicalResult(BaseModel):
    record_id: str
    source_entity: str
    source_entity_id: str
    relationship: str
    target_entity: str
    target_entity_id: str
    evidence: str
    attributes: dict = {}
    score: float


def _tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_PATTERN.findall(text.lower()) if token not in _STOP_WORDS]


def parse_attributes(value) -> dict:
    """relations.csv stores `attributes` as a JSON string (see
    relation_extraction.relations_to_dataframe) - parse it back to a dict.
    """
    if isinstance(value, str):
        return json.loads(value) if value else {}
    return dict(value) if value else {}


def record_to_document(row: pd.Series) -> str:
    """Flatten one relation record into a single bag-of-words document.
    Underscores in the relationship label are split into words so
    "INTERACTS_WITH" contributes the term "interacts", not one opaque token.
    """
    parts = [
        row["source_entity"],
        str(row["relationship"]).replace("_", " "),
        row["target_entity"],
        row["evidence"],
    ]
    return " ".join(str(p) for p in parts)


class BM25Retriever:
    """Fit a BM25 inverted index once, then rank relation records per query."""

    def __init__(
        self,
        relations: pd.DataFrame,
        k1: float = config.BM25_K1,
        b: float = config.BM25_B,
    ):
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")

        self.relations = relations.reset_index(drop=True)
        self.k1 = k1
        self.b = b
        self.document_lengths: list[int] = []
        self.document_frequencies: dict[str, int] = {}
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_id, (_, row) in enumerate(self.relations.iterrows()):
            term_frequencies = Counter(_tokenize(record_to_document(row)))
            self.document_lengths.append(sum(term_frequencies.values()))
            for term, frequency in term_frequencies.items():
                self.postings[term].append((doc_id, frequency))
                self.document_frequencies[term] = self.document_frequencies.get(term, 0) + 1

        self.document_count = len(self.document_lengths)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count if self.document_count else 0.0
        )
        print(
            f"[BM25] Indexed {self.document_count:,} records across "
            f"{len(self.document_frequencies):,} terms"
        )

    def search(self, query: str, top_k: int = config.TOP_K_RESULTS) -> list[LexicalResult]:
        if top_k <= 0 or not self.document_count:
            return []

        query_terms = set(_tokenize(query))
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            postings = self.postings.get(term)
            if not postings:
                continue

            document_frequency = self.document_frequencies[term]
            inverse_document_frequency = math.log(
                1 + (self.document_count - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for doc_id, term_frequency in postings:
                length_normalizer = 1 - self.b + self.b * (
                    self.document_lengths[doc_id] / self.average_document_length
                )
                scores[doc_id] += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1)
                    / (term_frequency + self.k1 * length_normalizer)
                )

        ranked_doc_ids = sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]
        results = []
        for doc_id in ranked_doc_ids:
            score = scores[doc_id]
            if score <= 0:
                continue
            row = self.relations.iloc[doc_id]
            results.append(
                LexicalResult(
                    record_id=row["record_id"],
                    source_entity=row["source_entity"],
                    source_entity_id=row["source_entity_id"],
                    relationship=row["relationship"],
                    target_entity=row["target_entity"],
                    target_entity_id=row["target_entity_id"],
                    evidence=row["evidence"],
                    attributes=parse_attributes(row.get("attributes", {})),
                    score=round(float(score), 4),
                )
            )
        return results


def lexical_result_to_fact(result: LexicalResult, index: EntityIndex, score: float) -> RetrievedFact:
    """Adapt a lexical result into the RetrievalResult/RetrievedFact shape so
    the UI (graph viz, evidence table, AI summary) needs no special-casing.
    hop=0 is a sentinel meaning "not from graph traversal" - lexical search
    never resolves an entity or traverses edges. Canonical display names come
    from the entity index so lexical results are visually consistent with
    pure-graph facts even when the underlying row used a different casing.
    """
    source_name = index.node_info.get(result.source_entity_id, {}).get("canonical_name", result.source_entity)
    target_name = index.node_info.get(result.target_entity_id, {}).get("canonical_name", result.target_entity)
    return RetrievedFact(
        source_entity_id=result.source_entity_id,
        source_name=source_name,
        relationship=result.relationship,
        target_entity_id=result.target_entity_id,
        target_name=target_name,
        hop=0,
        record_id=result.record_id,
        evidence=result.evidence,
        attributes=result.attributes,
        rank_score=score,
    )


def load_bm25_baseline(path=config.PROCESSED_DATA_DIR / "relations.csv") -> BM25Retriever:
    return BM25Retriever(pd.read_csv(path))


if __name__ == "__main__":
    retriever = load_bm25_baseline()
    for query in (
        "What drugs interact with Warfarin?",
        "What diseases are associated with Metformin?",
        "xyzabc123",
    ):
        print(f'\n=== "{query}" ===')
        for result in retriever.search(query, top_k=8):
            print(f"  ({result.score}) {result.source_entity} -[{result.relationship}]-> {result.target_entity}")
