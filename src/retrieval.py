"""
The core IR component: turns a natural-language query into a matched graph
entity and a ranked, 1-2 hop subgraph of relevant facts.

Pipeline (mirrors the project brief exactly):
    query normalization -> entity matching -> graph node identification
    -> 1-hop retrieval -> optional 2-hop retrieval -> ranked results

Entity matching is four transparent, explainable tiers (no ML, no vector
search) - each tier only runs if the previous one found nothing:

  1. EXACT     - the whole normalized query matches a known entity alias.
  2. SUBSTRING - a known entity name/alias appears as a contiguous phrase
     inside the query (found via word n-gram scanning, longest phrase
     first). This is what makes natural-language questions work, e.g.
     "What drugs interact with Warfarin?" finds "warfarin" without any
     sentence parsing.
  3. PARTIAL   - the reverse of tier 2: a query word/phrase appears as a
     whole-word fragment INSIDE a longer known alias, e.g. the query word
     "diabetes" is found inside the alias "type 2 diabetes mellitus".
     Needed because entity names in the data are often clinical/full forms
     while users type shorter informal terms.
  4. FUZZY     - if nothing matched exactly, difflib.SequenceMatcher finds
     the closest known alias to each query n-gram (handles typos, e.g.
     "worfarin"). Fully explainable: it's a string-similarity ratio, not a
     black box.

Multi-hop expansion follows the graph_pipeline.py convention: INTERACTS_WITH
edges are stored in one direction only per DDInter's source data, so
1-hop retrieval always checks BOTH outgoing and incoming edges of the
matched node - otherwise half of a drug's interactions would be invisible
depending on which side of the pair it happened to be stored on.
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from enum import Enum

import networkx as nx
import pandas as pd
from pydantic import BaseModel

from src import config
from src.data_pipeline import normalize_text
from src.graph_pipeline import load_graph

# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------

class EntityMatch(BaseModel):
    entity_id: str
    canonical_name: str
    entity_type: str
    match_type: str  # EXACT | SUBSTRING | FUZZY
    match_score: float


class QueryIntent(str, Enum):
    SINGLE_ENTITY = "single_entity"
    DIRECT_RELATIONSHIP = "direct_relationship"
    AND = "and"
    OR = "or"
    AMBIGUOUS = "ambiguous"
    NONE = "none"


class EntityQueryAnalysis(BaseModel):
    query: str
    entities: list[EntityMatch]
    intent: QueryIntent


class RetrievedFact(BaseModel):
    source_entity_id: str
    source_name: str
    relationship: str
    target_entity_id: str
    target_name: str
    hop: int
    record_id: str
    evidence: str
    attributes: dict = {}
    rank_score: float


class MultiEntityRetrievalResult(BaseModel):
    query: str
    entities: list[EntityMatch]
    intent: QueryIntent
    relationship_filter: str | None = None
    retrieval_time_ms: float = 0.0
    result_entity_ids: list[str] = []
    direct_facts: list[RetrievedFact] = []
    facts: list[RetrievedFact] = []
    context_facts: list[RetrievedFact] = []


class RetrievalResult(BaseModel):
    query: str
    matched_entity: EntityMatch | None = None
    candidate_matches: list[EntityMatch] = []  # populated when the top match was ambiguous
    relationship_filter: str | None = None
    relationship_filter_relaxed: bool = False  # True if a filter was inferred but found nothing, so it was dropped
    facts: list[RetrievedFact] = []
    subgraph_node_ids: list[str] = []
    retrieval_time_ms: float = 0.0
    hops_used: int = 0  # actual hop depth applied (manual value, or the auto-adaptive decision)
    hops_adaptive: bool = False  # True if hops_used was decided automatically, not passed by the caller
    used_lexical_fallback: bool = False  # True if no entity matched and lexical search answered instead


# ---------------------------------------------------------------------------
# Entity index (built once at startup, reused across queries
# "Performance": retrieval must not rebuild lookup structures per request)
# ---------------------------------------------------------------------------

@dataclass
class EntityIndex:
    alias_lookup: dict[str, list[str]]  # normalized alias -> [entity_id, ...]
    node_info: dict[str, dict]          # entity_id -> {canonical_name, entity_type, mention_count}
    max_alias_words: int = 4


def build_entity_index(entities: pd.DataFrame) -> EntityIndex:
    alias_lookup: dict[str, list[str]] = defaultdict(list)
    node_info: dict[str, dict] = {}
    max_alias_words = 1

    for _, row in entities.iterrows():
        node_info[row["entity_id"]] = {
            "canonical_name": row["canonical_name"],
            "entity_type": row["entity_type"],
            "mention_count": int(row["mention_count"]),
        }
        for alias in row["aliases"].split("|"):
            norm = normalize_text(alias)
            alias_lookup[norm].append(row["entity_id"])
            max_alias_words = max(max_alias_words, len(norm.split()))

    max_alias_words = min(max_alias_words, 8)  # bound n-gram generation cost
    print(f"[RETRIEVAL] Built entity index: {len(alias_lookup):,} aliases -> {len(node_info):,} entities")
    return EntityIndex(alias_lookup=dict(alias_lookup), node_info=node_info, max_alias_words=max_alias_words)


def load_retrieval_engine(
    entities_path=config.PROCESSED_ENTITIES_FILE, graph_path=config.GRAPH_FILE
) -> tuple[nx.DiGraph, EntityIndex]:
    """Convenience loader for app.py / evaluate.py: load the saved graph and
    build the entity index once, ready to answer many queries cheaply.
    """
    G = load_graph(graph_path)
    entities = pd.read_csv(entities_path)
    index = build_entity_index(entities)
    return G, index


# ---------------------------------------------------------------------------
# Query normalization
# ---------------------------------------------------------------------------

def normalize_query(text: str) -> str:
    """Same whitespace/case normalization as entity names, plus stripping
    trailing sentence punctuation ("Warfarin?" -> "warfarin").
    """
    text = re.sub(r"[?!.]+$", "", text.strip()).strip()
    return normalize_text(text)


# Common filler words stripped from the query before generating n-grams for
# tiers 2-4. Without this, a word like "with" can spuriously PARTIAL-match
# inside an unrelated alias that happens to contain it as ordinary English
# (e.g. "Polyethylene glycol (3350 with electrolytes)") and win before the
# real entity word is ever tried - found via evaluation on real queries
# ("What drugs interact with Worfarin?" was matching on "with", not
# "Worfarin"). Deliberately conservative: only interrogative/connector
# words from the query templates this project uses, never touches alias
# text itself.
_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "what", "which", "who",
    "does", "do", "did", "with", "and", "or", "for", "of", "to", "in",
    "on", "at", "by", "about", "show", "tell", "me", "connections",
    "connection", "between", "related", "relate", "interact", "interacts",
    "interaction", "interactions", "treat", "treats", "treating",
    "disease", "diseases", "drug", "drugs", "condition", "conditions",
    "associated", "associate",
}


def _content_words(words: list[str]) -> list[str]:
    filtered = [w for w in words if w not in _QUERY_STOPWORDS]
    return filtered or words  # never discard everything (e.g. a query that's ONLY stopwords)


def _word_ngrams(words: list[str], max_n: int) -> list[str]:
    """All contiguous word spans up to max_n words, longest first, so a
    multi-word entity name (e.g. "type 2 diabetes mellitus") is preferred
    over a shorter partial match.
    """
    ngrams = []
    n_words = len(words)
    for n in range(min(max_n, n_words), 0, -1):
        for i in range(n_words - n + 1):
            ngrams.append(" ".join(words[i : i + n]))
    return ngrams


# ---------------------------------------------------------------------------
# Entity matching (search_entity)
# ---------------------------------------------------------------------------

def _matches_for_alias(alias: str, index: EntityIndex, match_type: str, score: float) -> list[EntityMatch]:
    return [
        EntityMatch(
            entity_id=eid,
            canonical_name=index.node_info[eid]["canonical_name"],
            entity_type=index.node_info[eid]["entity_type"],
            match_type=match_type,
            match_score=score,
        )
        for eid in index.alias_lookup[alias]
    ]


def search_entity(query: str, index: EntityIndex) -> list[EntityMatch]:
    """Find the graph entity/entities a query refers to. Returns matches
    from the first tier that finds anything (exact > substring > fuzzy),
    best score first. Empty list means no entity was recognized.
    """
    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    # Tier 1: exact match on the whole query
    if normalized_query in index.alias_lookup:
        return _matches_for_alias(normalized_query, index, "EXACT", 1.0)

    words = normalized_query.split()
    ngrams = _word_ngrams(words, index.max_alias_words)

    # Tier 2: Find a known alias that exactly matches a phrase in the query.
    # Uses the original query words, including stopwords, because each n-gram
    # must match a complete alias.
    for ngram in ngrams:
        if ngram in index.alias_lookup:
            score = len(ngram.split()) / len(words)
            return _matches_for_alias(ngram, index, "SUBSTRING", round(score, 3))

    # Generate n-grams from content words to reduce false matches caused by
    # common words in the query, such as "with".
    content_ngrams = _word_ngrams(_content_words(words), index.max_alias_words)

    # Tier 3: Find aliases containing a query phrase as a whole-word sequence.
    # Return all matching aliases to preserve ambiguity when a phrase refers
    # to multiple entities (e.g., "diabetes" matching several diabetes types).
    # Sort aliases by length for display, with shorter aliases first.
    for ngram in content_ngrams:
        if not any(re.search(r"[a-z]{3,}", word) for word in ngram.split()):
            continue
        pattern = re.compile(rf"\b{re.escape(ngram)}\b")
        containing = [alias for alias in index.alias_lookup if pattern.search(alias)]
        if containing:
            containing.sort(key=len)  # shortest/most-specific first, for display order
            matches = []
            for alias in containing:
                score = round(len(ngram) / len(alias), 3)
                matches.extend(_matches_for_alias(alias, index, "PARTIAL", score))
            return matches

    # Tier 4: Find the closest alias to each content-word n-gram.
    # Limit n-gram length and apply the configured similarity cutoff.
    # Keep the alias with the highest similarity across all n-grams.
    best_alias, best_score = None, 0.0
    fuzzy_words = _content_words(words)
    for ngram in _word_ngrams(fuzzy_words, min(index.max_alias_words, config.FUZZY_NGRAM_MAX_WORDS)):
        close = get_close_matches(ngram, index.alias_lookup.keys(), n=1, cutoff=config.FUZZY_MATCH_CUTOFF)
        if close:
            score = SequenceMatcher(None, ngram, close[0]).ratio()
            if score > best_score:
                best_alias, best_score = close[0], score

    if best_alias:
        return _matches_for_alias(best_alias, index, "FUZZY", round(best_score, 3))

    return []


_SET_QUERY_TARGETS = (
    "drug", "drugs", "medicine", "medicines", "medication", "medications",
    "disease", "diseases", "condition", "conditions",
)
_AND_QUERY_PATTERN = re.compile(r"\b(?:both|all of|each of|common to|in common)\b")
_OR_QUERY_PATTERN = re.compile(r"\b(?:or|either|any of)\b")
_SET_QUERY_PATTERN = re.compile(
    r"\b(?:what|which|list|find|show|are there|is there|do any|does any|can any)\b"
)
_DIRECT_RELATIONSHIP_PATTERN = re.compile(
    r"\b(?:relationship|relationships|connection|connections|interact|interacts|"
    r"interaction|interactions|treat|treats|treated|indication|indications)\b"
)


def find_query_entities(query: str, index: EntityIndex) -> list[EntityMatch]:
    """Find every distinct, explicitly named entity, preferring longer aliases.

    This intentionally uses known aliases only. It does not fuzzy-match each
    query fragment, which could turn ordinary words into spurious entities.
    """
    normalized_query = normalize_query(query)
    if not normalized_query:
        return []

    candidates = []
    for alias, entity_ids in index.alias_lookup.items():
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(normalized_query):
            for entity_id in entity_ids:
                candidates.append((match.start(), match.end(), alias, entity_id))

    # Resolve overlaps by considering the earliest unmatched text position
    # and choosing the longest alias beginning there.
    candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))
    selected: list[EntityMatch] = []
    seen_entities: set[str] = set()
    cursor = 0
    while candidates:
        eligible = [candidate for candidate in candidates if candidate[0] >= cursor]
        if not eligible:
            break
        start = min(candidate[0] for candidate in eligible)
        at_start = [candidate for candidate in eligible if candidate[0] == start]
        end = max(candidate[1] for candidate in at_start)
        longest = [candidate for candidate in at_start if candidate[1] == end]
        for _, _, alias, entity_id in longest:
            if entity_id in seen_entities:
                continue
            info = index.node_info[entity_id]
            selected.append(
                EntityMatch(
                    entity_id=entity_id,
                    canonical_name=info["canonical_name"],
                    entity_type=info["entity_type"],
                    match_type="EXACT" if normalized_query == alias else "SUBSTRING",
                    match_score=1.0 if normalized_query == alias else round(len(alias) / len(normalized_query), 3),
                )
            )
            seen_entities.add(entity_id)
        cursor = end

    return selected


def detect_query_intent(query: str, entity_count: int) -> QueryIntent:
    """Classify supported multi-entity wording with conservative rules."""
    if entity_count == 0:
        return QueryIntent.NONE
    if entity_count == 1:
        return QueryIntent.SINGLE_ENTITY

    normalized_query = normalize_query(query)
    if _OR_QUERY_PATTERN.search(normalized_query):
        return QueryIntent.OR

    has_and = re.search(r"\band\b", normalized_query) is not None
    asks_for_set = (
        _SET_QUERY_PATTERN.search(normalized_query) is not None
        and any(re.search(rf"\b{target}\b", normalized_query) for target in _SET_QUERY_TARGETS)
    )
    if asks_for_set and (
        _AND_QUERY_PATTERN.search(normalized_query)
        or (has_and and _DIRECT_RELATIONSHIP_PATTERN.search(normalized_query))
    ):
        return QueryIntent.AND

    direct = _DIRECT_RELATIONSHIP_PATTERN.search(normalized_query)
    between = re.search(r"\bbetween\b", normalized_query) is not None
    if direct and (between or not asks_for_set):
        return QueryIntent.DIRECT_RELATIONSHIP

    return QueryIntent.AMBIGUOUS


def analyze_entity_query(query: str, index: EntityIndex) -> EntityQueryAnalysis:
    """Find named entities and classify the supported query intent."""
    entities = find_query_entities(query, index)
    return EntityQueryAnalysis(
        query=query,
        entities=entities,
        intent=detect_query_intent(query, len(entities)),
    )


# ---------------------------------------------------------------------------
# Relationship-type inference (best-effort, keyword-based, never mandatory)
# ---------------------------------------------------------------------------

_RELATIONSHIP_KEYWORDS = {
    config.REL_INTERACTS_WITH: ["interact", "interaction", "combine", "combination", "together with"],
    config.REL_TREATS: ["treat", "disease", "condition", "diagnos", "indicat", "used for", "prescribed for"],
}


def infer_relationship_filter(query: str) -> str | None:
    """Best-effort keyword scan to narrow results to one relationship type,
    e.g. "interact" -> INTERACTS_WITH. Returns None (no filter) if the
    query doesn't clearly signal one - callers must treat that as "show
    everything", not as a failure.
    """
    q = query.lower()
    scores = {rel: sum(1 for kw in kws if kw in q) for rel, kws in _RELATIONSHIP_KEYWORDS.items()}
    scores = {rel: n for rel, n in scores.items() if n > 0}
    if not scores:
        return None
    return max(scores, key=scores.get)


# ---------------------------------------------------------------------------
# Ranking (simple, transparent - no PageRank, no learned model)
# ---------------------------------------------------------------------------

_HOP_WEIGHT = {1: 10, 2: 5}
_SEVERITY_WEIGHT = {"Major": 3, "Moderate": 2, "Minor": 1, "Unknown": 0}


def _rank_score(hop: int, relationship: str, attributes: dict) -> float:
    """Rank purely by: hop distance first (closer = more relevant), then a
    relationship-appropriate support/confidence signal as a tiebreaker -
    interaction severity for INTERACTS_WITH, curator/resource count for
    TREATS. This is the "frequency/support if available" rule from the
    brief, not a graph centrality algorithm.
    """
    score = _HOP_WEIGHT.get(hop, 0)
    if relationship == config.REL_INTERACTS_WITH:
        score += _SEVERITY_WEIGHT.get(attributes.get("severity"), 0)
    elif relationship == config.REL_TREATS:
        support = attributes.get("n_curators", 0) + attributes.get("n_resources", 0)
        score += min(support, 5)
    return score


# ---------------------------------------------------------------------------
# 1-hop / 2-hop retrieval
# ---------------------------------------------------------------------------

def _edge_to_fact(G: nx.DiGraph, source_id: str, target_id: str, data: dict, hop: int) -> RetrievedFact:
    attrs = {k: v for k, v in data.items() if k not in ("relationship", "record_id", "evidence")}
    return RetrievedFact(
        source_entity_id=source_id,
        source_name=G.nodes[source_id]["canonical_name"],
        relationship=data["relationship"],
        target_entity_id=target_id,
        target_name=G.nodes[target_id]["canonical_name"],
        hop=hop,
        record_id=data["record_id"],
        evidence=data["evidence"],
        attributes=attrs,
        rank_score=_rank_score(hop, data["relationship"], attrs),
    )


def retrieve_one_hop(G: nx.DiGraph, entity_id: str, relationship_filter: str | None = None) -> list[RetrievedFact]:
    """All edges directly touching entity_id, in either direction (see
    module docstring on why both directions are always checked).
    """
    facts = []
    for _, target, data in G.out_edges(entity_id, data=True):
        if relationship_filter and data["relationship"] != relationship_filter:
            continue
        facts.append(_edge_to_fact(G, entity_id, target, data, hop=1))
    for source, _, data in G.in_edges(entity_id, data=True):
        if relationship_filter and data["relationship"] != relationship_filter:
            continue
        facts.append(_edge_to_fact(G, source, entity_id, data, hop=1))
    return facts


def retrieve_two_hop(
    G: nx.DiGraph,
    entity_id: str,
    one_hop_facts: list[RetrievedFact],
    relationship_filter: str | None = None,
    max_seed_nodes: int = config.TWO_HOP_MAX_SEED_NODES,
) -> list[RetrievedFact]:
    """Edges one step beyond the 1-hop neighbors. To prevent excessive
    expansion from a high-degree node, only the top `max_seed_nodes`
    1-hop neighbors (by rank_score) are expanded further.
    """
    seen_pairs = {(f.source_entity_id, f.target_entity_id) for f in one_hop_facts}

    seed_ids, seen_seeds = [], set()
    for f in sorted(one_hop_facts, key=lambda f: -f.rank_score):
        neighbor_id = f.target_entity_id if f.source_entity_id == entity_id else f.source_entity_id
        if neighbor_id != entity_id and neighbor_id not in seen_seeds:
            seen_seeds.add(neighbor_id)
            seed_ids.append(neighbor_id)
        if len(seed_ids) >= max_seed_nodes:
            break

    facts = []
    for seed_id in seed_ids:
        for _, target, data in G.out_edges(seed_id, data=True):
            if target == entity_id or (seed_id, target) in seen_pairs:
                continue
            if relationship_filter and data["relationship"] != relationship_filter:
                continue
            facts.append(_edge_to_fact(G, seed_id, target, data, hop=2))
        for source, _, data in G.in_edges(seed_id, data=True):
            if source == entity_id or (source, seed_id) in seen_pairs:
                continue
            if relationship_filter and data["relationship"] != relationship_filter:
                continue
            facts.append(_edge_to_fact(G, source, seed_id, data, hop=2))
    return facts


def _consolidate(facts: list[RetrievedFact]) -> list[RetrievedFact]:
    """Collapse duplicate (source, relationship, target) triples that can
    arise when 2-hop paths rediscover the same fact via a different seed
    node - keep the lowest hop / highest rank_score version.
    """
    best: dict[tuple[str, str, str], RetrievedFact] = {}
    for f in facts:
        key = (f.source_entity_id, f.relationship, f.target_entity_id)
        current = best.get(key)
        if current is None or f.rank_score > current.rank_score:
            best[key] = f
    return list(best.values())


def _fact_neighbor_id(fact: RetrievedFact, entity_id: str) -> str:
    return fact.target_entity_id if fact.source_entity_id == entity_id else fact.source_entity_id


def _requested_result_entity_type(query: str) -> str | None:
    normalized_query = normalize_query(query)
    asks_for_drugs = re.search(
        r"\b(?:drug|drugs|medicine|medicines|medication|medications)\b",
        normalized_query,
    )
    asks_for_diseases = re.search(
        r"\b(?:disease|diseases|condition|conditions)\b",
        normalized_query,
    )
    if bool(asks_for_drugs) == bool(asks_for_diseases):
        return None
    return config.ENTITY_DRUG if asks_for_drugs else config.ENTITY_DISEASE


def _fact_sort_key(fact: RetrievedFact) -> tuple:
    return (
        fact.hop,
        -fact.rank_score,
        fact.source_name.casefold(),
        fact.target_name.casefold(),
        fact.relationship,
        fact.record_id,
    )


def retrieve_multi_entity_relationships(
    query: str,
    G: nx.DiGraph,
    index: EntityIndex,
    top_k: int = config.TOP_K_RESULTS,
    include_context: bool = False,
    context_hops: int = 1,
) -> MultiEntityRetrievalResult:
    """Retrieve direct, shared (AND), or combined (OR) facts for named entities.

    Primary results always follow the classified query intent. Optional
    neighborhood facts are returned separately as context and are never
    presented as direct evidence between the named entities.
    """
    start = time.perf_counter()
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if context_hops not in (1, 2):
        raise ValueError("context_hops must be 1 or 2")

    analysis = analyze_entity_query(query, index)
    entities = analysis.entities
    entity_ids = {entity.entity_id for entity in entities}
    relationship_filter = infer_relationship_filter(query)
    primary_facts: list[RetrievedFact] = []
    result_entity_ids: list[str] = []

    if analysis.intent == QueryIntent.DIRECT_RELATIONSHIP:
        for position, first in enumerate(entities):
            for second in entities[position + 1 :]:
                for source_id, target_id in (
                    (first.entity_id, second.entity_id),
                    (second.entity_id, first.entity_id),
                ):
                    if not G.has_edge(source_id, target_id):
                        continue
                    data = G[source_id][target_id]
                    if relationship_filter and data["relationship"] != relationship_filter:
                        continue
                    primary_facts.append(_edge_to_fact(G, source_id, target_id, data, hop=1))
        primary_facts = _consolidate(primary_facts)
        primary_facts.sort(key=_fact_sort_key)
        primary_facts = primary_facts[:top_k]
        result_entity_ids = sorted(
            {
                endpoint
                for fact in primary_facts
                for endpoint in (fact.source_entity_id, fact.target_entity_id)
                if endpoint in entity_ids
            }
        )

    elif analysis.intent in (QueryIntent.AND, QueryIntent.OR):
        facts_by_entity = {
            entity.entity_id: retrieve_one_hop(G, entity.entity_id, relationship_filter)
            for entity in entities
        }
        requested_type = _requested_result_entity_type(query)
        neighbors_by_entity = {}
        for entity_id, facts in facts_by_entity.items():
            neighbors = set()
            for fact in facts:
                neighbor_id = _fact_neighbor_id(fact, entity_id)
                if neighbor_id in entity_ids:
                    continue
                if requested_type and index.node_info[neighbor_id]["entity_type"] != requested_type:
                    continue
                neighbors.add(neighbor_id)
            neighbors_by_entity[entity_id] = neighbors

        if analysis.intent == QueryIntent.AND:
            result_ids = set.intersection(*neighbors_by_entity.values()) if neighbors_by_entity else set()
        else:
            result_ids = set.union(*neighbors_by_entity.values()) if neighbors_by_entity else set()

        supporting_facts = [
            fact
            for entity_id, facts in facts_by_entity.items()
            for fact in facts
            if _fact_neighbor_id(fact, entity_id) in result_ids
        ]
        supporting_facts = _consolidate(supporting_facts)
        best_score_by_result = {
            result_id: max(
                fact.rank_score
                for fact in supporting_facts
                if result_id in (fact.source_entity_id, fact.target_entity_id)
            )
            for result_id in result_ids
        }
        result_entity_ids = sorted(
            result_ids,
            key=lambda result_id: (
                -best_score_by_result[result_id],
                index.node_info[result_id]["canonical_name"].casefold(),
                result_id,
            ),
        )[:top_k]
        selected_ids = set(result_entity_ids)
        primary_facts = [
            fact
            for fact in supporting_facts
            if any(
                endpoint in selected_ids
                for endpoint in (fact.source_entity_id, fact.target_entity_id)
            )
        ]
        primary_facts.sort(key=_fact_sort_key)

    direct_facts = (
        primary_facts if analysis.intent == QueryIntent.DIRECT_RELATIONSHIP else []
    )
    context_facts: list[RetrievedFact] = []
    if include_context and analysis.intent in (
        QueryIntent.DIRECT_RELATIONSHIP,
        QueryIntent.AND,
        QueryIntent.OR,
    ):
        for entity in entities:
            one_hop = retrieve_one_hop(G, entity.entity_id, relationship_filter)
            context_facts.extend(one_hop)
            if context_hops == 2 and one_hop:
                context_facts.extend(
                    retrieve_two_hop(G, entity.entity_id, one_hop, relationship_filter)
                )

        primary_keys = {
            (fact.source_entity_id, fact.relationship, fact.target_entity_id)
            for fact in primary_facts
        }
        context_facts = [
            fact
            for fact in _consolidate(context_facts)
            if (fact.source_entity_id, fact.relationship, fact.target_entity_id) not in primary_keys
            and not (
                fact.source_entity_id in entity_ids
                and fact.target_entity_id in entity_ids
            )
        ]
        context_facts.sort(key=_fact_sort_key)
        context_facts = context_facts[:top_k]

    return MultiEntityRetrievalResult(
        query=query,
        entities=entities,
        intent=analysis.intent,
        relationship_filter=relationship_filter,
        retrieval_time_ms=(time.perf_counter() - start) * 1000,
        result_entity_ids=result_entity_ids,
        direct_facts=direct_facts,
        facts=primary_facts,
        context_facts=context_facts,
    )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def retrieve_relationships(
    query: str,
    G: nx.DiGraph,
    index: EntityIndex,
    max_hops: int = config.MAX_HOPS,
    top_k: int = config.TOP_K_RESULTS,
    apply_relationship_filter: bool = True,
    adaptive_hops: bool = False,
    adaptive_hop_top_k: int | None = None,
) -> RetrievalResult:
    """The pipeline for one query. 

    `adaptive_hops=True` ignores `max_hops` and instead picks the hop
    depth per query based on the matched entity's own 1-hop connection
    count: if it already has enough direct connections to fill the result
    list, 2-hop expansion is skipped (hop=1 facts always outrank hop=2 in
    the existing ranking - see _rank_score - so with this many direct
    facts, 2-hop results would rarely surface anyway, and skipping them
    saves the traversal cost). Below that, expansion to 2 hops runs as
    usual to add context for an otherwise sparse entity.

    The "enough" threshold is `top_k` by default, but a caller that requests
    a wider candidate pool than it will actually display should pass its real
    display size as `adaptive_hop_top_k` - otherwise the decision would be
    judged against the oversized pool and wrongly call entities "low degree"
    that aren't from the caller's actual point of view.

    Off by default - existing callers (evaluate.py) are unaffected unless
    they opt in.
    """
    start = time.perf_counter()
    print(f'[RETRIEVAL] Query: "{query}"')

    matches = search_entity(query, index)
    if not matches:
        print("[RETRIEVAL] No matching entity found")
        return RetrievalResult(query=query, retrieval_time_ms=(time.perf_counter() - start) * 1000)

    # Rank matches by match score, then by entity mention count.
    # PARTIAL and FUZZY matches retain all alternative entities as candidates.
    # EXACT and SUBSTRING matches retain only alternatives tied with the
    # best match score, avoiding unnecessary ambiguity from weaker matches.
    matches.sort(key=lambda m: (-m.match_score, -index.node_info[m.entity_id]["mention_count"]))
    best = matches[0]
    if best.match_type in ("PARTIAL", "FUZZY"):
        candidates = [m for m in matches if m.entity_id != best.entity_id]
    else:
        candidates = [m for m in matches if m.match_score == best.match_score and m.entity_id != best.entity_id]
    print(f"[RETRIEVAL] Matched entity: {best.canonical_name} ({best.match_type}, score={best.match_score})")

    relationship_filter = infer_relationship_filter(query) if apply_relationship_filter else None
    if relationship_filter:
        print(f"[RETRIEVAL] Inferred relationship filter: {relationship_filter}")

    one_hop = retrieve_one_hop(G, best.entity_id, relationship_filter)
    relationship_filter_relaxed = False

    # If the filter produces no results, retry without it so that existing
    # relationships of other types are not omitted. Record the relaxation
    # so the caller can explain why the results may not match the query's
    # requested relationship type.
    if not one_hop and relationship_filter:
        print(f"[RETRIEVAL] No {relationship_filter} relationships found for this entity - "
              f"relaxing filter and showing all relationship types")
        relationship_filter_relaxed = True
        relationship_filter = None
        one_hop = retrieve_one_hop(G, best.entity_id, relationship_filter)

    # In adaptive mode, compare the full 1-hop result count with the threshold.
    # Use adaptive_hop_top_k when supplied; otherwise, use top_k.
    # Retrieve 2-hop relationships only when direct connections fall below
    # the threshold. Otherwise, respect the configured max_hops.
    if adaptive_hops:
        threshold = adaptive_hop_top_k if adaptive_hop_top_k is not None else top_k
        effective_max_hops = 1 if len(one_hop) >= threshold else 2
        print(f"[RETRIEVAL] Adaptive hops: {best.canonical_name} has {len(one_hop)} direct connections "
              f"(threshold={threshold}) -> using {effective_max_hops} hop(s)")
    else:
        effective_max_hops = max_hops

    all_facts = list(one_hop)
    if effective_max_hops >= 2:
        all_facts += retrieve_two_hop(G, best.entity_id, one_hop, relationship_filter)

    all_facts = _consolidate(all_facts)
    all_facts.sort(key=lambda f: (f.hop, -f.rank_score))
    all_facts = all_facts[:top_k]

    node_ids = {best.entity_id}
    for f in all_facts:
        node_ids.add(f.source_entity_id)
        node_ids.add(f.target_entity_id)

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[RETRIEVAL] Retrieved {len(all_facts)} relationships in {elapsed_ms:.1f}ms")

    return RetrievalResult(
        query=query,
        matched_entity=best,
        candidate_matches=candidates,
        relationship_filter=relationship_filter,
        relationship_filter_relaxed=relationship_filter_relaxed,
        facts=all_facts,
        subgraph_node_ids=sorted(node_ids),
        retrieval_time_ms=elapsed_ms,
        hops_used=effective_max_hops,
        hops_adaptive=adaptive_hops,
    )


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.retrieval`
    G, index = load_retrieval_engine()

    demo_queries = [
        "What drugs interact with Warfarin?",
        "What diseases are associated with Metformin?",
        "What drugs are related to Diabetes?",
        "xyzabc123",  # should gracefully find nothing
    ]
    for q in demo_queries:
        result = retrieve_relationships(q, G, index)
        print(f"\n=== \"{q}\" ===")
        if not result.matched_entity:
            print("No matching medical entity was found in the current knowledge graph.")
            continue
        print(f"Matched: {result.matched_entity.canonical_name} [{result.matched_entity.entity_type}]")
        if result.candidate_matches:
            alt = ", ".join(f"{m.canonical_name} ({m.match_score})" for m in result.candidate_matches)
            print(f"Other possible matches: {alt}")
        for f in result.facts[:8]:
            print(f"  (hop {f.hop}) {f.source_name} -[{f.relationship}]-> {f.target_name}  (score={f.rank_score})")
