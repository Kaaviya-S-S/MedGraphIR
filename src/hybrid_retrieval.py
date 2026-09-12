"""
Hybrid retrieval: combines TF-IDF lexical relevance with graph structural
relevance into a single ranked result.

This module does NOT reimplement either retrieval method - it calls
src.retrieval.retrieve_relationships() and src.keyword_baseline.
KeywordRetriever.search() exactly as they already exist, then fuses their
outputs. Neither existing module is modified in a way that changes its own
behavior (keyword_baseline.KeywordResult only gained extra fields, all
additive - see its docstring).

Pipeline:
    query -> TF-IDF retrieval (top HYBRID_CANDIDATE_POOL_K)
    query -> graph retrieval  (top HYBRID_CANDIDATE_POOL_K, existing ranking)
    -> merge candidates by record_id (the one identifier both methods share)
    -> min-max normalize each method's own scores to [0,1] independently
    -> HybridScore = alpha * tfidf_norm + beta * graph_norm
    -> sort, drop zero-score candidates, take top_k

Why record_id: both methods ultimately point at rows of the same
relations table (src/relation_extraction.py's output) - graph edges are
built 1:1 from those rows (graph_builder.py) and the keyword baseline
searches those rows directly. record_id is the one identifier guaranteed
unique across that whole table (verified in Milestone 1/3), so it is the
correct merge/dedup key - not entity names, which can collide or vary in
casing.

Why min-max normalization: TF-IDF cosine similarity is already in [0,1],
but the graph's rank_score (src.retrieval._rank_score) is an unbounded
additive score (hop weight + severity/support bonus, roughly 0-15) and
would otherwise dominate or be dominated arbitrarily depending on alpha/
beta. Rescaling each method's scores to [0,1] *within the current
candidate pool* keeps alpha/beta weights meaningful and comparable across
different queries, and is simple enough to explain in one sentence - no
learned fusion model, no reciprocal-rank fusion, just linear combination
of two rescaled relevance signals.
"""

import time

import networkx as nx

from src import config
from src.keyword_baseline import KeywordRetriever, KeywordResult
from src.retrieval import EntityIndex, RetrievedFact, RetrievalResult, retrieve_relationships


def _min_max_normalize(scores: dict[str, float]) -> dict[str, float]:
    """Rescale a {id: raw_score} dict to [0,1]. All-equal inputs collapse
    to 1.0 if the (equal) value is positive, else 0.0 - both methods here
    only ever produce positive raw scores for candidates they actually
    returned, so this only matters for a single-candidate pool.
    """
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {k: (1.0 if hi > 0 else 0.0) for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def keyword_result_to_fact(kr: KeywordResult, index: EntityIndex, hybrid_score: float) -> RetrievedFact:
    """Build a RetrievedFact for a candidate found ONLY via TF-IDF (never
    reached by graph traversal from the matched entity). hop=0 is a
    sentinel meaning "not from graph traversal" - distinct from the real
    hop=1/hop=2 values graph facts carry. Canonical display names are
    looked up from the entity index (not taken from the raw relations.csv
    text) so hybrid results are visually consistent with pure-graph facts
    even when the underlying row used a different casing variant.
    """
    source_name = index.node_info.get(kr.source_entity_id, {}).get("canonical_name", kr.source_entity)
    target_name = index.node_info.get(kr.target_entity_id, {}).get("canonical_name", kr.target_entity)
    return RetrievedFact(
        source_entity_id=kr.source_entity_id,
        source_name=source_name,
        relationship=kr.relationship,
        target_entity_id=kr.target_entity_id,
        target_name=target_name,
        hop=0,
        record_id=kr.record_id,
        evidence=kr.evidence,
        attributes=kr.attributes,
        rank_score=hybrid_score,
    )


def retrieve_hybrid(
    query: str,
    G: nx.DiGraph,
    index: EntityIndex,
    keyword_retriever: KeywordRetriever,
    alpha: float = config.HYBRID_ALPHA_DEFAULT,
    beta: float = config.HYBRID_BETA_DEFAULT,
    max_hops: int = config.MAX_HOPS,
    top_k: int = config.TOP_K_RESULTS,
    candidate_pool_k: int = config.HYBRID_CANDIDATE_POOL_K,
    adaptive_hops: bool = False,
) -> RetrievalResult:
    """Fuse graph retrieval and TF-IDF keyword retrieval into one ranked
    result. Reuses both methods' existing implementations unchanged -
    entity matching, relationship-filter inference, 1-2 hop expansion,
    and graph rank_score all come from retrieve_relationships() as-is.

    `alpha=1, beta=0` exactly reproduces pure TF-IDF's result set and
    order; `alpha=0, beta=1` exactly reproduces pure graph retrieval's -
    verified directly against retrieve_relationships()/KeywordRetriever.
    search() output for multiple real queries (see README for the
    specific test methodology and results).
    """
    start = time.perf_counter()

    # adaptive_hop_top_k=top_k: the adaptive decision must be judged
    # against the HYBRID's real, user-facing result size, not the wider
    # candidate_pool_k this internal call requests for merging purposes -
    # see retrieve_relationships' docstring.
    graph_result = retrieve_relationships(
        query, G, index, max_hops=max_hops, top_k=candidate_pool_k,
        adaptive_hops=adaptive_hops, adaptive_hop_top_k=top_k,
    )
    keyword_results = keyword_retriever.search(query, top_k=candidate_pool_k)

    graph_by_record = {f.record_id: f for f in graph_result.facts}
    keyword_by_record = {r.record_id: r for r in keyword_results}

    graph_norm = _min_max_normalize({rid: f.rank_score for rid, f in graph_by_record.items()})
    keyword_norm = _min_max_normalize({rid: r.score for rid, r in keyword_by_record.items()})

    # Ties in hybrid_score are common (e.g. many TREATS facts legitimately
    # share the same capped support bonus in _rank_score) and need a
    # deterministic tiebreaker. Using each candidate's own RANK POSITION
    # within its source method's already-ordered result list - blended by
    # the same alpha/beta weights - means the tiebreak collapses to pure
    # graph's own tie order when alpha=0, and pure keyword's own tie order
    # when beta=0, so both degenerate cases match exactly instead of only
    # approximately.
    keyword_rank_pos = {r.record_id: i for i, r in enumerate(keyword_results)}
    graph_rank_pos = {f.record_id: i for i, f in enumerate(graph_result.facts)}
    pool_size = max(len(keyword_results), len(graph_result.facts), 1)

    def _tiebreak(rid: str) -> float:
        kw_pos = keyword_rank_pos.get(rid, pool_size)  # not found -> worst possible position
        g_pos = graph_rank_pos.get(rid, pool_size)
        return alpha * kw_pos + beta * g_pos

    all_record_ids = set(graph_by_record) | set(keyword_by_record)
    merged_facts = []
    for rid in all_record_ids:
        found_by_keyword = rid in keyword_by_record
        found_by_graph = rid in graph_by_record

        # A candidate belongs in the output if a POSITIVELY-weighted method
        # actually retrieved it - regardless of what its normalized score
        # happens to be. Min-max normalization maps the worst-ranked
        # candidate in a pool to exactly 0.0, which is a legitimate low
        # score, not the same thing as "this method never found it"; a
        # naive `hybrid_score <= 0` check would wrongly drop that
        # candidate. This is what makes alpha=1/beta=0 exactly reproduce
        # pure TF-IDF's candidate set, and alpha=0/beta=1 exactly
        # reproduce pure graph's - not just approximately.
        contributes = (alpha > 0 and found_by_keyword) or (beta > 0 and found_by_graph)
        if not contributes:
            continue

        hybrid_score = alpha * keyword_norm.get(rid, 0.0) + beta * graph_norm.get(rid, 0.0)
        if found_by_graph:
            fact = graph_by_record[rid].model_copy(update={"rank_score": hybrid_score})
        else:
            fact = keyword_result_to_fact(keyword_by_record[rid], index, hybrid_score)
        merged_facts.append((fact, _tiebreak(rid)))

    merged_facts.sort(key=lambda pair: (-pair[0].rank_score, pair[1]))
    merged_facts = [fact for fact, _ in merged_facts[:top_k]]

    node_ids = {f.source_entity_id for f in merged_facts} | {f.target_entity_id for f in merged_facts}
    if graph_result.matched_entity:
        node_ids.add(graph_result.matched_entity.entity_id)

    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[HYBRID] alpha={alpha} beta={beta}: merged {len(graph_by_record)} graph + "
          f"{len(keyword_by_record)} keyword candidates -> {len(merged_facts)} results in {elapsed_ms:.1f}ms")

    return RetrievalResult(
        query=query,
        matched_entity=graph_result.matched_entity,
        candidate_matches=graph_result.candidate_matches,
        relationship_filter=graph_result.relationship_filter,
        relationship_filter_relaxed=graph_result.relationship_filter_relaxed,
        facts=merged_facts,
        subgraph_node_ids=sorted(node_ids),
        retrieval_time_ms=elapsed_ms,
        hops_used=graph_result.hops_used,
        hops_adaptive=graph_result.hops_adaptive,
    )


if __name__ == "__main__":
    # Quick manual smoke test: `python -m src.hybrid_retrieval`
    from src.keyword_baseline import load_keyword_baseline
    from src.retrieval import load_retrieval_engine

    G, index = load_retrieval_engine()
    keyword_retriever = load_keyword_baseline()

    demo_queries = [
        "What drugs interact with Warfarin?",
        "What diseases are associated with Metformin?",
        "xyzabc123",
    ]
    for q in demo_queries:
        result = retrieve_hybrid(q, G, index, keyword_retriever)
        print(f'\n=== "{q}" ===')
        matched = result.matched_entity.canonical_name if result.matched_entity else None
        print(f"Matched entity (graph-side): {matched}")
        for f in result.facts[:8]:
            hop_label = "graph" if f.hop > 0 else "keyword-only"
            print(f"  [{hop_label}] {f.source_name} -[{f.relationship}]-> {f.target_name} (hybrid_score={f.rank_score:.3f})")
