"""Graph-first retrieval with a lexical fallback.

Graph retrieval is the primary method: it matches a query to a graph entity
and returns that entity's ranked neighborhood. When a query matches no entity
at all - a random string, or a drug/disease the graph doesn't contain -
there is nothing to traverse, so rather than dead-end with an empty result
this policy falls back to BM25 over the raw query text.
"""

import time
from collections.abc import Callable

import networkx as nx

from src import config
from src.bm25_baseline import BM25Retriever, lexical_result_to_fact, load_bm25_baseline
from src.retrieval import EntityIndex, RetrievalResult, retrieve_relationships, load_retrieval_engine

def retrieve_with_fallback(
    query: str,
    G: nx.DiGraph,
    index: EntityIndex,
    bm25_retriever: BM25Retriever | None,
    max_hops: int = config.MAX_HOPS,
    top_k: int = config.TOP_K_RESULTS,
    adaptive_hops: bool = False,
    bm25_loader: Callable[[], BM25Retriever] | None = None,
) -> RetrievalResult:
    """Graph retrieval first; lexical BM25 only when no entity matched.

    A matched entity always wins, even if its neighborhood is small - the
    graph result is only discarded when entity matching found nothing, which
    is the one case where graph retrieval has no answer to give.
    """
    start = time.perf_counter()

    result = retrieve_relationships(
        query, G, index, max_hops=max_hops, adaptive_hops=adaptive_hops, top_k=top_k,
    )
    if result.matched_entity is not None:
        return result

    if bm25_retriever is None:
        if bm25_loader is None:
            raise ValueError("A BM25 retriever or lazy BM25 loader is required for fallback retrieval.")
        bm25_retriever = bm25_loader()

    lexical_results = bm25_retriever.search(query, top_k=top_k)
    if not lexical_results:
        # Neither method found anything - keep the graph result (empty, no
        # matched entity) so the caller's "no match" handling still applies.
        return result

    facts = [lexical_result_to_fact(lexical, index, lexical.score) for lexical in lexical_results]
    elapsed_ms = (time.perf_counter() - start) * 1000
    print(f"[FALLBACK] No graph entity matched - returned {len(facts)} BM25 results")

    return RetrievalResult(
        query=query,
        facts=facts,
        retrieval_time_ms=elapsed_ms,
        used_lexical_fallback=True,
    )


if __name__ == "__main__":
    G, index = load_retrieval_engine()
    bm25_retriever = load_bm25_baseline()

    for query in (
        "What drugs interact with Warfarin?",   # entity matches -> graph result
        "xyzabc123",                            # no entity -> lexical fallback (empty here)
        "metformin",                            # entity matches -> graph result
    ):
        result = retrieve_with_fallback(query, G, index, bm25_retriever, top_k=8)
        matched = result.matched_entity.canonical_name if result.matched_entity else None
        print(f'\n=== "{query}" ===')
        print(f"matched={matched} fallback={result.used_lexical_fallback} facts={len(result.facts)}")
        for fact in result.facts[:5]:
            print(f"  {fact.source_name} -[{fact.relationship}]-> {fact.target_name}")
