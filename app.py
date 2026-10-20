"""
Streamlit UI for the Medical Knowledge Graph IR System.

    USER QUERY -> GRAPH RETRIEVAL -> RELEVANT SUBGRAPH -> OPTIONAL OLLAMA SUMMARY

The graph is built once by build_graph.py and only ever LOADED here, via
st.cache_resource so it loads once per server process rather than once per
query - retrieval itself stays fast even though loading a 161k-edge graph from disk is not free.
"""

import base64

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

from src import config
from src.answer_generator import generate_answer
from src.bm25_baseline import lexical_result_to_fact, load_bm25_baseline
from src.fallback_retrieval import retrieve_with_fallback
from src.retrieval import RetrievalResult, RetrievedFact, load_retrieval_engine, retrieve_relationships

st.set_page_config(page_title="MedGraphIR", page_icon="\U0001F9EC", layout="wide")

_BACKGROUND_IMAGE_PATH = config.PROJECT_ROOT / "assets" / "bg1.jpg"

@st.cache_resource(show_spinner=False)
def _load_background_css(image_path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    return f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{encoded}");
        background-size: cover;
        background-position: center top;
        background-attachment: fixed;
    }}
    </style>
    """


_NODE_COLORS = {"DRUG": "#4C9AFF", "DISEASE": "#FF7452"}
_METRIC_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
_EVAL_DIR = config.PROJECT_ROOT / "evaluation"

@st.cache_resource(show_spinner="Loading knowledge graph...")
def get_engine():
    """Load the saved graph + entity index once per server process."""
    return load_retrieval_engine()


@st.cache_resource(show_spinner="Building BM25 index...")
def get_bm25_retriever():
    """Loaded lazily (only when BM25 is actually selected) since building the
    index takes several seconds - Graph stays the default and should not pay
    that cost on ordinary startup.
    """
    return load_bm25_baseline()


def render_subgraph_html(G, facts: list[RetrievedFact], matched_entity_id: str | None) -> str:
    """Interactive pyvis visualization built strictly from the retrieved
    facts (not a full induced subgraph of every node pair) - this keeps
    the picture consistent with the evidence table shown next to it: every
    edge drawn here has a matching row below, nothing extra sneaks in.

    `matched_entity_id` is None for pure lexical (BM25) results, which never
    resolve a specific entity - the graph then just shows all retrieved
    records with no node emphasized.
    """
    net = Network(height="520px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.barnes_hut(spring_length=160, spring_strength=0.02)

    node_ids = {matched_entity_id} if matched_entity_id else set()
    for f in facts:
        node_ids.add(f.source_entity_id)
        node_ids.add(f.target_entity_id)

    for node_id in node_ids:
        data = G.nodes[node_id]
        net.add_node(
            node_id,
            label=data["canonical_name"],
            title=f"{data['canonical_name']} ({data['entity_type']})",
            color=_NODE_COLORS.get(data["entity_type"], "#999999"),
            size=34 if node_id == matched_entity_id else 18,
        )

    seen_edges = set()
    for f in facts:
        key = (f.source_entity_id, f.target_entity_id, f.relationship)
        if key in seen_edges:
            continue
        seen_edges.add(key)
        net.add_edge(f.source_entity_id, f.target_entity_id, label=f.relationship, title=f.evidence)

    return net.generate_html()


def facts_to_dataframe(facts: list[RetrievedFact]) -> pd.DataFrame:
    rows = []
    for f in facts:
        row = {
            "Hop": f.hop,
            "Source": f.source_name,
            "Relationship": f.relationship,
            "Target": f.target_name,
            "Severity": f.attributes.get("severity", ""),
            "Indication category": f.attributes.get("indication_category", ""),
            "Record ID": f.record_id,
            "Score": f.rank_score,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.loc[:, (df != "").any(axis=0)]  # drop columns that are empty for every row


_METHOD_LABELS = {"graph": "Graph", "bm25": "BM25"}
_METHOD_ORDER = ["graph", "bm25"]


def _grouped_bar_chart(wide_df: pd.DataFrame, metrics: list[str], y_title: str) -> alt.Chart:
    """Grouped (side-by-side) bar chart: one x-axis category per row of
    wide_df (e.g. method), with one bar per metric column within each
    group. Built with Altair (bundled with Streamlit, no new dependency)
    because st.bar_chart's default is STACKED bars for multiple columns,
    which misrepresents independent [0,1] scores as if they summed to
    something meaningful.
    """
    category_col = wide_df.index.name or "category"
    long_df = wide_df[metrics].reset_index().melt(id_vars=category_col, var_name="Metric", value_name="Value")
    return (
        alt.Chart(long_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(f"{category_col}:N", title=None, sort=list(wide_df.index), axis=alt.Axis(labelAngle=0)),
            xOffset=alt.XOffset("Metric:N", sort=metrics),
            y=alt.Y("Value:Q", title=y_title),
            color=alt.Color(
                "Metric:N", sort=metrics,
                scale=alt.Scale(domain=metrics, range=_METRIC_COLORS[: len(metrics)]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            tooltip=[
                alt.Tooltip(f"{category_col}:N", title="Method"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        )
        .properties(height=320)
    )


def _single_bar_chart(wide_df: pd.DataFrame, column: str, y_title: str, color: str = _METRIC_COLORS[0]) -> alt.Chart:
    """Single-series bar chart (no legend needed - the axis/title already
    names the one series, per the accessibility guidance of not adding a
    legend box for a single series).
    """
    category_col = wide_df.index.name or "category"
    df = wide_df[[column]].reset_index()
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color=color)
        .encode(
            x=alt.X(f"{category_col}:N", title=None, sort=list(wide_df.index), axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{column}:Q", title=y_title),
            tooltip=[alt.Tooltip(f"{category_col}:N", title="Method"), alt.Tooltip(f"{column}:Q", title=y_title, format=".1f")],
        )
        .properties(height=280)
    )


def render_evaluation_page():
    """Displays the results of evaluation/evaluate.py - it reads the CSVs
    that script already saved, rather than re-running the ~30s evaluation
    (graph + BM25, over 30 queries) on every page view. Same "build once,
    load fast" pattern as the knowledge graph itself.
    """
    st.title("Evaluation")
    st.markdown(
        "This page separates the original graph-derived neighbor benchmark from the newer "
        "independent human-judgment evaluation. The graph-derived benchmark checks retrieval "
        "against graph neighbors; it is not an independent measure of user relevance."
    )

    summary_path = _EVAL_DIR / "results_summary.csv"
    detailed_path = _EVAL_DIR / "results_detailed.csv"
    test_queries_path = _EVAL_DIR / "test_queries.csv"

    if not summary_path.exists():
        st.warning(
            "No evaluation results found yet. Run `python evaluation/evaluate.py` "
            "from the project root, then reload this page."
        )
        return

    saved_summary = pd.read_csv(summary_path, index_col=0)
    available_methods = [method for method in _METHOD_ORDER if method in saved_summary.index]
    summary_df = saved_summary.reindex(available_methods)
    if "bm25" not in saved_summary.index:
        st.info("The saved graph-derived results predate BM25. Re-run `python evaluation/evaluate.py` to refresh them.")
    if test_queries_path.exists():
        st.caption(
            f"Graph-derived benchmark: {len(pd.read_csv(test_queries_path))} queries. "
            "Its expected neighbors are generated from the same graph being searched."
        )

    st.subheader("1. Overall metrics")
    table_df = summary_df.rename(index=_METHOD_LABELS)[
        ["precision", "recall", "f1", "precision_at_5", "precision_at_10", "retrieval_time_ms"]
    ]
    table_df.columns = ["Precision", "Recall", "F1", "P@5", "P@10", "Avg. time (ms)"]
    table_df.index.name = "Method"
    st.dataframe(
        table_df.style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}",
            "P@5": "{:.3f}", "P@10": "{:.3f}", "Avg. time (ms)": "{:.1f}",
        }),
        width="stretch",
    )

    st.subheader("2. Metric comparison chart")
    st.caption("Macro-averaged Precision, Recall, and F1 across the graph-derived evaluation set.")
    chart_df = summary_df[["precision", "recall", "f1"]].rename(index=_METHOD_LABELS)
    chart_df.columns = ["Precision", "Recall", "F1"]
    st.altair_chart(_grouped_bar_chart(chart_df, ["Precision", "Recall", "F1"], "Score"), width="stretch")

    st.subheader("3. Retrieval speed")
    st.caption("Average time per query, milliseconds.")
    time_df = summary_df[["retrieval_time_ms"]].rename(index=_METHOD_LABELS)
    st.altair_chart(_single_bar_chart(time_df, "retrieval_time_ms", "Time (ms)"), width="stretch")
    st.caption(
        "Per-query latency excludes one-time graph/index loading. Compare methods "
        "within the same run and hardware; the graph-derived relevance metrics below "
        "are not independent user-relevance judgments."
    )

    judged_summary_path = _EVAL_DIR / "results_judged_summary.csv"
    judged_detailed_path = _EVAL_DIR / "results_judged_detailed.csv"
    st.subheader("4. Independent human-judgment evaluation")
    if judged_summary_path.exists():
        judged_df = pd.read_csv(judged_summary_path, index_col=0).rename(index=_METHOD_LABELS)
        judged_df.index.name = "Method"
        st.dataframe(judged_df, width="stretch")
        st.caption(
            "These scores use human-assigned relevance grades for pooled records "
            "(0 = irrelevant, 1 = related, 2 = directly answers)."
        )
        if judged_detailed_path.exists():
            with st.expander("Per-query human-judgment results"):
                st.dataframe(pd.read_csv(judged_detailed_path), width="stretch", hide_index=True)
    else:
        st.info(
            "To produce independent relevance results, run "
            "`python evaluation/build_judgment_pool.py`, assign a 0/1/2 relevance "
            "grade to every row in `evaluation/judgment_candidates.csv`, then run "
            "`python evaluation/evaluate_judgments.py`."
        )

    st.divider()
    st.markdown(
        "**Interpretation:** the graph-derived benchmark is structurally favorable to "
        "graph retrieval because its expected entities are taken from that same graph. "
        "It measures neighbor reproduction, not independent user relevance. Consult the "
        "human-judgment section for relevance comparisons; lexical methods also do not "
        "perform the graph method's entity resolution or relationship filtering."
    )

    if detailed_path.exists():
        with st.expander("Per-query results"):
            detailed_df = pd.read_csv(detailed_path)
            method_filter = st.multiselect(
                "Filter by method", options=sorted(detailed_df["method"].unique()),
                default=sorted(detailed_df["method"].unique()),
            )
            st.dataframe(
                detailed_df[detailed_df["method"].isin(method_filter)],
                width="stretch", hide_index=True,
            )


def render_about_page():
    st.title("About")
    st.markdown(
        "**Medical Knowledge Graph IR** is an academic Information Retrieval prototype "
        "that models drug-drug interactions and drug-disease relationships as a graph, "
        "then answers natural-language queries by matching an entity and retrieving its "
        "1-2 hop neighborhood - not by asking a language model to recall facts from memory."
    )

    st.subheader("Data sources")
    st.markdown(
        "- **[DDInter 2.0](https://ddinter2.scbdd.com/)** - drug-drug interactions "
        "(~222k rows, severity-labeled)\n"
        "- **[PharmacotherapyDB 1.0](https://github.com/dhimmel/indications)** - "
        "physician-curated drug-disease indications (601 drugs, 97 diseases)"
    )

    st.subheader("How it works")
    st.markdown(
        "Query -> entity matching (exact / substring / partial / fuzzy) -> adaptive 1-2 hop "
        "graph traversal (the depth is auto-picked per query from the matched entity's "
        "connection count, with a manual override available) -> ranked facts -> optional "
        "local-LLM (Ollama) summary grounded only in those retrieved facts. Retrieval "
        "defaults to Auto: graph-first, falling back to a BM25 lexical search only when no "
        "entity matches. Graph and BM25 can also be selected directly for comparison."
    )

    st.subheader("Tech stack")
    st.markdown("Python, NetworkX, scikit-learn, Streamlit, Pydantic, Ollama (optional).")

    st.divider()
    st.info(
        "**Disclaimer:** This system is an academic information retrieval prototype. "
        "Retrieved information should not be used as a substitute for professional medical advice.",
        icon="⚠️",
    )

def main():
    # st.markdown(_load_background_css(_BACKGROUND_IMAGE_PATH), unsafe_allow_html=True)
    with st.sidebar:
        st.markdown("### Medical KG IR")
        page = st.radio(
            "Navigate", ["\U0001F50E Search", "\U0001F4CA Evaluation", "ℹ️ About"],
            label_visibility="collapsed",
        )
        st.divider()

    if page.endswith("Evaluation"):
        render_evaluation_page()
        return
    if page.endswith("About"):
        render_about_page()
        return

    st.title("\U0001F9EC Medical Knowledge Graph Retrieval System")
    st.markdown(
        "An academic Information Retrieval prototype for exploring drug-drug "
        "interactions and drug-disease relationships."
    )
    st.info(
        "**Disclaimer:** This system is an academic information retrieval prototype. "
        "Retrieved information should not be used as a substitute for professional medical advice.",
        icon="⚠️",
    )

    try:
        G, index = get_engine()
    except FileNotFoundError:
        st.error(
            "No knowledge graph found. Run `python build_graph.py` first to build "
            f"and save the graph to `{config.GRAPH_FILE}`."
        )
        return
    except Exception as exc:
        st.error(
            f"The saved knowledge graph at `{config.GRAPH_FILE}` could not be loaded "
            f"({type(exc).__name__}: {exc}). Try rebuilding it with `python build_graph.py`."
        )
        return

    with st.sidebar:
        st.markdown(f"\U0001F4C8 {G.number_of_nodes():,} nodes · {G.number_of_edges():,} edges")

        adaptive_hops = st.checkbox(
            "Auto-adaptive hops", value=True,
            help="Picks 1 hop for high-degree entities (e.g. Warfarin, "
                 "864 interactions - 2-hop results would rarely surface "
                 "anyway) and 2 hops for low-degree ones (more context "
                 "when there's little direct data).",
        )
        if adaptive_hops:
            max_hops = config.MAX_HOPS  # placeholder; actual value is decided per query
        else:
            max_hops = st.radio("Max hops", options=[1, 2], index=config.MAX_HOPS - 1, horizontal=True)

        use_ollama = st.checkbox("Generate AI summary (requires Ollama)", value=True)
        method = st.radio(
            "Retrieval Method", ["Auto", "Graph", "BM25"], index=0, horizontal=True,
            help="Auto: graph retrieval when the query matches an entity, otherwise "
                 "fall back to BM25 lexical search.",
        )

    with st.form("query_form"):
        query = st.text_input("Enter a medical query", placeholder="What drugs interact with Warfarin?")
        submitted = st.form_submit_button("Search")

    if not submitted:
        return
    if not query or not query.strip():
        st.warning("Please enter a query.")
        return

    if method == "Graph":
        result = retrieve_relationships(query, G, index, max_hops=max_hops, adaptive_hops=adaptive_hops)
    elif method == "Auto":
        # Graph-first policy: graph retrieval when the query matches an entity,
        # lexical BM25 when no entity matched
        result = retrieve_with_fallback(
            query, G, index, get_bm25_retriever(), max_hops=max_hops, adaptive_hops=adaptive_hops,
        )
    else:  # BM25
        # BM25 does no entity resolution at all - it only ranks records by
        # lexical similarity to the raw query text.
        bm25_retriever = get_bm25_retriever()
        bm25_results = bm25_retriever.search(query, top_k=config.TOP_K_RESULTS)
        facts = [lexical_result_to_fact(r, index, r.score) for r in bm25_results]
        result = RetrievalResult(query=query, facts=facts)

    # Lexical retrieval does not resolve an entity by design - only Graph can
    # report "no matching entity found" (Auto reports it by falling back).
    if method == "Graph" and result.matched_entity is None:
        st.warning("No matching medical entity was found in the current knowledge graph.")
        return

    if result.used_lexical_fallback:
        st.warning("No exact entity found, showing keyword matches for the raw query text.")

    if result.candidate_matches:
        alt_names = ", ".join(m.canonical_name for m in result.candidate_matches)
        st.info(
            f"This query is ambiguous - also matched: {alt_names}. "
            f"Showing results for **{result.matched_entity.canonical_name}**."
        )

    match = result.matched_entity
    st.subheader("Matched Entity")
    if match:
        st.markdown(f"**{match.canonical_name}**  •  {match.entity_type}  •  "
                    f"matched via {match.match_type} (score {match.match_score})")
    else:
        st.caption(f"{method} does not resolve a specific entity - showing the top "
                    "lexically similar records for your query text.")

    if not result.facts:
        filter_desc = f" {result.relationship_filter}" if result.relationship_filter else ""
        st.warning(f"No{filter_desc} relationships were found for this query in the knowledge graph.")
        return

    if result.relationship_filter_relaxed:
        st.info(
            f"No relationships of the expected type were found for **{match.canonical_name}**, "
            "so showing all relationship types found for it instead."
        )

    if result.hops_adaptive:
        reason = "already has enough direct connections to fill the results" if result.hops_used == 1 \
            else "has few direct connections, so 2-hop context was added"
        st.caption(f"Auto-adaptive: used {result.hops_used} hop(s) - {match.canonical_name if match else 'this entity'} {reason}.")

    st.subheader("Retrieved Relationships")
    for f in result.facts[:8]:
        detail = f"  _(severity: {f.attributes['severity']})_" if "severity" in f.attributes else ""
        st.markdown(f"- **{f.source_name}** → [{f.relationship}] → **{f.target_name}**{detail}")
    if len(result.facts) > 8:
        st.caption(f"...and {len(result.facts) - 8} more (see full table below)")

    st.subheader("Knowledge Graph")
    html = render_subgraph_html(G, result.facts, match.entity_id if match else None)
    components.html(html, height=540, scrolling=True)

    st.subheader("Retrieved Evidence")
    st.dataframe(facts_to_dataframe(result.facts), width="stretch", hide_index=True)
    if method == "BM25" or result.used_lexical_fallback:
        st.caption("Hop 0 = found via BM25 lexical match, not graph traversal.")
    st.caption(f"Retrieved {len(result.facts)} relationships in {result.retrieval_time_ms:.1f}ms")

    if use_ollama:
        st.subheader("AI Summary")
        with st.spinner("Generating summary..."):
            answer = generate_answer(query, result)
        if answer.ollama_available:
            st.success(answer.summary)
        else:
            st.caption(f"_{answer.note}_")


if __name__ == "__main__":
    main()
