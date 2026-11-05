# 🧬 Medical Knowledge Graph-Based Information Retrieval System

MedGraphIR models drug-drug interactions and drug-disease relationships as a graph, then answers
natural-language queries by matching an entity and retrieving its 1-2 hop neighborhood — not by 
asking a language model to recall facts from memory.

Retrieval happens over a real knowledge graph built from **DDInter 2.0**
(~222k drug-drug interactions) and **PharmacotherapyDB** (physician-curated
drug-disease indications), compared against a traditional TF-IDF baseline
and a hybrid method, with results measured on a 30-query evaluation set —
not assumed.

> **This is not a diagnosis tool.** It retrieves and summarizes existing,
> sourced relationships. See [Medical Safety](#medical-safety) below.

## Architecture

```mermaid
flowchart TD
    A[DDInter 2.0 + PharmacotherapyDB] --> B[Preprocessing]
    B --> C[Entity Extraction / Normalization]
    C --> D[Relation Extraction]
    D --> E[NetworkX Knowledge Graph]
    E --> F[Query Processing]
    F --> G[Entity Matching]
    G --> H[1-2 Hop Graph Retrieval]
    H --> I[Relevant Facts + Source Provenance]
    I --> J[Optional Ollama Summary]
    I --> K[Streamlit UI]
    J --> K
```

The knowledge graph is built **offline** by a standalone script; the
Streamlit app only ever loads and queries the saved graph, never rebuilds
it per request.

## Features

- **Graph-based retrieval** — tiered entity matching (exact / substring /
  partial / fuzzy) with auto-adaptive 1-2 hop traversal
- **TF-IDF keyword baseline** and a **hybrid (TF-IDF + graph)** method for
  comparison
- **Measured evaluation**, not assumed — Precision/Recall/F1/P@K over a
  30-query, graph-derived ground-truth set
- **Optional local LLM summary** (Ollama) grounded only in retrieved facts
  — never invents medical information
- **Streamlit UI** with Search, Evaluation, and About pages

## Quick Start

```bash
pip install -r requirements.txt

python build_graph.py        # build the knowledge graph once (~25s)
streamlit run app.py         # launch the UI
```

Try a query: *"What drugs interact with Warfarin?"*

To reproduce the evaluation numbers:

```bash
python evaluation/evaluate.py
```

## Results

<<<<<<< Updated upstream
Measured on the full dataset, 30 test queries:

| Method | Precision | Recall | F1 | Avg. time |
|---|---|---|---|---|
| **Graph retrieval** | 1.000 | 0.699 | 0.717 | 2.1ms |
| Hybrid (TF-IDF + Graph) | 0.696 | 0.598 | 0.438 | 30.4ms |
| TF-IDF baseline | 0.586 | 0.465 | 0.348 | 26.9ms |

Graph retrieval wins on both accuracy and speed. Hybrid never beats pure
graph on this dataset (0/30 query wins) — reported honestly, not assumed
away; see the documentation for why.

## Tech Stack

Python, Pandas, NetworkX, scikit-learn, Streamlit, Pydantic, Ollama
(optional).

## Project Structure
=======
Retrieval was evaluated by comparing ranked results from the knowledge graph
and BM25 against relevance judgments. We report Precision@K (P@K), Recall@20,
Mean Average Precision (MAP@20), and Normalized Discounted Cumulative Gain
(NDCG@10).

| Query set | Method | P@5 | P@20 | Recall@20 | MAP@20 | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|
| Single-entity (30 queries) | Graph | 0.807 | 0.573 | 0.648 | 0.865 | 0.959 |
| Single-entity (30 queries) | BM25 | 0.780 | 0.670 | 0.690 | 0.933 | 0.798 |
| Multi-entity (7 queries) | Graph | 0.743 | 0.436 | 0.870 | 1.000 | 1.000 |
| Multi-entity (7 queries) | BM25 | 0.314 | 0.279 | 0.429 | 0.539 | 0.558 |

Results are mixed for single-entity queries: graph retrieval ranks some
relevant results higher, while BM25 retrieves more relevant results overall.
Graph retrieval scores higher in the current multi-entity test, but that
benchmark is small and its relevance labels were initially suggested from
graph results, so the comparison may favor graph retrieval. Treat those
multi-entity scores as preliminary.

## Limitations

- Interaction evidence is only a severity label (DDInter's bulk export has no mechanism text).
- Disease coverage is limited to the 97 diseases in PharmacotherapyDB.
- Entity linking is name-based, not a full ontology linker (e.g. brand names may not resolve).
- Multi-entity intent detection uses simple wording rules; unclear queries are reported as ambiguous rather than interpreted freely.

## Project structure
>>>>>>> Stashed changes

```
├── data/               Raw data, processed tables, saved graph
├── src/                Pipeline modules (preprocessing, entity linking,
│                       relation extraction, graph, retrieval, hybrid,
│                       keyword baseline, Ollama summarization)
├── evaluation/          Test queries, ground truth, evaluation script
├── build_graph.py       Standalone graph-build script
├── app.py               Streamlit UI
└── DOCUMENTATION.md      Full technical write-up
```

## Data Sources

- **[DDInter 2.0](https://ddinter2.scbdd.com/)** — drug-drug interactions,
  severity-labeled
- **[PharmacotherapyDB 1.0](https://github.com/dhimmel/indications)** —
  physician-curated drug-disease indications

## Medical Safety

This system is an **academic information retrieval prototype**. Retrieved
information should not be used as a substitute for professional medical
advice.
