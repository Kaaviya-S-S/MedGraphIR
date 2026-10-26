import unittest
from unittest import mock

import networkx as nx
import pandas as pd

from evaluation.relevance_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.draft_judgments import _grade
from src.bm25_baseline import BM25Retriever, LexicalResult
from src.answer_generator import build_multi_entity_prompt, generate_answer
from src.data_pipeline import clean_indications, clean_interactions, link_entities
from src.fallback_retrieval import retrieve_with_fallback
from src.graph_pipeline import build_graph, extract_structured_relations
from src.retrieval import (
    EntityIndex,
    EntityMatch,
    MultiEntityRetrievalResult,
    QueryIntent,
    RetrievalResult,
    RetrievedFact,
    analyze_entity_query,
    build_entity_index,
    retrieve_multi_entity_relationships,
    search_entity,
)


def relation(record_id, source, target, evidence=""):
    return {
        "record_id": record_id,
        "source_entity": source,
        "source_entity_id": f"drug::{source.lower()}",
        "relationship": "INTERACTS_WITH",
        "target_entity": target,
        "target_entity_id": f"drug::{target.lower()}",
        "evidence": evidence,
        "attributes": "{}",
    }


class BM25RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = BM25Retriever(
            pd.DataFrame(
                [
                    relation("r1", "warfarin", "aspirin"),
                    relation("r2", "metformin", "warfarin", "warfarin interaction"),
                    relation("r3", "metformin", "aspirin"),
                ]
            )
        )

    def test_search_ranks_matching_records_and_respects_limit(self):
        results = self.retriever.search("aspirin", top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].record_id, "r1")
        self.assertGreater(results[0].score, 0)

    def test_search_returns_empty_for_no_matching_terms(self):
        self.assertEqual(self.retriever.search("xyzabc123"), [])

    def test_search_handles_empty_corpus_and_nonpositive_limit(self):
        empty = BM25Retriever(pd.DataFrame(columns=[
            "record_id", "source_entity", "source_entity_id", "relationship",
            "target_entity", "target_entity_id", "evidence", "attributes",
        ]))
        self.assertEqual(empty.search("warfarin"), [])
        self.assertEqual(self.retriever.search("warfarin", top_k=0), [])

    def test_bm25_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            BM25Retriever(pd.DataFrame(), k1=0)
        with self.assertRaises(ValueError):
            BM25Retriever(pd.DataFrame(), b=1.1)


class RelevanceMetricTests(unittest.TestCase):
    def test_binary_ranking_metrics(self):
        ranking = ["a", "x", "b"]
        relevant = {"a", "b"}
        self.assertAlmostEqual(precision_at_k(ranking, relevant, 2), 0.5)
        self.assertAlmostEqual(recall_at_k(ranking, relevant, 2), 0.5)
        self.assertAlmostEqual(average_precision_at_k(ranking, relevant, 3), 5 / 6)

    def test_ndcg_uses_graded_relevance(self):
        ideal = {"a": 2, "b": 1, "x": 0}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "x"], ideal, 3), 1.0)
        self.assertLess(ndcg_at_k(["b", "a", "x"], ideal, 3), 1.0)

    def test_invalid_cutoffs_and_undefined_recall_are_rejected(self):
        with self.assertRaises(ValueError):
            precision_at_k([], set(), 0)
        with self.assertRaises(ValueError):
            recall_at_k([], set(), 5)


class DraftJudgmentTests(unittest.TestCase):
    def test_interaction_query_grades_matching_relationship_as_direct(self):
        row = pd.Series({
            "query": "Find medicines that interact with Warfarin.",
            "source_entity": "Aspirin",
            "relationship": "INTERACTS_WITH",
            "target_entity": "Warfarin",
        })
        grade, _ = _grade(row)
        self.assertEqual(grade, 2)

    def test_wrong_relation_for_matching_entity_is_context(self):
        row = pd.Series({
            "query": "Find medicines that interact with Warfarin.",
            "source_entity": "Warfarin",
            "relationship": "TREATS",
            "target_entity": "Other",
        })
        grade, _ = _grade(row)
        self.assertEqual(grade, 1)

    def test_record_without_query_entity_is_irrelevant(self):
        row = pd.Series({
            "query": "Find medicines that interact with Warfarin.",
            "source_entity": "Aspirin",
            "relationship": "INTERACTS_WITH",
            "target_entity": "Other",
        })
        grade, _ = _grade(row)
        self.assertEqual(grade, 0)


class MultiEntityQueryAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.index = build_entity_index(
            pd.DataFrame(
                [
                    {
                        "entity_id": "drug::warfarin",
                        "canonical_name": "Warfarin",
                        "entity_type": "DRUG",
                        "mention_count": 20,
                        "aliases": "Warfarin",
                    },
                    {
                        "entity_id": "drug::aspirin",
                        "canonical_name": "Aspirin",
                        "entity_type": "DRUG",
                        "mention_count": 15,
                        "aliases": "Aspirin",
                    },
                    {
                        "entity_id": "disease::diabetes",
                        "canonical_name": "Type 2 diabetes",
                        "entity_type": "DISEASE",
                        "mention_count": 10,
                        "aliases": "Diabetes|Type 2 diabetes",
                    },
                    {
                        "entity_id": "disease::asthma",
                        "canonical_name": "Asthma",
                        "entity_type": "DISEASE",
                        "mention_count": 8,
                        "aliases": "Asthma",
                    },
                ]
            )
        )

    def test_finds_multiple_entities_and_prefers_longest_alias(self):
        analysis = analyze_entity_query(
            "What conditions link Type 2 diabetes and asthma?", self.index
        )

        self.assertEqual(
            [entity.entity_id for entity in analysis.entities],
            ["disease::diabetes", "disease::asthma"],
        )
        self.assertEqual(analysis.intent, QueryIntent.AMBIGUOUS)

    def test_classifies_direct_relationship_questions(self):
        analysis = analyze_entity_query(
            "Does Warfarin interact with Aspirin?", self.index
        )

        self.assertEqual(analysis.intent, QueryIntent.DIRECT_RELATIONSHIP)

    def test_classifies_explicit_and_query(self):
        analysis = analyze_entity_query(
            "What drugs interact with both Warfarin and Aspirin?", self.index
        )

        self.assertEqual(analysis.intent, QueryIntent.AND)

    def test_classifies_set_query_with_implicit_and(self):
        analysis = analyze_entity_query(
            "What drugs interact with Warfarin and Aspirin?", self.index
        )

        self.assertEqual(analysis.intent, QueryIntent.AND)

    def test_classifies_existence_question_for_shared_drugs_as_and(self):
        analysis = analyze_entity_query(
            "Are there drugs that treat Type 2 diabetes and asthma?",
            self.index,
        )

        self.assertEqual(analysis.intent, QueryIntent.AND)

    def test_classifies_or_query(self):
        analysis = analyze_entity_query(
            "Which drugs interact with Warfarin or Aspirin?", self.index
        )

        self.assertEqual(analysis.intent, QueryIntent.OR)

    def test_ambiguous_multi_entity_query_is_not_guessed(self):
        analysis = analyze_entity_query("Warfarin and Aspirin", self.index)

        self.assertEqual(analysis.intent, QueryIntent.AMBIGUOUS)

    def test_single_and_unmatched_queries_keep_distinct_intents(self):
        single = analyze_entity_query("What interacts with Warfarin?", self.index)
        unmatched = analyze_entity_query("xyzabc123", self.index)

        self.assertEqual(single.intent, QueryIntent.SINGLE_ENTITY)
        self.assertEqual(unmatched.intent, QueryIntent.NONE)

    def test_repeated_alias_for_same_entity_is_deduplicated(self):
        analysis = analyze_entity_query("Warfarin and warfarin", self.index)

        self.assertEqual(len(analysis.entities), 1)
        self.assertEqual(analysis.entities[0].entity_id, "drug::warfarin")

    def test_numeric_result_count_does_not_match_numbered_entity_name(self):
        matches = search_entity(
            "What are the top 2 drugs to treat cancer?", self.index
        )

        self.assertNotIn(
            "disease::diabetes",
            {match.entity_id for match in matches},
        )


class MultiEntityRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = build_entity_index(
            pd.DataFrame(
                [
                    ("drug::warfarin", "Warfarin", "DRUG", "Warfarin"),
                    ("drug::aspirin", "Aspirin", "DRUG", "Aspirin"),
                    ("drug::metformin", "Metformin", "DRUG", "Metformin"),
                    ("drug::shared", "Shared drug", "DRUG", "Shared drug"),
                    ("disease::diabetes", "Diabetes", "DISEASE", "Diabetes"),
                    ("disease::asthma", "Asthma", "DISEASE", "Asthma"),
                ],
                columns=["entity_id", "canonical_name", "entity_type", "aliases"],
            ).assign(mention_count=1)
        )
        self.graph = nx.DiGraph()
        for entity_id, info in self.index.node_info.items():
            self.graph.add_node(entity_id, **info)

        self.graph.add_edge(
            "drug::aspirin",
            "drug::warfarin",
            relationship="INTERACTS_WITH",
            record_id="ddinter_reverse",
            evidence="DDInter severity: Major",
            severity="Major",
        )
        self.graph.add_edge(
            "drug::metformin",
            "disease::diabetes",
            relationship="TREATS",
            record_id="treats_diabetes_metformin",
            evidence="Metformin indication",
            n_curators=2,
            n_resources=1,
        )
        self.graph.add_edge(
            "drug::aspirin",
            "disease::diabetes",
            relationship="TREATS",
            record_id="treats_diabetes_aspirin",
            evidence="Aspirin indication",
            n_curators=1,
            n_resources=1,
        )
        self.graph.add_edge(
            "drug::shared",
            "disease::asthma",
            relationship="TREATS",
            record_id="treats_asthma_shared",
            evidence="Shared drug indication",
            n_curators=1,
            n_resources=1,
        )
        self.graph.add_edge(
            "drug::shared",
            "disease::diabetes",
            relationship="TREATS",
            record_id="treats_diabetes_shared",
            evidence="Shared drug indication",
            n_curators=1,
            n_resources=1,
        )

    def test_direct_relationship_finds_reverse_stored_interaction(self):
        result = retrieve_multi_entity_relationships(
            "Does Warfarin interact with Aspirin?", self.graph, self.index
        )

        self.assertEqual(result.intent, QueryIntent.DIRECT_RELATIONSHIP)
        self.assertEqual([fact.record_id for fact in result.direct_facts], ["ddinter_reverse"])
        self.assertEqual(result.direct_facts[0].source_entity_id, "drug::aspirin")

    def test_and_returns_only_shared_result_entities_and_supporting_records(self):
        result = retrieve_multi_entity_relationships(
            "Which diseases are treated by both Metformin and Aspirin?",
            self.graph,
            self.index,
        )

        self.assertEqual(result.intent, QueryIntent.AND)
        self.assertEqual(result.result_entity_ids, ["disease::diabetes"])
        self.assertEqual(
            {fact.record_id for fact in result.facts},
            {"treats_diabetes_metformin", "treats_diabetes_aspirin"},
        )

    def test_and_supports_three_or_more_named_entities(self):
        result = retrieve_multi_entity_relationships(
            "Which diseases are treated by Metformin, Aspirin, and Shared drug?",
            self.graph,
            self.index,
        )

        self.assertEqual(result.intent, QueryIntent.AND)
        self.assertEqual(result.result_entity_ids, ["disease::diabetes"])
        self.assertEqual(
            {fact.record_id for fact in result.facts},
            {
                "treats_diabetes_metformin",
                "treats_diabetes_aspirin",
                "treats_diabetes_shared",
            },
        )

    def test_or_returns_union_of_result_entities(self):
        result = retrieve_multi_entity_relationships(
            "Which drugs treat Diabetes or Asthma?", self.graph, self.index
        )

        self.assertEqual(result.intent, QueryIntent.OR)
        self.assertEqual(
            set(result.result_entity_ids),
            {"drug::metformin", "drug::aspirin", "drug::shared"},
        )
        self.assertEqual(
            {fact.record_id for fact in result.facts},
            {
                "treats_diabetes_metformin",
                "treats_diabetes_aspirin",
                "treats_diabetes_shared",
                "treats_asthma_shared",
            },
        )

    def test_mixed_drug_disease_query_returns_directional_treatment_fact(self):
        result = retrieve_multi_entity_relationships(
            "Does Metformin treat Diabetes?", self.graph, self.index
        )

        self.assertEqual(result.intent, QueryIntent.DIRECT_RELATIONSHIP)
        self.assertEqual([fact.record_id for fact in result.direct_facts], ["treats_diabetes_metformin"])
        self.assertEqual(
            (result.direct_facts[0].source_entity_id, result.direct_facts[0].target_entity_id),
            ("drug::metformin", "disease::diabetes"),
        )

    def test_indirect_path_is_not_reported_as_direct_evidence(self):
        result = retrieve_multi_entity_relationships(
            "Does Warfarin treat Metformin?",
            self.graph,
            self.index,
            include_context=True,
        )

        self.assertEqual(result.direct_facts, [])
        self.assertEqual(result.facts, [])
        self.assertTrue(result.context_facts)
        self.assertNotIn("ddinter_reverse", {fact.record_id for fact in result.direct_facts})

    def test_ambiguous_query_returns_no_guessed_results(self):
        result = retrieve_multi_entity_relationships(
            "Warfarin and Aspirin", self.graph, self.index
        )

        self.assertEqual(result.intent, QueryIntent.AMBIGUOUS)
        self.assertEqual(result.facts, [])
        self.assertEqual(result.context_facts, [])

    def test_context_is_separate_from_primary_results(self):
        result = retrieve_multi_entity_relationships(
            "Does Warfarin interact with Aspirin?",
            self.graph,
            self.index,
            include_context=True,
        )

        self.assertEqual([fact.record_id for fact in result.direct_facts], ["ddinter_reverse"])
        self.assertNotIn(
            "ddinter_reverse",
            {fact.record_id for fact in result.context_facts},
        )


class MultiEntityAnswerTests(unittest.TestCase):
    def setUp(self):
        asthma = EntityMatch(
            entity_id="disease::asthma",
            canonical_name="Asthma",
            entity_type="DISEASE",
            match_type="SUBSTRING",
            match_score=0.5,
        )
        migraine = EntityMatch(
            entity_id="disease::migraine",
            canonical_name="Migraine",
            entity_type="DISEASE",
            match_type="SUBSTRING",
            match_score=0.5,
        )
        self.result = MultiEntityRetrievalResult(
            query="Are there drugs that treat asthma and migraine?",
            entities=[asthma, migraine],
            intent=QueryIntent.AND,
            facts=[
                RetrievedFact(
                    source_entity_id="drug::prednisone",
                    source_name="Prednisone",
                    relationship="TREATS",
                    target_entity_id="disease::asthma",
                    target_name="Asthma",
                    hop=1,
                    record_id="prednisone_asthma",
                    evidence="Asthma indication",
                    rank_score=10,
                ),
                RetrievedFact(
                    source_entity_id="drug::prednisone",
                    source_name="Prednisone",
                    relationship="TREATS",
                    target_entity_id="disease::migraine",
                    target_name="Migraine",
                    hop=1,
                    record_id="prednisone_migraine",
                    evidence="Migraine indication",
                    rank_score=10,
                ),
            ],
            context_facts=[
                RetrievedFact(
                    source_entity_id="drug::other",
                    source_name="Other drug",
                    relationship="TREATS",
                    target_entity_id="disease::asthma",
                    target_name="Asthma",
                    hop=2,
                    record_id="context_only",
                    evidence="Context only",
                    rank_score=5,
                )
            ],
        )

    def test_multi_entity_prompt_explains_and_and_excludes_neighborhood_context(self):
        prompt = build_multi_entity_prompt(self.result.query, self.result)

        self.assertIn("Query intent: and", prompt)
        self.assertIn("supporting facts", prompt)
        self.assertIn("Prednisone --[TREATS]--> Asthma", prompt)
        self.assertIn("Prednisone --[TREATS]--> Migraine", prompt)
        self.assertNotIn("Other drug", prompt)
        self.assertNotIn("Context only", prompt)

    def test_generate_answer_accepts_multi_entity_result(self):
        ollama = mock.Mock()
        ollama.generate.return_value = {"response": "Prednisone is linked to both diseases."}
        with mock.patch.dict("sys.modules", {"ollama": ollama}):
            answer = generate_answer(self.result.query, self.result)

        self.assertTrue(answer.ollama_available)
        self.assertEqual(answer.facts_used, 2)
        self.assertEqual(answer.summary, "Prednisone is linked to both diseases.")
        self.assertIn("Retrieved primary facts", ollama.generate.call_args.kwargs["prompt"])


class FallbackRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.index = EntityIndex(alias_lookup={}, node_info={})

    @staticmethod
    def _lexical_result():
        return LexicalResult(
            record_id="r1",
            source_entity="Aspirin",
            source_entity_id="drug::aspirin",
            relationship="INTERACTS_WITH",
            target_entity="Warfarin",
            target_entity_id="drug::warfarin",
            evidence="",
            attributes={},
            score=1.0,
        )

    def test_matched_entity_uses_graph_result_and_skips_lexical_search(self):
        matched = RetrievalResult(
            query="q",
            matched_entity=EntityMatch(
                entity_id="drug::x", canonical_name="X", entity_type="DRUG",
                match_type="EXACT", match_score=1.0,
            ),
        )
        bm25 = mock.Mock()
        with mock.patch("src.fallback_retrieval.retrieve_relationships", return_value=matched):
            result = retrieve_with_fallback("q", None, self.index, bm25)

        self.assertIs(result, matched)
        bm25.search.assert_not_called()

    def test_matched_entity_does_not_load_lazy_bm25_retriever(self):
        matched = RetrievalResult(
            query="q",
            matched_entity=EntityMatch(
                entity_id="drug::x", canonical_name="X", entity_type="DRUG",
                match_type="EXACT", match_score=1.0,
            ),
        )
        bm25_loader = mock.Mock()
        with mock.patch("src.fallback_retrieval.retrieve_relationships", return_value=matched):
            result = retrieve_with_fallback("q", None, self.index, None, bm25_loader=bm25_loader)

        self.assertIs(result, matched)
        bm25_loader.assert_not_called()

    def test_no_entity_match_loads_lazy_bm25_retriever(self):
        bm25 = mock.Mock()
        bm25.search.return_value = [self._lexical_result()]
        bm25_loader = mock.Mock(return_value=bm25)
        with mock.patch(
            "src.fallback_retrieval.retrieve_relationships",
            return_value=RetrievalResult(query="q"),
        ):
            result = retrieve_with_fallback("q", None, self.index, None, bm25_loader=bm25_loader)

        bm25_loader.assert_called_once_with()
        self.assertTrue(result.used_lexical_fallback)

    def test_no_entity_match_falls_back_to_lexical_results(self):
        bm25 = mock.Mock()
        bm25.search.return_value = [self._lexical_result()]
        with mock.patch(
            "src.fallback_retrieval.retrieve_relationships",
            return_value=RetrievalResult(query="q"),
        ):
            result = retrieve_with_fallback("q", None, self.index, bm25)

        self.assertTrue(result.used_lexical_fallback)
        self.assertIsNone(result.matched_entity)
        self.assertEqual([fact.record_id for fact in result.facts], ["r1"])

    def test_empty_graph_result_kept_when_lexical_finds_nothing(self):
        bm25 = mock.Mock()
        bm25.search.return_value = []
        graph_result = RetrievalResult(query="q")
        with mock.patch("src.fallback_retrieval.retrieve_relationships", return_value=graph_result):
            result = retrieve_with_fallback("q", None, self.index, bm25)

        self.assertIs(result, graph_result)
        self.assertFalse(result.used_lexical_fallback)
        self.assertEqual(result.facts, [])


class GraphPipelineTests(unittest.TestCase):
    def test_build_keeps_cleaned_source_facts_and_their_provenance(self):
        interactions = pd.DataFrame(
            [
                {
                    "DDInterID_A": "DD1",
                    "DDInterID_B": "DD2",
                    "Drug_A": " Warfarin ",
                    "Drug_B": "Aspirin",
                    "Level": "Major",
                    "atc_code": "A",
                    "atc_label": "Alimentary tract",
                },
                {
                    "DDInterID_A": "DD1",
                    "DDInterID_B": "DD2",
                    "Drug_A": "Warfarin",
                    "Drug_B": "Aspirin",
                    "Level": "Moderate",
                    "atc_code": "B",
                    "atc_label": "Blood",
                },
                {
                    "DDInterID_A": "DD3",
                    "DDInterID_B": "DD4",
                    "Drug_A": " ",
                    "Drug_B": "Missing",
                    "Level": "Minor",
                    "atc_code": "A",
                    "atc_label": "Alimentary tract",
                },
            ]
        )
        indications = pd.DataFrame(
            [
                {
                    "drugbank_id": "DB1",
                    "doid_id": "DOID:1",
                    "drug": "Warfarin",
                    "disease": "Heart failure",
                    "category": "DM",
                    "n_curators": 2,
                    "n_resources": 3,
                },
                {
                    "drugbank_id": "DB2",
                    "doid_id": "DOID:2",
                    "drug": "Aspirin",
                    "disease": "Other disease",
                    "category": "NOT",
                    "n_curators": 1,
                    "n_resources": 1,
                },
            ]
        )

        interactions = clean_interactions(interactions)
        indications = clean_indications(indications)
        entities, interactions, indications = link_entities(interactions, indications)
        relations = extract_structured_relations(interactions, indications)
        graph = build_graph(entities, relations)

        self.assertEqual(len(interactions), 1)
        self.assertEqual(len(indications), 1)
        self.assertEqual(len(relations), 2)
        self.assertEqual(
            graph["drug::warfarin"]["drug::aspirin"]["severity"],
            "Major",
        )
        self.assertEqual(
            graph["drug::warfarin"]["disease::heart_failure"]["relationship"],
            "TREATS",
        )
        self.assertFalse(graph.has_edge("drug::aspirin", "drug::warfarin"))


if __name__ == "__main__":
    unittest.main()
