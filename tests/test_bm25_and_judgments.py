import unittest
from unittest import mock

import pandas as pd

from evaluation.relevance_metrics import (
    average_precision_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from evaluation.draft_judgments import _grade
from src.bm25_baseline import BM25Retriever, LexicalResult
from src.fallback_retrieval import retrieve_with_fallback
from src.retrieval import EntityIndex, EntityMatch, RetrievalResult


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


if __name__ == "__main__":
    unittest.main()
