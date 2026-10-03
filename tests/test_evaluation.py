import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.schemas import RagSource
from evaluation.runner import cited_relevant_evidence, evaluate_qa_case, looks_like_abstention, prepare_hits, rate, retrieval_metrics


class EvaluationScoringTests(unittest.TestCase):
    def test_rank_twenty_boundary_and_missing_evidence(self):
        def hit(index, text):
            return RagSource(reference=f"[{index}]", chunk_id=index, document_id=1,
                             source_file="policy.pdf", page_number=1, chunk_index=index,
                             text=text, distance=0.1)
        case = {"evidence": [
            {"source_file": "policy.pdf", "anchor": "required rule"},
            {"source_file": "missing.pdf", "anchor": "other rule"},
        ]}
        hits = [hit(index, "irrelevant") for index in range(1, 20)] + [hit(20, "required rule")]
        metrics = retrieval_metrics(case, hits)
        self.assertEqual(metrics["evidence_ranks"], [20, None])
        self.assertEqual(metrics["recall_at"]["5"], 0)
        self.assertEqual(metrics["recall_at"]["20"], 0.5)
        self.assertFalse(metrics["all_evidence_at"]["20"])

    def test_duplicate_overlapping_chunks_do_not_count_as_extra_evidence(self):
        source = RagSource(reference="[1]", chunk_id=1, document_id=1,
                           source_file="policy.pdf", page_number=1, chunk_index=0,
                           text="required rule", distance=0.1)
        case = {"evidence": [
            {"source_file": "policy.pdf", "anchor": "required rule"},
            {"source_file": "policy.pdf", "anchor": "absent rule"},
        ]}
        metrics = retrieval_metrics(case, [source, source.model_copy(update={"chunk_id": 2})])
        self.assertEqual(metrics["recall_at"]["3"], 0.5)

    def test_recall_tracks_each_required_source_at_multiple_ranks(self):
        def hit(chunk_id, source_file, text):
            return RagSource(reference=f"[{chunk_id}]", chunk_id=chunk_id, document_id=5,
                             source_file=source_file, page_number=1, chunk_index=chunk_id,
                             text=text, distance=chunk_id / 100)

        hits = [hit(1, "a.pdf", "first anchor")]
        hits += [hit(i, "other.pdf", "irrelevant") for i in range(2, 5)]
        hits += [hit(5, "b.pdf", "second anchor")]
        case = {"evidence": [
            {"source_file": "a.pdf", "anchor": "first anchor"},
            {"source_file": "b.pdf", "anchor": "second anchor"},
        ]}
        metrics = retrieval_metrics(case, hits)
        self.assertEqual(metrics["evidence_ranks"], [1, 5])
        self.assertEqual(metrics["recall_at"], {"1": 0.5, "3": 0.5, "5": 1.0, "20": 1.0})
        self.assertEqual(metrics["all_evidence_at"], {"1": False, "3": False, "5": True, "20": True})
        self.assertFalse(evaluate_qa_case({**case, "id": "two", "category": "cross_document"}, hits, "retrieval")["retrieval_hit"])

    def test_citation_must_point_to_chunk_with_expected_evidence(self):
        sources = [
            RagSource(reference="[1]", chunk_id=3, document_id=5, source_file="policy.pdf",
                      page_number=1, chunk_index=0, text="Passwords must contain at least 14 characters.", distance=0.1),
            RagSource(reference="[2]", chunk_id=4, document_id=5, source_file="policy.pdf",
                      page_number=1, chunk_index=1, text="SMS is not approved.", distance=0.2),
        ]
        evidence = {"source_file": "policy.pdf", "anchor": "at least 14 characters"}
        self.assertTrue(cited_relevant_evidence("14 characters [1].", sources, evidence))
        self.assertFalse(cited_relevant_evidence("14 characters [2].", sources, evidence))
        self.assertFalse(cited_relevant_evidence("14 characters.", sources, evidence))
        self.assertFalse(cited_relevant_evidence("14 characters [1].", sources, {**evidence, "source_file": "other.pdf"}))

    def test_cross_document_retrieval_requires_both_sources(self):
        hits = [
            RagSource(reference="[1]", chunk_id=3, document_id=5, source_file="device.pdf",
                      page_number=1, chunk_index=0, text="Only managed laptops may use VPN.", distance=0.1),
            RagSource(reference="[2]", chunk_id=4, document_id=6, source_file="vpn.pdf",
                      page_number=1, chunk_index=0, text="VPN requires a company-issued laptop.", distance=0.2),
        ]
        case = {
            "id": "cross", "category": "cross_document", "question": "Which laptop?",
            "evidence": [
                {"source_file": "device.pdf", "anchor": "Only managed laptops"},
                {"source_file": "vpn.pdf", "anchor": "company-issued laptop"},
            ],
            "answer_terms": ["laptop"],
        }
        self.assertTrue(evaluate_qa_case(case, hits, "retrieval")["passed"])
        self.assertFalse(evaluate_qa_case(case, hits[:1], "retrieval")["passed"])

    def test_abstention_check_is_an_explicit_phrase_heuristic(self):
        self.assertTrue(looks_like_abstention("The supplied documents do not specify that cap."))
        self.assertFalse(looks_like_abstention("The monthly cap is $100."))

    def test_rate_counts_failed_cases(self):
        self.assertEqual(rate([{"passed": True}, {"passed": False}, {"error_type": "RuntimeError"}]), 0.333)
        self.assertEqual(rate([{"passed": True}, {"passed": False}, {"error_type": "RuntimeError"}], completed_only=True), 0.5)

    def test_questions_are_embedded_in_one_query_batch(self):
        vector = [1.0] + [0.0] * 1023
        row = {"chunk_id": 3, "document_id": 5, "source_file": "policy.pdf", "page_number": 1,
               "chunk_index": 0, "text": "Policy", "distance": 0.1}
        cases = [{"id": "one", "question": "First?"}, {"id": "two", "question": "Second?"}]
        settings = SimpleNamespace(voyage_api_key=SecretStr("test-key"))
        with patch("evaluation.runner.get_settings", return_value=settings):
            with patch("evaluation.runner.embed_texts", return_value=[vector, vector]) as embed:
                with patch("evaluation.runner.embedding_repository.search_chunks_for_documents", return_value=[row]) as search:
                    hits = prepare_hits(cases, [5, 6])
        embed.assert_called_once_with(
            ["First?", "Second?"], "test-key", "voyage-4", 1024, input_type="query"
        )
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args.args[-1], [5, 6])
        self.assertEqual(hits["one"][0].chunk_id, 3)
        self.assertEqual(hits["two"][0].chunk_id, 3)


if __name__ == "__main__":
    unittest.main()
