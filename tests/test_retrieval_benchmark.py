import unittest

from app.schemas import DocumentPage, ExtractedDocument
from evaluation.retrieval_benchmark import load_suite, render_report, summarize, validate_suite
from evaluation.runner import retrieval_metrics


class RetrievalBenchmarkTests(unittest.TestCase):
    def test_all_expected_evidence_exists_in_actual_chunks(self):
        cases, documents = load_suite()
        self.assertEqual(len(cases), 41)
        self.assertEqual(len(documents), 18)
        self.assertEqual({case["category"] for case in cases}, {
            "original_policy", "paraphrase", "near_match", "exact_term", "cross_document", "identifier"
        })

    def test_incorrect_gold_labels_fail_before_provider_calls(self):
        documents = [ExtractedDocument(source_file="a.txt", pages=[DocumentPage(page_number=1, text="a rule")])]
        cases = [{"id": "bad", "question": "Which rule?", "evidence": [{"source_file": "a.txt", "anchor": "absent"}]}]
        with self.assertRaisesRegex(ValueError, "Evidence missing"):
            validate_suite(cases, documents)
        cases[0]["evidence"][0]["anchor"] = "a rule"
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_suite(cases * 2, documents)

    def test_missing_candidates_count_as_failure_not_excluded(self):
        case = {"evidence": [{"source_file": "a.txt", "anchor": "rule"}]}
        missing = {"id": "missing", "database_ms": 12, **retrieval_metrics(case, [])}
        metrics = summarize([missing])
        self.assertEqual(metrics["questions"], 1)
        self.assertEqual(metrics["mean_recall_at"]["5"], 0)
        self.assertEqual(metrics["all_evidence_counts"]["20"], 0)
        self.assertEqual(metrics["failures_at_5"], ["missing"])

    def test_report_uses_recorded_denominators_and_failure_ranks(self):
        def result(rank):
            case = {"id": "random_mfa_prompt", "database_ms": 1.25,
                    "evidence_ranks": [rank],
                    "recall_at": {str(k): float(rank <= k) for k in (1, 3, 5, 20)},
                    "all_evidence_at": {str(k): rank <= k for k in (1, 3, 5, 20)}}
            return {"summary": summarize([case]), "cases": [case], "by_category": {}}
        report = {"generated_at": "2026-09-30T00:00:00Z", "chunk_count": 37,
                  "dataset_sha256": "test", "strategies": {
                      "vector": result(1), "lexical": result(20), "hybrid": result(12)}}
        text = render_report(report)
        self.assertIn("| Hybrid | 0/1 | 0/1 | 0/1 | 1/1 |", text)
        self.assertIn("hybrid rank 12", text)
        self.assertIn("| `random_mfa_prompt` | [1] | [12] |", text)
        self.assertIn("excluding query embedding", text)


if __name__ == "__main__":
    unittest.main()
