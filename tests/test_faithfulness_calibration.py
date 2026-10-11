import unittest

from evaluation.faithfulness_calibration import calibration_report, human_agreement, rescore, wilson_interval


def result(case_id, faithful, claims=()):
    return {"id": case_id, "faithful": faithful, "claims": list(claims)}


def claim(text, verdict, judge_error=False):
    return {"claim": text, "verdict": verdict, "judge_error": judge_error, "evidence_quote": None}


CASES = [
    {"id": "a__control", "mutation": "none", "expected": "faithful"},
    {"id": "b__control", "mutation": "none", "expected": "faithful"},
    {"id": "a__number", "mutation": "number", "expected": "unfaithful", "marker": "12 characters",
     "expected_claim_label": "contradicted"},
    {"id": "b__invented_fact", "mutation": "invented_fact", "expected": "unfaithful", "marker": "special character",
     "expected_claim_label": "unsupported"},
    {"id": "c__number", "mutation": "number", "expected": "unfaithful", "marker": "72 hours",
     "expected_claim_label": "contradicted"},
]


class CalibrationTests(unittest.TestCase):
    def test_wilson_interval_is_wide_for_small_samples(self):
        low, high = wilson_interval(9, 10)
        self.assertLess(low, 0.6)
        self.assertGreater(high, 0.98)
        self.assertIsNone(wilson_interval(0, 0))

    def test_report_separates_recall_precision_and_false_alarms(self):
        report = calibration_report(CASES, [
            result("a__control", True),
            result("b__control", False),
            result("a__number", False, [claim("Minimum is 12 characters", "contradicted")]),
            result("b__invented_fact", False, [claim("Needs a special character", "contradicted")]),
            result("c__number", True, [claim("Install within 72 hours", "supported")]),
        ])
        self.assertEqual(report["recall_unfaithful"]["count"], "2/3")
        self.assertEqual(report["precision_unfaithful"]["count"], "2/3")
        self.assertEqual(report["false_alarm_rate"]["count"], "1/2")
        self.assertEqual(report["recall_by_mutation"]["number"]["count"], "1/2")
        self.assertEqual((report["target_claim_found"], report["target_verdict_matches"]), (3, 1))
        self.assertEqual(report["false_alarm_ids"], ["b__control"])

    def test_judge_errors_are_excluded_not_counted_as_misses(self):
        report = calibration_report(CASES[:3], [
            result("a__control", True), {"id": "b__control", "error_type": "JudgeIncomplete"},
            result("a__number", False, [claim("Minimum is 12 characters", "contradicted")]),
        ])
        self.assertEqual((report["judged"], report["errors"]), (2, 1))
        self.assertEqual(report["recall_unfaithful"]["count"], "1/1")

    def test_human_agreement_reports_recall_on_claims_the_human_flagged(self):
        results = {"results": [{"id": "q", "claims": [
            claim("one", "supported"), claim("two", "supported"), claim("three", "unsupported"), claim("four", "supported"),
        ]}]}
        labels = {"q": {
            "one": {"label": "supported", "note": ""}, "two": {"label": "unsupported", "note": "adds scope"},
            "three": {"label": "unsupported", "note": ""}, "four": {"label": "not_a_claim", "note": ""},
        }}
        report = human_agreement(results, labels)
        self.assertEqual((report["labeled_claims"], report["not_a_claim"]), (3, 1))
        self.assertEqual(report["agreement"]["count"], "2/3")
        self.assertEqual(report["recall_on_human_flags"]["count"], "1/2")
        self.assertEqual(report["disagreements"][0]["claim"], "two")

    def test_rescore_reapplies_quote_checks_to_saved_verdicts(self):
        case = {"sources": [{"number": 1, "text": "Discovery time and whether it was unlocked."}]}
        saved = {"id": "x", "faithful": False, "usage": {"input_tokens": 1, "output_tokens": 1}, "claims": [{
            "claim": "Give discovery time", "cited": [1], "verdict": "supported", "evidence_source": 1,
            "evidence_quote": "discovery time, and whether it was unlocked", "judge_error": True,
        }]}
        rescored = rescore(case, saved)
        self.assertTrue(rescored["faithful"])
        self.assertEqual(rescored["usage"], saved["usage"])


if __name__ == "__main__":
    unittest.main()
