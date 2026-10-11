import unittest

from evaluation.label_faithfulness import pending_answers, parse_label


ANSWERS = {"answers": [
    {"id": "sms_mfa", "question": "Is SMS approved?", "answer": "No [1].",
     "sources": [{"number": 1, "source_file": "p.pdf", "page_number": 1, "text": "SMS is not approved."}]},
    {"id": "failed", "error_type": "AnthropicRequestError"},
]}
RESULTS = {"results": [
    {"id": "sms_mfa", "claims": [
        {"claim": "SMS is not approved", "verdict": "supported", "evidence_quote": "SMS is not approved."},
        {"claim": "Keys are approved", "verdict": "unsupported", "evidence_quote": None},
    ]},
    {"id": "failed", "error_type": "JudgeIncomplete"},
]}


class LabelFaithfulnessTests(unittest.TestCase):
    def test_parse_label_reads_letter_and_optional_note(self):
        self.assertEqual(parse_label("s paraphrase of section 3"), ("supported", "paraphrase of section 3"))
        self.assertEqual(parse_label(" U "), ("unsupported", ""))
        self.assertEqual(parse_label("x"), ("not_a_claim", ""))
        self.assertIsNone(parse_label("supported"))
        self.assertIsNone(parse_label(""))

    def test_pending_answers_hide_judge_verdicts_and_skip_labeled_claims(self):
        labels = {"sms_mfa": {"SMS is not approved": {"label": "supported", "note": ""}}}
        pending = pending_answers(RESULTS, ANSWERS, labels)
        self.assertEqual([item["id"] for item in pending], ["sms_mfa"])
        self.assertEqual(pending[0]["claims"], ["Keys are approved"])
        self.assertNotIn("verdict", str(pending))
        self.assertNotIn("evidence_quote", str(pending))

    def test_fully_labeled_answers_are_not_pending(self):
        labels = {"sms_mfa": {"SMS is not approved": {}, "Keys are approved": {}}}
        self.assertEqual(pending_answers(RESULTS, ANSWERS, labels), [])


if __name__ == "__main__":
    unittest.main()
