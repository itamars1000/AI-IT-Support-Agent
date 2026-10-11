import json
import unittest
from collections import Counter
from types import SimpleNamespace

from evaluation.faithfulness_heldout import (
    HELDOUT_SETS, MUTATION_LABELS, MUTATOR_MODEL, TWINS, MutationSpec, assign_mutations,
    build_split, mutate_answer, rejection, split_questions,
)
from evaluation.faithfulness_judge import JUDGE_MODEL
from evaluation.retrieval_benchmark import CHALLENGE_FILE


CHALLENGE = json.loads(CHALLENGE_FILE.read_text(encoding="utf-8"))["qa"]


class HeldoutSplitTests(unittest.TestCase):
    def test_split_is_reproducible_and_covers_every_question_once(self):
        first, second = split_questions(CHALLENGE), split_questions(CHALLENGE)
        self.assertEqual(first, second)
        assigned = [case_id for name in HELDOUT_SETS for case_id in first[name]]
        self.assertEqual(sorted(assigned), sorted(case["id"] for case in CHALLENGE))

    def test_twin_questions_share_a_split(self):
        splits = split_questions(CHALLENGE)
        for pair in TWINS:
            homes = {name for name in HELDOUT_SETS for case_id in pair if case_id in splits[name]}
            self.assertEqual(len(homes), 1, pair)

    def test_sets_are_balanced_by_size_and_category(self):
        split = build_split()
        self.assertLessEqual(abs(len(split["heldout_a"]) - len(split["heldout_b"])), 2)
        a, b = (split["categories"][name] for name in HELDOUT_SETS)
        for category in a:
            self.assertLessEqual(abs(a[category] - b[category]), 2, category)

    def test_dev_questions_never_enter_a_heldout_set(self):
        split = build_split()
        self.assertFalse(set(split["dev"]) & set(split["heldout_a"] + split["heldout_b"]))
        self.assertEqual(len(split["dev"]), 11)


ITEM = {
    "id": "vpn_total_limit", "set": "heldout_a", "category": "near_match", "question": "Max VPN time?",
    "answer": "VPN sessions end after **12 hours** of total connection time [1].",
    "sources": [{"number": 1, "source_file": "vpn.pdf", "page_number": 1,
                 "text": "Sessions disconnect after 30 minutes idle or 12 hours of total connection time."}],
}


def spec(find, replace, marker, label="contradicted"):
    return MutationSpec(find=find, replace=replace, marker=marker, expected_label=label)


class FakeBetaMessages:
    def __init__(self, specs):
        self.specs = list(specs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(stop_reason="end_turn", parsed_output=self.specs.pop(0))


class MutationTests(unittest.TestCase):
    def test_types_rotate_within_each_set(self):
        answers = [{"id": f"q{index}", "set": "heldout_a", "category": "identifier"} for index in range(8)]
        counts = Counter(assign_mutations(answers).values())
        self.assertEqual(counts, {kind: 2 for kind in MUTATION_LABELS})

    def test_rejection_explains_each_invalid_edit(self):
        self.assertIsNone(rejection(spec("**12 hours**", "**8 hours**", "8 hours"), ITEM, "number"))
        self.assertIn("exactly once", rejection(spec("24 hours", "8 hours", "8 hours"), ITEM, "number"))
        self.assertIn("any source", rejection(spec("**12 hours**", "**30 minutes**", "30 minutes"), ITEM, "number"))
        self.assertIn("from replace", rejection(spec("**12 hours**", "**8 hours**", "9 hours"), ITEM, "number"))
        self.assertIn("expected_label", rejection(spec("**12 hours**", "**8 hours**", "8 hours", "unsupported"), ITEM, "number"))
        self.assertIn("new number", rejection(spec("**12 hours**", "**half a day**", "half a day"), ITEM, "number"))

    def test_rejected_edit_is_retried_with_the_reason(self):
        messages = FakeBetaMessages([
            spec("**12 hours**", "**30 minutes**", "30 minutes"),
            spec("**12 hours**", "**8 hours**", "8 hours"),
        ])
        case = mutate_answer(SimpleNamespace(beta=SimpleNamespace(messages=messages)), ITEM, "number")
        self.assertEqual(case["answer"], "VPN sessions end after **8 hours** of total connection time [1].")
        self.assertEqual((case["expected"], case["set"]), ("unfaithful", "heldout_a"))
        self.assertIn("must not appear", messages.calls[1]["messages"][0]["content"])
        self.assertEqual(messages.calls[0]["model"], MUTATOR_MODEL)
        self.assertNotEqual(MUTATOR_MODEL, JUDGE_MODEL)

    def test_edit_still_invalid_after_retries_is_recorded_as_failure(self):
        bad = spec("missing text", "x", "x")
        messages = FakeBetaMessages([bad, bad])
        result = mutate_answer(SimpleNamespace(beta=SimpleNamespace(messages=messages)), ITEM, "number")
        self.assertEqual(result["error_type"], "MutationRejected")


if __name__ == "__main__":
    unittest.main()
