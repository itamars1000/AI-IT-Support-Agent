import json
import unittest

from evaluation.faithfulness_heldout import HELDOUT_SETS, TWINS, build_split, split_questions
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


if __name__ == "__main__":
    unittest.main()
