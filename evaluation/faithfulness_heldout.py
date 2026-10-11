"""Build held-out faithfulness sets from questions the judge has never been tuned on.

The existing mutation set is the development set: its failures were inspected
and fixed. Held-out sets come from the challenge retrieval questions, split
before any answer exists:

    python -m evaluation.faithfulness_heldout split

Held-out A is scored once per decision; held-out B stays sealed for the next one.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from evaluation.retrieval_benchmark import CHALLENGE_FILE
from evaluation.runner import CASE_FILE


SPLIT_FILE = Path(__file__).with_name("faithfulness_split.json")
SEED = 20261011
HELDOUT_SETS = ("heldout_a", "heldout_b")
# Near-identical questions share a split: a fix learned on one would leak into the other.
TWINS = (
    ("gp1001", "gp1002"), ("gp401", "gp403"), ("otp410", "otp411"),
    ("kb431", "kb432"), ("p1_ack", "p3_ack"),
)


def question_groups(cases: list[dict]) -> list[list[dict]]:
    by_id = {case["id"]: case for case in cases}
    grouped = {case_id for pair in TWINS for case_id in pair}
    groups = [[by_id[case_id] for case_id in pair] for pair in TWINS]
    groups += [[case] for case in cases if case["id"] not in grouped]
    for group in groups:
        if len({case["category"] for case in group}) != 1:
            raise ValueError(f"Twin questions must share a category: {[case['id'] for case in group]}")
    return groups


def split_questions(cases: list[dict], seed: int = SEED) -> dict[str, list[str]]:
    """Assign whole groups, category by category, to the held-out set with fewer questions."""
    rng = random.Random(seed)
    by_category = defaultdict(list)
    for group in question_groups(cases):
        by_category[group[0]["category"]].append(group)
    splits = {name: [] for name in HELDOUT_SETS}
    for category in sorted(by_category):
        groups = by_category[category]
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)  # place pairs first, then balance with singles
        in_category = {name: 0 for name in HELDOUT_SETS}
        for group in groups:
            target = min(HELDOUT_SETS, key=lambda name: (in_category[name], len(splits[name]), rng.random()))
            in_category[target] += len(group)
            splits[target] += [case["id"] for case in group]
    return splits


def build_split() -> dict:
    dev = [case["id"] for case in json.loads(CASE_FILE.read_text(encoding="utf-8"))["qa"]]
    challenge = json.loads(CHALLENGE_FILE.read_text(encoding="utf-8"))["qa"]
    if set(dev) & {case["id"] for case in challenge}:
        raise ValueError("Development and challenge questions must not overlap")
    categories = {case["id"]: case["category"] for case in challenge}
    splits = split_questions(challenge)
    return {
        "description": "Faithfulness split. dev was inspected and tuned on; held-out sets are scored, never debugged.",
        "seed": SEED,
        "dev": dev,
        **{name: sorted(ids) for name, ids in splits.items()},
        "categories": {
            name: {category: sum(categories[case_id] == category for case_id in ids)
                   for category in sorted(set(categories.values()))}
            for name, ids in splits.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out faithfulness sets")
    parser.add_argument("step", choices=("split",))
    parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE.name} already exists; a split is made once, before any answers are seen.")
    split = build_split()
    SPLIT_FILE.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(split[name]) for name in ("dev", *HELDOUT_SETS)} | {"categories": split["categories"]}, indent=2))


if __name__ == "__main__":
    main()
