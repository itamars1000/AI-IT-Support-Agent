"""Build held-out faithfulness sets from questions the judge has never been tuned on.

The existing mutation set is the development set: its failures were inspected
and fixed. Held-out sets come from the challenge retrieval questions, split
before any answer exists:

    python -m evaluation.faithfulness_heldout split
    python -m evaluation.faithfulness_heldout generate
    python -m evaluation.faithfulness_heldout mutate    # a separate model writes one error per answer
    python -m evaluation.faithfulness_heldout review    # shows held-out B edits only; A stays unseen

Held-out A is scored once per decision; held-out B stays sealed for the next one.
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path
from typing import Literal

import anthropic
from pydantic import BaseModel, ConfigDict, ValidationError

from app.config import get_settings
from evaluation.faithfulness_judge import generate_answer, normalize
from evaluation.retrieval_benchmark import CHALLENGE_FILE, load_suite, prepare_corpus
from evaluation.runner import CASE_FILE, TOP_K, prepare_hits, safe_case


SPLIT_FILE = Path(__file__).with_name("faithfulness_split.json")
ANSWERS_FILE = Path(__file__).with_name("faithfulness_heldout_answers.json")
MUTATIONS_FILE = Path(__file__).with_name("faithfulness_heldout_mutations.json")
# A different model from the judge, so the two do not share blind spots.
MUTATOR_MODEL = "claude-sonnet-5-5"
MUTATION_LABELS = {
    "number": "contradicted", "negation": "contradicted",
    "invented_fact": "unsupported", "broader_scope": "unsupported",
}
MUTATOR_PROMPT = """You write test cases for a checker that decides whether an answer is faithful to its numbered sources.

Make exactly one small edit to the answer so that it contains one unfaithful claim of the requested type. Leave the rest of the answer unchanged, including its formatting and [n] citations.

- number: change one number, duration, count or time limit in the answer to a different value.
- negation: reverse one statement, for example allowed to not allowed, or required to not required.
- invented_fact: add one short sentence with a specific, plausible fact that no source states, cited like the surrounding text.
- broader_scope: widen one statement to a larger group, system or situation than the sources state.

Return:
- find: an exact substring of the answer to replace, copied character for character
- replace: the text that replaces it
- marker: a distinctive phrase of 2 to 5 words taken from replace that appears nowhere in the original answer or in any source
- expected_label: contradicted for number and negation, unsupported for invented_fact and broader_scope"""
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


def generate_heldout() -> dict:
    """Answer every held-out question once, over the policy and runbook corpus."""
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    home = {case_id: name for name in HELDOUT_SETS for case_id in split[name]}
    cases, documents = load_suite()
    selected = [case for case in cases if case["id"] in home]
    hits = prepare_hits(selected, prepare_corpus(documents))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "answer_model": get_settings().anthropic_model,
        "answers": [
            {**safe_case(generate_answer, case, hits[case["id"]][:TOP_K]),
             "set": home[case["id"]], "category": case["category"]}
            for case in selected
        ],
    }


class MutationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    find: str
    replace: str
    marker: str
    expected_label: Literal["contradicted", "unsupported"]


def assign_mutations(answers: list[dict]) -> dict[str, str]:
    """Rotate mutation types within each set, ordered by category, so types stay balanced."""
    assigned = {}
    for name in HELDOUT_SETS:
        items = sorted((item for item in answers if item["set"] == name), key=lambda item: (item["category"], item["id"]))
        for item, kind in zip(items, cycle(MUTATION_LABELS)):
            assigned[item["id"]] = kind
    return assigned


def rejection(spec: MutationSpec, item: dict, kind: str) -> str | None:
    """Why an edit cannot serve as a test case with a known label, or None when it can."""
    answer = item["answer"]
    if answer.count(spec.find) != 1:
        return "find must occur exactly once in the answer"
    if spec.replace == spec.find:
        return "replace must differ from find"
    marker = normalize(spec.marker)
    if not marker or marker not in normalize(spec.replace):
        return "marker must come from replace"
    if marker in normalize(answer) or any(marker in normalize(source["text"]) for source in item["sources"]):
        return "marker must not appear in the original answer or any source"
    if spec.expected_label != MUTATION_LABELS[kind]:
        return f"expected_label for {kind} is {MUTATION_LABELS[kind]}"
    if kind == "number" and not any(character.isdigit() for character in spec.marker):
        return "a number edit's marker must contain the new number"
    return None


def mutate_answer(client: anthropic.Anthropic, item: dict, kind: str, attempts: int = 2) -> dict:
    payload = {"mutation_type": kind, "sources": [
        {"number": source["number"], "text": source["text"]} for source in item["sources"]
    ], "answer": item["answer"]}
    problem = None
    for _ in range(attempts):
        content = json.dumps(payload, ensure_ascii=False)
        if problem:
            content += f"\n\nYour previous edit was rejected: {problem}. Write a new edit."
        try:
            response = client.beta.messages.parse(
                model=MUTATOR_MODEL,
                max_tokens=16000,
                system=MUTATOR_PROMPT,
                messages=[{"role": "user", "content": content}],
                output_format=MutationSpec,
                output_config={"effort": "low"},
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except (anthropic.APIError, ValidationError) as error:
            return {"id": item["id"], "error_type": type(error).__name__}
        spec = response.parsed_output
        if response.stop_reason != "end_turn" or spec is None:
            return {"id": item["id"], "error_type": "MutatorIncomplete", "stop_reason": response.stop_reason}
        problem = rejection(spec, item, kind)
        if problem is None:
            return {
                "id": f"{item['id']}__{kind}", "base_case": item["id"], "set": item["set"],
                "category": item["category"], "mutation": kind, "expected": "unfaithful",
                "change": {"find": spec.find, "replace": spec.replace}, "marker": spec.marker,
                "expected_claim_label": spec.expected_label, "question": item["question"],
                "sources": item["sources"], "answer": item["answer"].replace(spec.find, spec.replace),
            }
    return {"id": item["id"], "error_type": "MutationRejected", "reason": problem}


def build_mutations(answers: dict, client: anthropic.Anthropic) -> dict:
    items = [item for item in answers["answers"] if "error_type" not in item]
    kinds = assign_mutations(items)
    controls = [
        {"id": f"{item['id']}__control", "base_case": item["id"], "set": item["set"], "category": item["category"],
         "mutation": "none", "expected": "faithful", "question": item["question"],
         "sources": item["sources"], "answer": item["answer"]}
        for item in items
    ]
    results = [mutate_answer(client, item, kinds[item["id"]]) for item in items]
    return {
        "description": "Held-out answers with one injected unfaithful claim each, written by a separate model, plus unchanged controls.",
        "mutator_model": MUTATOR_MODEL,
        "answer_model": answers["answer_model"],
        "cases": controls + [result for result in results if "error_type" not in result],
        "failures": [result for result in results if "error_type" in result],
    }


def review(mutations: dict) -> None:
    """Print held-out B edits for a plausibility check; report only counts for held-out A."""
    for name in HELDOUT_SETS:
        cases = [case for case in mutations["cases"] if case["set"] == name and case["expected"] == "unfaithful"]
        counts = {kind: sum(case["mutation"] == kind for case in cases) for kind in MUTATION_LABELS}
        print(f"{name}: {len(cases)} mutations {counts}")
    print(f"failures: {[item['id'] for item in mutations['failures']]}\n")
    for case in mutations["cases"]:
        if case["set"] == "heldout_b" and case["expected"] == "unfaithful":
            print(f"[{case['id']}] {case['mutation']} -> {case['expected_claim_label']}")
            print(f"  find:    {case['change']['find']}")
            print(f"  replace: {case['change']['replace']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out faithfulness sets")
    parser.add_argument("step", choices=("split", "generate", "mutate", "review"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.step == "mutate":
        if MUTATIONS_FILE.exists():
            sys.exit(f"{MUTATIONS_FILE.name} already exists; held-out mutations are written once and frozen.")
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        if not api_key:
            parser.error("Set ANTHROPIC_API_KEY in .env before writing mutations")
        answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
        mutations = build_mutations(answers, anthropic.Anthropic(api_key=api_key))
        MUTATIONS_FILE.write_text(json.dumps(mutations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        review(mutations)
        return
    if args.step == "review":
        review(json.loads(MUTATIONS_FILE.read_text(encoding="utf-8")))
        return
    if args.step == "generate":
        if ANSWERS_FILE.exists():
            sys.exit(f"{ANSWERS_FILE.name} already exists; held-out answers are generated once and frozen.")
        report = generate_heldout()
        ANSWERS_FILE.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        failed = [item["id"] for item in report["answers"] if "error_type" in item]
        print(json.dumps({"output": str(ANSWERS_FILE), "answers": len(report["answers"]), "errors": failed}))
        return
    if SPLIT_FILE.exists():
        sys.exit(f"{SPLIT_FILE.name} already exists; a split is made once, before any answers are seen.")
    split = build_split()
    SPLIT_FILE.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: len(split[name]) for name in ("dev", *HELDOUT_SETS)} | {"categories": split["categories"]}, indent=2))


if __name__ == "__main__":
    main()
