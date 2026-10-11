"""Label judged claims by hand, without seeing the judge's verdicts.

Run after faithfulness_judge.py judge:

    python -m evaluation.label_faithfulness

Each claim gets s (supported), u (unsupported) or c (contradicted), or x when
the judge's text is not a real claim. Text after the letter is saved as a note,
for example "s paraphrase of section 3". Labels are saved after every claim,
so q quits and a later run resumes where it stopped.
"""

import json
import sys
from pathlib import Path

from evaluation.faithfulness_judge import ANSWERS_FILE, RESULTS_FILE


LABELS_FILE = Path(__file__).with_name("faithfulness_labels.json")
LABELS = {"s": "supported", "u": "unsupported", "c": "contradicted", "x": "not_a_claim"}
GUIDELINES = """Labeling rules
  supported     a reasonable reader would conclude the claim directly from a source;
                paraphrase and direct implication count
  unsupported   the claim adds something no source gives: a fact, a number,
                a broader or narrower scope, or more certainty
  contradicted  a source states the opposite or a different value
  An omission is not a faithfulness problem. Judge only what the claim states.
"""


def parse_label(text: str) -> tuple[str, str] | None:
    """Return (label, note), or None when the input is not a label letter."""
    letter, _, note = text.strip().partition(" ")
    label = LABELS.get(letter.casefold())
    return (label, note.strip()) if label else None


def pending_answers(results: dict, answers: dict, labels: dict) -> list[dict]:
    """Answers with unlabeled claims, carrying only what a blind labeler may see."""
    saved_answers = {item["id"]: item for item in answers["answers"] if "error_type" not in item}
    pending = []
    for result in results["results"]:
        if "error_type" in result or result["id"] not in saved_answers:
            continue
        done = labels.get(result["id"], {})
        claims = [item["claim"] for item in result["claims"] if item["claim"] not in done]
        if claims:
            pending.append({**saved_answers[result["id"]], "claims": claims})
    return pending


def show_answer(item: dict) -> None:
    print("=" * 78)
    print(f"[{item['id']}] {item['question']}\n")
    for source in item["sources"]:
        print(f"--- Source [{source['number']}] {source['source_file']}, page {source['page_number']}")
        print(source["text"].strip() + "\n")
    print("--- Answer")
    print(item["answer"].strip() + "\n")


def ask_label(claim: str, position: str) -> tuple[str, str] | None:
    """Return a label, or None when the labeler quits."""
    while True:
        text = input(f"Claim {position}: {claim}\n  [s/u/c/x, optional note | q quits] > ")
        if text.strip().casefold() == "q":
            return None
        parsed = parse_label(text)
        if parsed:
            return parsed
        print("  Enter s, u, c or x.")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if not RESULTS_FILE.exists() or not ANSWERS_FILE.exists():
        sys.exit("Run faithfulness_judge.py generate and judge first.")
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    answers = json.loads(ANSWERS_FILE.read_text(encoding="utf-8"))
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8")) if LABELS_FILE.exists() else {}

    pending = pending_answers(results, answers, labels)
    remaining = sum(len(item["claims"]) for item in pending)
    print(GUIDELINES)
    print(f"{remaining} claims to label in {len(pending)} answers.\n")
    for item in pending:
        show_answer(item)
        for index, claim in enumerate(item["claims"], 1):
            parsed = ask_label(claim, f"{index}/{len(item['claims'])}")
            if parsed is None:
                print(f"Saved. Labels are in {LABELS_FILE}.")
                return
            label, note = parsed
            labels.setdefault(item["id"], {})[claim] = {"label": label, "note": note}
            LABELS_FILE.write_text(json.dumps(labels, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"All claims are labeled. Labels are in {LABELS_FILE}.")


if __name__ == "__main__":
    main()
