"""Measure the faithfulness judge against labels known in advance.

    python -m evaluation.faithfulness_calibration run      # judge the mutation set, then report
    python -m evaluation.faithfulness_calibration report   # rescore saved verdicts with the current checks
    python -m evaluation.faithfulness_calibration human    # compare the judge with hand labels

Mutation cases each contain one injected unfaithful claim; controls are the same
answers unchanged. The headline is recall on the unfaithful answers: a judge that
calls everything supported would still agree with most controls.
"""

import argparse
import json
import sys
from collections import Counter
from math import sqrt
from pathlib import Path

import anthropic

from app.config import get_settings
from evaluation.faithfulness_judge import (
    JUDGE_MODEL, RESULTS_FILE, ClaimVerdict, JudgeResult, judge_answer, normalize, score_answer,
)
from evaluation.label_faithfulness import LABELS_FILE


MUTATIONS_FILE = Path(__file__).with_name("faithfulness_mutations.json")
CALIBRATION_FILE = Path(__file__).with_name("faithfulness_calibration_results.json")
VERDICTS = ("supported", "unsupported", "contradicted")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float] | None:
    """95% interval for a rate measured on a small sample."""
    if not total:
        return None
    rate = successes / total
    centre = (rate + z * z / (2 * total)) / (1 + z * z / total)
    margin = z * sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return [round(max(0.0, centre - margin), 3), round(min(1.0, centre + margin), 3)]


def ratio(successes: int, total: int) -> dict:
    return {
        "count": f"{successes}/{total}",
        "rate": round(successes / total, 3) if total else None,
        "interval_95": wilson_interval(successes, total),
    }


def target_claim(case: dict, result: dict) -> dict | None:
    """The judged claim that carries the injected change, found by its marker text."""
    return next((item for item in result["claims"] if normalize(case["marker"]) in normalize(item["claim"])), None)


def rescore(case: dict, result: dict) -> dict:
    """Reapply the current deterministic checks to saved judge verdicts, without an API call."""
    if "error_type" in result:
        return result
    verdicts = JudgeResult(claims=[
        ClaimVerdict(**{field: item[field] for field in ClaimVerdict.model_fields}) for item in result["claims"]
    ])
    return {**result, **score_answer(verdicts, {source["number"]: source["text"] for source in case["sources"]})}


def calibration_report(cases: list[dict], results: list[dict]) -> dict:
    by_id = {item["id"]: item for item in results}
    judged = [(case, by_id[case["id"]]) for case in cases if "error_type" not in by_id.get(case["id"], {"error_type": "missing"})]
    flagged = lambda result: not result["faithful"]

    matrix = Counter((case["expected"], "unfaithful" if flagged(result) else "faithful") for case, result in judged)
    caught = matrix[("unfaithful", "unfaithful")]
    unfaithful = caught + matrix[("unfaithful", "faithful")]
    false_alarms = matrix[("faithful", "unfaithful")]
    controls = false_alarms + matrix[("faithful", "faithful")]

    by_type: dict[str, dict] = {}
    claim_level = []
    for case, result in judged:
        if case["expected"] != "unfaithful":
            continue
        group = by_type.setdefault(case["mutation"], {"caught": 0, "total": 0})
        group["total"] += 1
        group["caught"] += flagged(result)
        claim = target_claim(case, result)
        claim_level.append({
            "id": case["id"],
            "answer_flagged": flagged(result),
            "target_claim": claim["claim"] if claim else None,
            "expected_label": case["expected_claim_label"],
            "judge_verdict": claim["verdict"] if claim else None,
            "judge_error": claim["judge_error"] if claim else None,
        })

    return {
        "judged": len(judged),
        "errors": len(cases) - len(judged),
        "confusion_matrix": {f"expected_{e}__judge_{p}": matrix[(e, p)] for e in ("faithful", "unfaithful") for p in ("faithful", "unfaithful")},
        "recall_unfaithful": ratio(caught, unfaithful),
        "precision_unfaithful": ratio(caught, caught + false_alarms),
        "false_alarm_rate": ratio(false_alarms, controls),
        "recall_by_mutation": {kind: ratio(group["caught"], group["total"]) for kind, group in sorted(by_type.items())},
        "target_claim_found": sum(item["target_claim"] is not None for item in claim_level),
        "target_verdict_matches": sum(item["judge_verdict"] == item["expected_label"] for item in claim_level),
        "mutations": claim_level,
        "false_alarm_ids": [case["id"] for case, result in judged if case["expected"] == "faithful" and flagged(result)],
    }


def human_agreement(results: dict, labels: dict) -> dict:
    """Claim-level confusion between hand labels (rows) and judge verdicts (columns)."""
    matrix = Counter()
    disagreements = []
    for result in results["results"]:
        for item in result.get("claims", []):
            label = labels.get(result["id"], {}).get(item["claim"])
            if label is None or label["label"] == "not_a_claim":
                continue
            matrix[(label["label"], item["verdict"])] += 1
            if label["label"] != item["verdict"]:
                disagreements.append({"id": result["id"], "claim": item["claim"], "human": label["label"],
                                      "judge": item["verdict"], "judge_quote": item["evidence_quote"], "note": label["note"]})
    total = sum(matrix.values())
    agree = sum(matrix[(verdict, verdict)] for verdict in VERDICTS)
    human_flags = sum(count for (human, _), count in matrix.items() if human != "supported")
    caught = sum(count for (human, judge), count in matrix.items() if human != "supported" and judge != "supported")
    return {
        "labeled_claims": total,
        "not_a_claim": sum(item["label"] == "not_a_claim" for claims in labels.values() for item in claims.values()),
        "agreement": ratio(agree, total),
        "recall_on_human_flags": ratio(caught, human_flags),
        "matrix_human_by_judge": {human: {judge: matrix[(human, judge)] for judge in VERDICTS} for human in VERDICTS},
        "disagreements": disagreements,
    }


def print_calibration(report: dict) -> None:
    m = report["confusion_matrix"]
    print(f"Judged {report['judged']} cases ({report['errors']} errors), judge {JUDGE_MODEL}\n")
    print("                     judge: faithful   judge: unfaithful")
    print(f"expected faithful    {m['expected_faithful__judge_faithful']:>15}   {m['expected_faithful__judge_unfaithful']:>17}")
    print(f"expected unfaithful  {m['expected_unfaithful__judge_faithful']:>15}   {m['expected_unfaithful__judge_unfaithful']:>17}\n")
    for name in ("recall_unfaithful", "precision_unfaithful", "false_alarm_rate"):
        value = report[name]
        print(f"{name:<22} {value['count']:>6}  95% interval {value['interval_95']}")
    print("\nRecall by mutation type:")
    for kind, value in report["recall_by_mutation"].items():
        print(f"  {kind:<16} {value['count']}")
    print(f"\nInjected claim found in the judge's claims: {report['target_claim_found']}/{len(report['mutations'])}; "
          f"verdict matched the expected label: {report['target_verdict_matches']}")
    for item in report["mutations"]:
        if not item["answer_flagged"] or item["judge_verdict"] != item["expected_label"]:
            print(f"  {item['id']}: expected {item['expected_label']}, judge {item['judge_verdict']}, "
                  f"answer flagged {item['answer_flagged']}")
    if report["false_alarm_ids"]:
        print(f"False alarms on controls: {', '.join(report['false_alarm_ids'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the faithfulness judge")
    parser.add_argument("step", choices=("run", "report", "human"))
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.step == "human":
        if not RESULTS_FILE.exists() or not LABELS_FILE.exists():
            parser.error("Judge answers and label them first")
        report = human_agreement(json.loads(RESULTS_FILE.read_text(encoding="utf-8")),
                                 json.loads(LABELS_FILE.read_text(encoding="utf-8")))
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    cases = json.loads(MUTATIONS_FILE.read_text(encoding="utf-8"))["cases"]
    if args.step == "run":
        settings = get_settings()
        api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        if not api_key:
            parser.error("Set ANTHROPIC_API_KEY in .env before judging")
        client = anthropic.Anthropic(api_key=api_key)
        results = [judge_answer(client, case) for case in cases]
        CALIBRATION_FILE.write_text(json.dumps({"judge_model": JUDGE_MODEL, "results": results},
                                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif not CALIBRATION_FILE.exists():
        parser.error("Run the calibration first")
    saved = {item["id"]: item for item in json.loads(CALIBRATION_FILE.read_text(encoding="utf-8"))["results"]}
    results = [rescore(case, saved[case["id"]]) for case in cases if case["id"] in saved]
    print_calibration(calibration_report(cases, results))


if __name__ == "__main__":
    main()
