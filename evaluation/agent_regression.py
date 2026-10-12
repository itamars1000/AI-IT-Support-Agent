"""Regression suite for the agent's decisions with real Claude and fake tools.

    python -m evaluation.agent_regression run --model claude-haiku-4-5-20251001
    python -m evaluation.agent_regression report --label claude-haiku-4-5-20251001
    python -m evaluation.agent_regression compare --baseline A --candidate B

Search returns fixed chunks from the fixtures and tickets live in memory, so no
database is touched and retrieval quality cannot change a result. Each scenario
runs several times because the model's choices vary between runs. Development
scenarios print their failures; held-out scenarios print totals only and are
run once per label.
"""

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from app.config import get_settings
from app.errors import AnswerProviderUnavailable, TicketNotFound
from app.schemas import AgentRequest, EscalationResult, SearchHit, TicketPriority, TicketResponse
from app.services.agent_service import run_agent
from app.services.anthropic_client import AnthropicRequestError
from app.services.chunking_service import chunk_document
from evaluation.retrieval_benchmark import load_suite


SCENARIO_FILE = Path(__file__).with_name("agent_scenarios.json")
FIXTURE_DIR = Path(__file__).with_name("fixtures")
SETS = ("dev", "heldout")
HEBREW = re.compile(r"[֐-׿]")
LATIN = re.compile(r"[A-Za-z]")


def results_file(case_set: str, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", label)
    return Path(__file__).with_name(f"agent_regression_{case_set}_{safe}.json")


def resolve_search(scenarios: list[dict]) -> dict[str, list[SearchHit]]:
    """Turn each scenario's source/anchor references into the application's chunks."""
    _, documents = load_suite()
    chunks = {document.source_file: chunk_document(document) for document in documents}
    resolved = {}
    for scenario in scenarios:
        hits = []
        for rank, ref in enumerate(scenario["search"], 1):
            if "fixture" in ref:
                source_file, text = ref["fixture"], (FIXTURE_DIR / ref["fixture"]).read_text(encoding="utf-8")
                page, index = 1, 0
            else:
                chunk = next((chunk for chunk in chunks.get(ref["source_file"], []) if ref["anchor"] in chunk.text), None)
                if chunk is None:
                    raise ValueError(f"{scenario['id']}: anchor not found in {ref['source_file']}")
                source_file, text, page, index = ref["source_file"], chunk.text, chunk.page_number, chunk.chunk_index
            hits.append(SearchHit(chunk_id=900300 + rank, document_id=9003, source_file=source_file,
                                  page_number=page, chunk_index=index, text=text, distance=0.1 * rank))
        resolved[scenario["id"]] = hits
    return resolved


def fake_tickets(ids: dict) -> tuple[dict, object, object, object]:
    now = datetime.now(timezone.utc)
    tickets = {
        ids["old_open_ticket"]: TicketResponse(id=ids["old_open_ticket"], title="VPN keeps disconnecting",
                                               description="Synthetic ticket", priority="medium", status="open",
                                               created_at=now - timedelta(days=8)),
        ids["recent_open_ticket"]: TicketResponse(id=ids["recent_open_ticket"], title="Printer jammed",
                                                  description="Synthetic ticket", priority="medium", status="open",
                                                  created_at=now - timedelta(days=1)),
    }

    def get(ticket_id: int) -> TicketResponse:
        if ticket_id not in tickets:
            raise TicketNotFound()
        return tickets[ticket_id]

    def create(payload, idempotency_key=None) -> TicketResponse:
        ticket = TicketResponse(id=ids["created_ticket"], title=payload.title, description=payload.description,
                                priority=payload.priority, status="open", created_at=now)
        tickets[ticket.id] = ticket
        return ticket

    def escalate(ticket_id: int) -> EscalationResult:
        ticket = get(ticket_id)
        if ticket.created_at < now - timedelta(days=7) and ticket.escalated_at is None:
            tickets[ticket_id] = ticket.model_copy(update={"priority": TicketPriority.high, "escalated_at": now})
            return EscalationResult(ticket=tickets[ticket_id], escalated=True, reason="escalated")
        return EscalationResult(ticket=ticket, escalated=False, reason="too_recent")

    return tickets, get, create, escalate


def run_scenario(scenario: dict, hits: list[SearchHit], ids: dict, settings) -> dict:
    """One real agent run with fake tools; returns its steps and answer, or the error type."""
    allowed = scenario.get("allowed_actions", [])
    request = AgentRequest(
        message=scenario["message"], history=scenario.get("history", []), allowed_actions=allowed,
        idempotency_key=uuid4() if "create_ticket" in allowed else None,
        approved_ticket=scenario.get("approved_ticket"),
    )
    _, get, create, escalate = fake_tickets(ids)
    with patch("app.services.agent_service.get_settings", return_value=settings), \
            patch("app.services.agent_service.search_documents", return_value=hits), \
            patch("app.services.agent_service.ticket_service.get_ticket", side_effect=get), \
            patch("app.services.agent_service.ticket_service.create_ticket", side_effect=create), \
            patch("app.services.agent_service.ticket_service.escalate_ticket", side_effect=escalate):
        try:
            response = run_agent(request)
        except AnthropicRequestError as error:
            return {"error_type": type(error).__name__, "error_detail": str(error)}
    return {
        "completed": response.completed,
        "steps": [step.model_dump(mode="json") for step in response.steps],
        "answer": response.answer,
    }


def is_hebrew(text: str) -> bool:
    hebrew, latin = len(HEBREW.findall(text)), len(LATIN.findall(text))
    return hebrew > 0 and hebrew >= 0.5 * latin


def failed_checks(checks: dict, run: dict) -> list[str]:
    """Every check the run fails, as short readable reasons; an empty list is a pass."""
    if "error_type" in run:
        return [f"error: {run['error_detail']}"]
    failures = []
    if not run["completed"]:
        failures.append("workflow did not complete")
    names = [step["name"] for step in run["steps"]]
    answer = run["answer"].casefold()
    failures += [f"did not call {name}" for name in checks.get("must_call", []) if name not in names]
    failures += [f"called {name}" for name in checks.get("forbid_call", []) if name in names]
    failures += [
        f"called {name} {names.count(name)} times, expected {count}"
        for name, count in checks.get("call_count", {}).items() if names.count(name) != count
    ]
    for rule in checks.get("args", []):
        inputs = [step["input"] for step in run["steps"] if step["name"] == rule["tool"]]
        if not inputs:
            failures.append(f"no {rule['tool']} call to check {rule['field']}")
            continue
        value = str(inputs[-1].get(rule["field"], "")).casefold()
        if "equals" in rule and value != rule["equals"]:
            failures.append(f"{rule['tool']}.{rule['field']} is {value!r}, expected {rule['equals']!r}")
        if "contains_any" in rule and not any(term.casefold() in value for term in rule["contains_any"]):
            failures.append(f"{rule['tool']}.{rule['field']} lacks any of {rule['contains_any']}")
        if "not_contains_any" in rule and any(term.casefold() in value for term in rule["not_contains_any"]):
            failures.append(f"{rule['tool']}.{rule['field']} contains one of {rule['not_contains_any']}")
    if "answer_citation" in checks and bool(re.search(r"\[\d+\]", run["answer"])) != checks["answer_citation"]:
        failures.append("answer has no [n] citation" if checks["answer_citation"] else "answer cites a source")
    missing = [term for term in checks.get("answer_contains_all", []) if term.casefold() not in answer]
    if missing:
        failures.append(f"answer lacks {missing}")
    if checks.get("answer_contains_any") and not any(term.casefold() in answer for term in checks["answer_contains_any"]):
        failures.append("answer lacks every expected phrase")
    present = [term for term in checks.get("answer_not_contains_any", []) if term.casefold() in answer]
    if present:
        failures.append(f"answer contains {present}")
    if checks.get("answer_language") == "he" and not is_hebrew(run["answer"]):
        failures.append("answer is not in Hebrew")
    return failures


def summarize(scenarios: list[dict], runs: dict[str, list[dict]]) -> dict:
    per_scenario = {}
    for scenario in scenarios:
        outcomes = [failed_checks(scenario["checks"], run) for run in runs[scenario["id"]]]
        per_scenario[scenario["id"]] = {
            "group": scenario["group"],
            "passed": sum(not failures for failures in outcomes),
            "runs": len(outcomes),
            "failures": [failures for failures in outcomes if failures],
        }
    stable = sum(item["passed"] == item["runs"] for item in per_scenario.values())
    flaky = sum(0 < item["passed"] < item["runs"] for item in per_scenario.values())
    return {
        "scenarios": len(per_scenario),
        "all_runs_passed": stable,
        "flaky": flaky,
        "never_passed": len(per_scenario) - stable - flaky,
        "run_pass_rate": round(sum(item["passed"] for item in per_scenario.values())
                               / max(1, sum(item["runs"] for item in per_scenario.values())), 3),
        "per_scenario": per_scenario,
    }


def print_summary(case_set: str, label: str, summary: dict) -> None:
    print(f"{case_set} | {label}: {summary['all_runs_passed']}/{summary['scenarios']} scenarios passed every run, "
          f"{summary['flaky']} flaky, {summary['never_passed']} never passed; run pass rate {summary['run_pass_rate']}")
    if case_set != "dev":
        print("  Held-out set: scenario-level results are not shown.")
        return
    for scenario_id, item in summary["per_scenario"].items():
        if item["failures"]:
            print(f"  {scenario_id} {item['passed']}/{item['runs']}: {'; '.join(item['failures'][0])}")


def compare(case_set: str, baseline: dict, candidate: dict, labels: tuple[str, str]) -> None:
    rows = []
    for scenario_id, before in baseline["per_scenario"].items():
        after = candidate["per_scenario"].get(scenario_id)
        if after and after["passed"] / after["runs"] != before["passed"] / before["runs"]:
            rows.append((scenario_id, f"{before['passed']}/{before['runs']}", f"{after['passed']}/{after['runs']}",
                         after["passed"] / after["runs"] > before["passed"] / before["runs"]))
    improved, regressed = sum(row[3] for row in rows), sum(not row[3] for row in rows)
    print(f"{case_set}: {labels[0]} {baseline['run_pass_rate']} -> {labels[1]} {candidate['run_pass_rate']} "
          f"run pass rate; {improved} scenarios improved, {regressed} regressed")
    if case_set == "dev":
        for scenario_id, before, after, better in rows:
            print(f"  {'better' if better else 'WORSE '} {scenario_id}: {before} -> {after}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent regression suite")
    parser.add_argument("step", choices=("run", "report", "compare"))
    parser.add_argument("--model", help="Claude model for run (default: ANTHROPIC_MODEL)")
    parser.add_argument("--label", help="Results label (default: the model)")
    parser.add_argument("--set", choices=(*SETS, "all"), default="all", dest="case_set")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--baseline")
    parser.add_argument("--candidate")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    data = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
    sets = SETS if args.case_set == "all" else (args.case_set,)

    if args.step == "compare":
        if not args.baseline or not args.candidate:
            parser.error("compare needs --baseline and --candidate labels")
        for case_set in sets:
            files = [results_file(case_set, label) for label in (args.baseline, args.candidate)]
            if all(path.exists() for path in files):
                baseline, candidate = (json.loads(path.read_text(encoding="utf-8"))["summary"] for path in files)
                compare(case_set, baseline, candidate, (args.baseline, args.candidate))
        return

    settings = get_settings()
    model = args.model or settings.anthropic_model
    label = args.label or model
    if args.step == "report":
        for case_set in sets:
            path = results_file(case_set, label)
            if path.exists():
                print_summary(case_set, label, json.loads(path.read_text(encoding="utf-8"))["summary"])
        return

    if not settings.anthropic_api_key or not settings.anthropic_api_key.get_secret_value():
        raise AnswerProviderUnavailable()
    run_settings = settings.model_copy(update={"anthropic_model": model})
    hits = resolve_search(data["scenarios"])
    for case_set in sets:
        path = results_file(case_set, label)
        if case_set == "heldout" and path.exists():
            print(f"heldout | {label}: already run; held-out scenarios are run once per label.")
            continue
        scenarios = [scenario for scenario in data["scenarios"] if scenario["set"] == case_set]
        runs = {
            scenario["id"]: [run_scenario(scenario, hits[scenario["id"]], data["tickets"], run_settings)
                             for _ in range(args.repeats)]
            for scenario in scenarios
        }
        summary = summarize(scenarios, runs)
        path.write_text(json.dumps({
            "generated_at": datetime.now(timezone.utc).isoformat(), "model": model, "label": label,
            "repeats": args.repeats, "summary": summary, "runs": runs,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print_summary(case_set, label, summary)


if __name__ == "__main__":
    main()
