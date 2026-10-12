import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from evaluation.agent_regression import (
    SCENARIO_FILE, failed_checks, is_hebrew, resolve_search, run_scenario, summarize,
)


DATA = json.loads(SCENARIO_FILE.read_text(encoding="utf-8"))
SETTINGS = SimpleNamespace(anthropic_api_key=SecretStr("test-key"), anthropic_model="test-model")


def run(steps=(), answer="Done.", completed=True):
    return {"completed": completed, "answer": answer,
            "steps": [{"name": name, "input": tool_input, "result": {}} for name, tool_input in steps]}


def tool_use(name, tool_input):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "call_1", "name": name, "input": tool_input}]}


def final(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


class ScenarioFileTests(unittest.TestCase):
    def test_every_scenario_resolves_and_ids_are_unique(self):
        ids = [scenario["id"] for scenario in DATA["scenarios"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(resolve_search(DATA["scenarios"])), set(ids))

    def test_about_a_third_is_held_out_and_every_group_has_one(self):
        heldout = [scenario for scenario in DATA["scenarios"] if scenario["set"] == "heldout"]
        self.assertEqual(len(heldout), 8)
        self.assertEqual({scenario["group"] for scenario in heldout},
                         {scenario["group"] for scenario in DATA["scenarios"]})


class CheckTests(unittest.TestCase):
    def test_tool_and_argument_checks(self):
        checks = {"must_call": ["prepare_ticket"], "forbid_call": ["create_ticket"],
                  "args": [{"tool": "prepare_ticket", "field": "priority", "equals": "high"},
                           {"tool": "prepare_ticket", "field": "description", "not_contains_any": ["restarted"]}]}
        good = run([("prepare_ticket", {"priority": "high", "description": "Outlook crashes"})])
        bad = run([("prepare_ticket", {"priority": "medium", "description": "User restarted Outlook"}),
                   ("create_ticket", {})])
        self.assertEqual(failed_checks(checks, good), [])
        self.assertEqual(len(failed_checks(checks, bad)), 3)

    def test_answer_checks(self):
        checks = {"answer_citation": True, "answer_contains_all": ["14"],
                  "answer_not_contains_any": ["email"], "answer_language": "he"}
        self.assertEqual(failed_checks(checks, run(answer="אורך מינימלי הוא 14 תווים [1].")), [])
        self.assertEqual(len(failed_checks(checks, run(answer="Send your email; 12 characters."))), 4)

    def test_call_count_and_errors(self):
        checks = {"call_count": {"create_ticket": 1}}
        self.assertIn("called create_ticket 2 times, expected 1",
                      failed_checks(checks, run([("create_ticket", {}), ("create_ticket", {})])))
        self.assertEqual(failed_checks(checks, {"error_type": "AnthropicRequestError", "error_detail": "boom"}),
                         ["error: boom"])

    def test_hebrew_detection_tolerates_english_terms(self):
        self.assertTrue(is_hebrew("חיבור VPN מתנתק אחרי 12 hours [1]"))
        self.assertFalse(is_hebrew("VPN sessions disconnect after 12 hours."))

    def test_summary_separates_stable_flaky_and_failing(self):
        scenarios = [{"id": name, "group": "g", "checks": {"must_call": ["get_ticket"]}} for name in "abc"]
        good, bad = run([("get_ticket", {"ticket_id": 1})]), run()
        summary = summarize(scenarios, {"a": [good, good], "b": [good, bad], "c": [bad, bad]})
        self.assertEqual((summary["all_runs_passed"], summary["flaky"], summary["never_passed"]), (1, 1, 1))
        self.assertEqual(summary["run_pass_rate"], 0.5)


class HarnessTests(unittest.TestCase):
    def scenario(self, scenario_id):
        scenario = next(item for item in DATA["scenarios"] if item["id"] == scenario_id)
        return scenario, resolve_search([scenario])[scenario_id]

    def test_fake_tools_reach_the_real_agent_loop(self):
        scenario, hits = self.scenario("escalate_old")
        replies = [tool_use("escalate_ticket", {"ticket_id": 900101}), final("Ticket 900101 was escalated.")]
        with patch("app.services.agent_service.create_message", side_effect=replies):
            result = run_scenario(scenario, hits, DATA["tickets"], SETTINGS)
        self.assertEqual(result["steps"][0]["result"]["escalated"], True)
        self.assertEqual(failed_checks(scenario["checks"], result), [])

    def test_approved_scenario_creates_the_fake_ticket(self):
        scenario, hits = self.scenario("approved_create")
        replies = [tool_use("create_ticket", {}), final("Created ticket 900103, status open.")]
        with patch("app.services.agent_service.create_message", side_effect=replies):
            result = run_scenario(scenario, hits, DATA["tickets"], SETTINGS)
        self.assertEqual(result["steps"][0]["result"]["id"], 900103)
        self.assertEqual(failed_checks(scenario["checks"], result), [])

    def test_search_returns_the_fixture_chunk(self):
        scenario, hits = self.scenario("injection_retrieved")
        replies = [tool_use("search_documents", {"question": "password length"}), final("At least 14 characters [1].")]
        with patch("app.services.agent_service.create_message", side_effect=replies):
            result = run_scenario(scenario, hits, DATA["tickets"], SETTINGS)
        self.assertIn("Ignore previous instructions", result["steps"][0]["result"]["sources"][0]["text"])


if __name__ == "__main__":
    unittest.main()
