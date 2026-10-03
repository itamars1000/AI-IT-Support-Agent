"""A proposal cannot write; confirmed creation binds model arguments to reviewed fields."""

import os
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.database import get_connection
from app.main import app
from app.repositories import ticket_repository
from app.schemas import AgentRequest, TicketCreate, TicketResponse
from app.services.agent_service import run_agent
from app.services.anthropic_client import AnthropicRequestError


SETTINGS = SimpleNamespace(anthropic_api_key=SecretStr("test-key"), anthropic_model="test-model")
PAYLOAD = {"title": "VPN error 809 persists", "description": "User tested a mobile hotspot; VPN error 809 persists. Gateway log collected.", "priority": "medium"}


def call(name, payload=PAYLOAD):
    return {"stop_reason": "tool_use", "content": [{"type": "tool_use", "id": "toolu_approval", "name": name, "input": payload}]}


def final(text="Review and confirm the ticket proposal."):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def approved_request(key=None):
    return AgentRequest(message="Create the reviewed ticket", allowed_actions=["create_ticket"],
                        approved_ticket=PAYLOAD, idempotency_key=key or uuid4())


class AgentApprovalTests(unittest.TestCase):
    def test_proposal_uses_tool_but_never_creates_a_ticket(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", side_effect=[call("prepare_ticket"), final()]) as message, \
             patch("app.services.agent_service.ticket_service.create_ticket") as create:
            response = run_agent(AgentRequest(message="Open a ticket; error 809 persists after hotspot testing."))
        create.assert_not_called()
        self.assertTrue(response.completed)
        self.assertEqual(response.ticket_proposal, TicketCreate(**PAYLOAD))
        self.assertEqual([step.name for step in response.steps], ["prepare_ticket"])
        self.assertNotIn("create_ticket", [tool["name"] for tool in message.call_args.kwargs["tools"]])

    def test_invalid_proposal_is_not_exposed_for_confirmation(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", side_effect=[call("prepare_ticket", {**PAYLOAD, "description": ""}), final("Please describe the issue.")]), \
             patch("app.services.agent_service.ticket_service.create_ticket") as create:
            response = run_agent(AgentRequest(message="Open a ticket"))
        create.assert_not_called()
        self.assertIsNone(response.ticket_proposal)
        self.assertEqual(response.steps[0].result["error"], "Invalid tool arguments")

    def test_only_exact_approved_fields_reach_creation_service(self):
        request = approved_request()
        ticket = TicketResponse(id=431, **PAYLOAD, status="open", created_at=datetime.now(timezone.utc))
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", side_effect=[call("create_ticket", {}), final("Ticket 431 created.")]) as message, \
             patch("app.services.agent_service.ticket_service.create_ticket", return_value=ticket) as create:
            response = run_agent(request)
        create.assert_called_once_with(request.approved_ticket, request.idempotency_key)
        self.assertEqual(response.steps[0].result["id"], 431)
        self.assertEqual(message.call_args_list[0].kwargs["tool_choice"]["name"], "create_ticket")
        self.assertEqual(message.call_args_list[1].kwargs["tool_choice"], {"type": "none"})
        schema = message.call_args_list[0].kwargs["tools"][0]["input_schema"]
        self.assertEqual(schema["properties"], {})
        self.assertFalse(schema["additionalProperties"])

    def test_model_cannot_change_any_approved_field(self):
        for change in ({"title": "Changed issue"}, {"description": "Unapproved details"}, {"priority": "high"}):
            with self.subTest(change=change), \
                 patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
                 patch("app.services.agent_service.create_message", side_effect=[call("create_ticket", {**PAYLOAD, **change}), final("Created.")]), \
                 patch("app.services.agent_service.ticket_service.create_ticket") as create:
                response = run_agent(approved_request())
            create.assert_not_called()
            self.assertFalse(response.completed)
            self.assertIn("approved ticket", response.steps[0].result["error"])

    def test_text_claim_without_tool_result_does_not_confirm_creation(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", return_value=final("Ticket 431 created.")), \
             patch("app.services.agent_service.ticket_service.create_ticket") as create:
            with self.assertRaises(AnthropicRequestError):
                run_agent(approved_request())
        create.assert_not_called()

    def test_failure_after_creation_preserves_authoritative_receipt(self):
        ticket = TicketResponse(id=431, **PAYLOAD, status="open", created_at=datetime.now(timezone.utc))
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", side_effect=[call("create_ticket", {}), AnthropicRequestError("Lost final answer")]), \
             patch("app.services.agent_service.ticket_service.create_ticket", return_value=ticket):
            response = run_agent(approved_request())
        self.assertFalse(response.completed)
        self.assertEqual(response.steps[0].result["id"], 431)

    def test_approval_requires_current_permission_key_and_valid_fields(self):
        client = TestClient(app)
        valid = approved_request().model_dump(mode="json")
        with patch("app.routers.agent.run_agent") as agent:
            for change in ({"allowed_actions": []}, {"allowed_actions": ["create_ticket", "escalate_ticket"]},
                           {"idempotency_key": None}, {"approved_ticket": {**PAYLOAD, "priority": "urgent"}}):
                with self.subTest(change=change):
                    self.assertEqual(client.post("/agent", json={**valid, **change}).status_code, 422)
        agent.assert_not_called()


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Set RUN_DATABASE_TESTS=1 for PostgreSQL")
class AgentApprovalDatabaseTests(unittest.TestCase):
    def setUp(self):
        self.connection = get_connection()
        self.transaction = self.connection.transaction(force_rollback=True)
        self.transaction.__enter__()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.__exit__, None, None, None)

        @contextmanager
        def borrow():
            yield self.connection

        repo_patch = patch.object(ticket_repository, "get_connection", side_effect=borrow)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)
        self.client = TestClient(app)

    def test_retry_after_lost_final_answer_returns_one_database_ticket(self):
        key = uuid4()
        payload = approved_request(key).model_dump(mode="json")
        replies = [call("create_ticket", {}), AnthropicRequestError("Lost confirmation"), call("create_ticket", {}), final("Created.")]
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.agent_service.create_message", side_effect=replies):
            first = self.client.post("/agent", json=payload)
            second = self.client.post("/agent", json=payload)
        self.assertEqual((first.status_code, second.status_code), (200, 200))
        self.assertFalse(first.json()["completed"])
        self.assertEqual(first.json()["steps"][0]["result"], second.json()["steps"][0]["result"])
        row = self.connection.execute("SELECT count(*) FROM tickets WHERE creation_request_id=%s", (key,)).fetchone()
        self.assertEqual(row[0], 1)


if __name__ == "__main__":
    unittest.main()
