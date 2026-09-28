import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch
from uuid import UUID

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
from app.schemas import TicketResponse, ToolCallRequest
from app.services.anthropic_client import AnthropicRequestError
from app.services.tool_calling_service import run_tool_call


SETTINGS = SimpleNamespace(anthropic_api_key=SecretStr("test-key"), anthropic_model="test-model")
REQUEST_ID = UUID("11111111-1111-4111-8111-111111111111")
TICKET = TicketResponse(
    id=42, title="Laptop will not start", description="Screen remains black",
    priority="high", status="open", created_at=datetime(2026, 9, 28, tzinfo=timezone.utc),
)


class ToolCallingTests(unittest.TestCase):
    def test_create_ticket_round_trip_executes_once(self):
        first = {
            "stop_reason": "tool_use",
            "content": [{"type": "tool_use", "id": "toolu_1", "name": "create_ticket", "input": {
                "title": "Laptop will not start", "description": "Screen remains black", "priority": "high",
            }}],
        }
        second = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Created ticket #42."}]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", side_effect=[first, second]) as message:
                with patch("app.services.tool_calling_service.ticket_service.create_ticket", return_value=TICKET) as create:
                    response = run_tool_call(ToolCallRequest(
                        message="Open a high priority ticket: my laptop will not start",
                        allowed_actions=["create_ticket"], idempotency_key=REQUEST_ID,
                    ))
        create.assert_called_once()
        self.assertEqual(create.call_args.args[0].priority.value, "high")
        self.assertEqual(response.answer, "Created ticket #42.")
        self.assertEqual(response.tool_call.result.id, 42)
        self.assertEqual(message.call_count, 2)
        followup_messages = message.call_args.args[1]
        self.assertEqual(followup_messages[1], {"role": "assistant", "content": first["content"]})
        result_block = followup_messages[2]["content"][0]
        self.assertEqual(result_block["tool_use_id"], "toolu_1")
        self.assertEqual(json.loads(result_block["content"])["id"], 42)
        self.assertEqual(message.call_args.kwargs["tool_choice"], {"type": "none"})

    def test_get_ticket_calls_read_service(self):
        first = {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_2", "name": "get_ticket", "input": {"ticket_id": 42}}
        ]}
        second = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Ticket #42 is open."}]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", side_effect=[first, second]):
                with patch("app.services.tool_calling_service.ticket_service.get_ticket", return_value=TICKET) as get:
                    response = run_tool_call(ToolCallRequest(message="Check ticket 42"))
        get.assert_called_once_with(42)
        self.assertEqual(response.tool_call.name, "get_ticket")

    def test_text_response_does_not_execute_tool(self):
        first = {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Which ticket ID?"}]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", return_value=first):
                with patch("app.services.tool_calling_service.ticket_service.create_ticket") as create:
                    response = run_tool_call(ToolCallRequest(message="Help me with a ticket"))
        self.assertEqual(response.answer, "Which ticket ID?")
        self.assertIsNone(response.tool_call)
        create.assert_not_called()

    def test_invalid_arguments_do_not_execute_tool(self):
        first = {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_3", "name": "create_ticket", "input": {"title": "Laptop"}}
        ]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", return_value=first):
                with patch("app.services.tool_calling_service.ticket_service.create_ticket") as create:
                    with self.assertRaises(AnthropicRequestError):
                        run_tool_call(ToolCallRequest(
                            message="Open a ticket", allowed_actions=["create_ticket"], idempotency_key=REQUEST_ID
                        ))
        create.assert_not_called()

    def test_multiple_calls_do_not_execute_any_tool(self):
        call = {"type": "tool_use", "id": "toolu_4", "name": "get_ticket", "input": {"ticket_id": 42}}
        first = {"stop_reason": "tool_use", "content": [call, {**call, "id": "toolu_5"}]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", return_value=first):
                with patch("app.services.tool_calling_service.ticket_service.get_ticket") as get:
                    with self.assertRaises(AnthropicRequestError):
                        run_tool_call(ToolCallRequest(message="Check tickets"))
        get.assert_not_called()

    def test_second_api_failure_preserves_created_ticket_result(self):
        first = {"stop_reason": "tool_use", "content": [{
            "type": "tool_use", "id": "toolu_6", "name": "create_ticket",
            "input": {"title": "Laptop will not start", "description": "Screen remains black", "priority": "high"},
        }]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", side_effect=[first, AnthropicRequestError("failed")]):
                with patch("app.services.tool_calling_service.ticket_service.create_ticket", return_value=TICKET) as create:
                    response = run_tool_call(ToolCallRequest(
                        message="Open a high priority ticket", allowed_actions=["create_ticket"],
                        idempotency_key=REQUEST_ID,
                    ))
        create.assert_called_once()
        self.assertEqual(response.tool_call.result.id, 42)
        self.assertIn("42", response.answer)

    def test_http_rejects_empty_message(self):
        response = TestClient(app).post("/tool-call", json={"message": "  "})
        self.assertEqual(response.status_code, 422)

    def test_write_tool_is_not_available_without_permission(self):
        first = {"stop_reason": "tool_use", "content": [{
            "type": "tool_use", "id": "toolu_7", "name": "create_ticket",
            "input": {"title": "Laptop", "description": "Black screen"},
        }]}
        with patch("app.services.tool_calling_service.get_settings", return_value=SETTINGS):
            with patch("app.services.tool_calling_service.create_message", return_value=first) as message:
                with patch("app.services.tool_calling_service.ticket_service.create_ticket") as create:
                    with self.assertRaises(AnthropicRequestError):
                        run_tool_call(ToolCallRequest(message="Open a ticket"))
        create.assert_not_called()
        self.assertNotIn("create_ticket", [tool["name"] for tool in message.call_args.kwargs["tools"]])


if __name__ == "__main__":
    unittest.main()
