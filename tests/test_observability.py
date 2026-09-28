import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
from app.observability import current_trace, new_trace
from app.services.anthropic_client import create_message
from app.services.search_service import search_documents
from app.schemas import SearchRequest


class ObservabilityTests(unittest.TestCase):
    def test_request_logs_success_and_validation_failure(self):
        with patch("app.observability.logger.info") as log:
            good = TestClient(app).get("/health")
            bad = TestClient(app).post("/agent", json={"message": " "})
        self.assertEqual(good.status_code, 200)
        self.assertEqual(bad.status_code, 422)
        events = [json.loads(call.args[0]) for call in log.call_args_list]
        self.assertEqual([event["status"] for event in events], ["success", "failure"])
        self.assertEqual(events[0]["request_id"], good.headers["X-Request-ID"])
        self.assertEqual(events[1]["request_id"], bad.headers["X-Request-ID"])
        self.assertNotEqual(events[0]["request_id"], events[1]["request_id"])
        self.assertEqual(events[0]["number_of_tool_calls"], 0)
        self.assertIsNone(events[0]["total_tokens"])

    def test_llm_usage_accumulates_across_calls(self):
        trace = new_trace("/agent")
        token = current_trace.set(trace)
        response = {"content": [], "usage": {"input_tokens": 12, "output_tokens": 8}}
        try:
            with patch("app.services.anthropic_client.urlopen") as urlopen:
                urlopen.return_value.__enter__.side_effect = [
                    io.BytesIO(json.dumps(response).encode()),
                    io.BytesIO(json.dumps(response).encode()),
                ]
                create_message("system", [], "secret", "test-model")
                create_message("system", [], "secret", "test-model")
        finally:
            current_trace.reset(token)
        self.assertEqual(trace.llm_model, "test-model")
        self.assertEqual(trace.input_tokens + trace.output_tokens, 40)
        self.assertGreaterEqual(trace.llm_latency_ms, 0)

    def test_search_records_retrieval_and_embedding_model(self):
        trace = new_trace("/search")
        token = current_trace.set(trace)
        try:
            with patch("app.services.search_service.get_settings") as settings:
                settings.return_value.voyage_api_key = None
                with self.assertRaises(Exception):
                    search_documents(SearchRequest(question="policy"))
        finally:
            current_trace.reset(token)
        self.assertEqual(trace.embedding_model, "voyage-4")
        self.assertGreaterEqual(trace.retrieval_latency_ms, 0)

    def test_agent_tool_event_and_summary_share_request_id(self):
        first = {"stop_reason": "tool_use", "content": [
            {"type": "tool_use", "id": "toolu_1", "name": "get_ticket", "input": {"ticket_id": "bad"}}
        ]}
        final = {"stop_reason": "end_turn", "content": [
            {"type": "text", "text": "Please provide a numeric ticket ID."}
        ]}
        settings = SimpleNamespace(anthropic_api_key=SecretStr("test-key"), anthropic_model="test-model")
        with patch("app.services.agent_service.get_settings", return_value=settings):
            with patch("app.services.agent_service.create_message", side_effect=[first, final]):
                with patch("app.observability.logger.info") as log:
                    response = TestClient(app).post("/agent", json={"message": "Check ticket"})
        self.assertEqual(response.status_code, 200)
        events = [json.loads(call.args[0]) for call in log.call_args_list]
        self.assertEqual([event["event"] for event in events], ["tool_call", "request"])
        self.assertEqual(events[0]["tool"], "get_ticket")
        self.assertEqual(events[0]["status"], "failure")
        self.assertEqual(events[1]["tool_names"], ["get_ticket"])
        self.assertEqual(events[1]["number_of_tool_calls"], 1)
        self.assertEqual(events[0]["request_id"], response.headers["X-Request-ID"])


if __name__ == "__main__":
    unittest.main()
