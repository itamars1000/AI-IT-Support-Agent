"""The chat receives measured execution metadata, including failures, without secrets."""

import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import URLError

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
from app.services.embedding_service import EMBEDDING_DIMENSIONS


SETTINGS = SimpleNamespace(anthropic_api_key=SecretStr("private-anthropic-key"),
                           anthropic_model="test-model", voyage_api_key=SecretStr("private-voyage-key"),
                           retrieval_mode="vector")
HIT = {"chunk_id": 1, "document_id": 2, "source_file": "vpn.pdf", "page_number": 1,
       "chunk_index": 0, "text": "Private document content", "distance": 0.1}


def tool(name, arguments):
    return {"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "toolu_test", "name": name, "input": arguments}],
        "usage": {"input_tokens": 10, "output_tokens": 5}}


def final(answer="Answer [1]."):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": answer}],
            "usage": {"input_tokens": 20, "output_tokens": 8}}


def provider_response(data):
    return io.BytesIO(json.dumps(data).encode())


class RequestActivityTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_multi_step_trace_has_ordered_calls_real_metrics_and_no_content(self):
        proposal = {"title": "VPN issue", "description": "Private ticket details", "priority": "medium"}
        replies = [tool("search_documents", {"question": "Private search query"}),
                   tool("prepare_ticket", proposal), final()]
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.search_service.get_settings", return_value=SETTINGS), \
             patch("app.services.search_service.embed_texts", return_value=[[0.0] * EMBEDDING_DIMENSIONS]), \
             patch("app.services.search_service.embedding_repository.search_chunks", return_value=[HIT]), \
             patch("app.services.anthropic_client.urlopen") as provider:
            provider.return_value.__enter__.side_effect = [provider_response(reply) for reply in replies]
            response = self.client.post("/agent", json={"message": "Private user message"})
        self.assertEqual(response.status_code, 200)
        trace = response.json()["trace"]
        self.assertEqual(trace["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(trace["endpoint"], "/agent")
        self.assertEqual(trace["status"], "success")
        self.assertEqual([event["kind"] for event in trace["events"]], ["llm", "tool", "llm", "tool", "llm"])
        self.assertEqual(trace["tool_names"], ["search_documents", "prepare_ticket"])
        self.assertEqual(trace["number_of_tool_calls"], 2)
        self.assertEqual(trace["total_tokens"], 58)
        self.assertEqual(trace["llm_model"], "test-model")
        self.assertIsNotNone(trace["embedding_model"])
        self.assertGreater(trace["retrieval_latency_ms"], 0)
        self.assertAlmostEqual(trace["llm_latency_ms"], sum(event["latency_ms"] for event in trace["events"] if event["kind"] == "llm"), places=3)
        self.assertGreaterEqual(trace["latency_ms"], trace["llm_latency_ms"])
        serialized = json.dumps(trace)
        for private in ("private-anthropic-key", "private-voyage-key", "Private user message",
                        "Private search query", "Private ticket details", "Private document content", "input", "result"):
            self.assertNotIn(private, serialized)

    def test_plain_answer_has_no_tools_or_embedding_and_requests_are_isolated(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.anthropic_client.urlopen") as provider:
            provider.return_value.__enter__.side_effect = [provider_response(final("Hello")), provider_response(final("Hi"))]
            first = self.client.post("/agent", json={"message": "Hello"}).json()["trace"]
            second = self.client.post("/agent", json={"message": "Hi"}).json()["trace"]
        self.assertNotEqual(first["request_id"], second["request_id"])
        for trace in (first, second):
            self.assertEqual(trace["number_of_tool_calls"], 0)
            self.assertIsNone(trace["embedding_model"])
            self.assertEqual(len(trace["events"]), 1)
            self.assertEqual(trace["total_tokens"], 28)

    def test_failed_tool_can_be_recovered_without_calling_the_request_failed(self):
        replies = [tool("get_ticket", {"ticket_id": "invalid"}), final("Please provide a numeric ticket ID.")]
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.anthropic_client.urlopen") as provider:
            provider.return_value.__enter__.side_effect = [provider_response(reply) for reply in replies]
            data = self.client.post("/agent", json={"message": "Read ticket"}).json()
        self.assertTrue(data["completed"])
        self.assertEqual(data["trace"]["status"], "success")
        self.assertEqual(data["trace"]["events"][1]["status"], "failure")

    def test_provider_failure_before_any_tool_returns_a_failure_trace(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.anthropic_client.urlopen", side_effect=URLError("Private provider details")):
            response = self.client.post("/agent", json={"message": "Hello"})
        self.assertEqual(response.status_code, 502)
        trace = response.json()["trace"]
        self.assertEqual(trace["status"], "failure")
        self.assertEqual(trace["request_id"], response.headers["X-Request-ID"])
        self.assertEqual(trace["events"][0]["status"], "failure")
        self.assertIsNone(trace["total_tokens"])
        self.assertNotIn("Private provider details", response.text)

    def test_provider_failure_after_a_tool_preserves_partial_activity(self):
        with patch("app.services.agent_service.get_settings", return_value=SETTINGS), \
             patch("app.services.anthropic_client.urlopen") as provider:
            provider.return_value.__enter__.side_effect = [provider_response(tool("get_ticket", {"ticket_id": "invalid"})), URLError("offline")]
            response = self.client.post("/agent", json={"message": "Read ticket"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertFalse(data["completed"])
        self.assertEqual(data["trace"]["status"], "failure")
        self.assertEqual([event["status"] for event in data["trace"]["events"]], ["success", "failure", "failure"])


if __name__ == "__main__":
    unittest.main()
