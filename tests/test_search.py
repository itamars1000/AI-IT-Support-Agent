import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from pydantic import SecretStr

from app.main import app
from app.schemas import SearchRequest
from app.services.search_service import search_documents
from app.services.voyage_client import VoyageRequestError, embed_texts


class SearchTests(unittest.TestCase):
    def test_query_uses_query_input_type(self):
        response = io.BytesIO(b'{"data":[{"index":0,"embedding":[1.0,0.0]}]}')
        with patch("app.services.voyage_client.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            embed_texts(["password policy?"], "test-key", "voyage-4", 2, input_type="query")
            body = json.loads(urlopen.call_args.args[0].data)
            self.assertEqual(body["input_type"], "query")

    def test_service_passes_vector_and_filters_to_repository(self):
        vector = [1.0] + [0.0] * 1023
        row = {
            "chunk_id": 7, "document_id": 5, "source_file": "policy.pdf",
            "page_number": 2, "chunk_index": 1, "text": "Password policy",
            "distance": 0.12,
        }
        settings = SimpleNamespace(voyage_api_key=SecretStr("test-key"))
        with patch("app.services.search_service.get_settings", return_value=settings):
            with patch("app.services.search_service.embed_texts", return_value=[vector]) as embed:
                with patch("app.services.search_service.embedding_repository.search_chunks", return_value=[row]) as repo:
                    hits = search_documents(SearchRequest(question="  password policy?  ", top_k=3, document_id=5))
        embed.assert_called_once_with(
            ["password policy?"], api_key="test-key", model="voyage-4",
            dimensions=1024, input_type="query",
        )
        repo.assert_called_once_with(vector, "voyage", "voyage-4", 3, 5)
        self.assertEqual(hits[0].page_number, 2)
        self.assertEqual(hits[0].distance, 0.12)

    def test_invalid_vector_never_reaches_database(self):
        settings = SimpleNamespace(voyage_api_key=SecretStr("test-key"))
        with patch("app.services.search_service.get_settings", return_value=settings):
            with patch("app.services.search_service.embed_texts", return_value=[[float("nan")] * 1024]):
                with patch("app.services.search_service.embedding_repository.search_chunks") as repo:
                    with self.assertRaises(VoyageRequestError):
                        search_documents(SearchRequest(question="password policy?"))
                    repo.assert_not_called()

    def test_http_validates_request(self):
        client = TestClient(app)
        for payload in ({"question": "  "}, {"question": "x", "top_k": 0}, {"question": "x", "document_id": -1}):
            self.assertEqual(client.post("/search", json=payload).status_code, 422)

    def test_http_reports_missing_provider_configuration(self):
        with patch("app.services.search_service.get_settings", return_value=SimpleNamespace(voyage_api_key=None)):
            response = TestClient(app).post("/search", json={"question": "password policy?"})
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Embedding provider is not configured"})


if __name__ == "__main__":
    unittest.main()
