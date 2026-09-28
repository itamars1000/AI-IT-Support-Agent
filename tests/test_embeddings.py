import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.services.embedding_service import EMBEDDING_DIMENSIONS, embed_pending_chunks
from app.services.voyage_client import VoyageRequestError, embed_texts


class EmbeddingTests(unittest.TestCase):
    def test_missing_key_does_not_read_chunks_or_call_provider(self):
        with patch("app.services.embedding_service.get_settings", return_value=SimpleNamespace(voyage_api_key=None)):
            with patch("app.services.embedding_service.embedding_repository.select_unembedded_chunks") as select:
                with self.assertRaisesRegex(ValueError, "VOYAGE_API_KEY"):
                    embed_pending_chunks()
                select.assert_not_called()

    def test_batch_is_saved_after_valid_provider_response(self):
        vector = [0.0] * EMBEDDING_DIMENSIONS
        vector[0] = 1.0
        with patch("app.services.embedding_service.get_settings", return_value=SimpleNamespace(voyage_api_key=SecretStr("test-key"))):
            with patch("app.services.embedding_service.embedding_repository.select_unembedded_chunks", side_effect=[[(10, "Policy text")], []]):
                with patch("app.services.embedding_service.embed_texts", return_value=[vector]) as embed:
                    with patch("app.services.embedding_service.embedding_repository.insert_embeddings") as insert:
                        self.assertEqual(embed_pending_chunks(), 1)
                        self.assertEqual(embed.call_args.kwargs["model"], "voyage-4")
                        insert.assert_called_once_with("voyage", "voyage-4", [(10, vector)])

    def test_invalid_vector_is_not_saved(self):
        with patch("app.services.embedding_service.get_settings", return_value=SimpleNamespace(voyage_api_key=SecretStr("test-key"))):
            with patch("app.services.embedding_service.embedding_repository.select_unembedded_chunks", return_value=[(10, "Policy text")]):
                with patch("app.services.embedding_service.embed_texts", return_value=[[1.0, 2.0]]):
                    with patch("app.services.embedding_service.embedding_repository.insert_embeddings") as insert:
                        with self.assertRaisesRegex(ValueError, "invalid vector"):
                            embed_pending_chunks()
                        insert.assert_not_called()

    def test_http_adapter_reorders_results_by_index(self):
        data = {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}
        response = io.BytesIO(json.dumps(data).encode("utf-8"))
        with patch("app.services.voyage_client.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            vectors = embed_texts(["a", "b"], "test-key", "voyage-4", 2)
            self.assertEqual(vectors, [[1.0, 0.0], [0.0, 1.0]])
            request = urlopen.call_args.args[0]
            body = json.loads(request.data)
            self.assertEqual(body["input_type"], "document")
            self.assertEqual(body["output_dimension"], 2)
            self.assertFalse(body["truncation"])

    def test_http_adapter_rejects_missing_index(self):
        response = io.BytesIO(b'{"data":[{"index":1,"embedding":[1.0]}]}')
        with patch("app.services.voyage_client.urlopen") as urlopen:
            urlopen.return_value.__enter__.return_value = response
            with self.assertRaises(VoyageRequestError):
                embed_texts(["a"], "test-key", "voyage-4", 1)


if __name__ == "__main__":
    unittest.main()
