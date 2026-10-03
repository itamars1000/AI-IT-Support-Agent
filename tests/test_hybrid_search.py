import os
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.database import get_connection
from app.repositories import embedding_repository as repository
from app.schemas import SearchRequest
from app.services.search_service import search_documents


class HybridRoutingTests(unittest.TestCase):
    def test_explicit_hybrid_uses_same_embedding_and_document_scope(self):
        vector = [1.0] + [0.0] * 1023
        settings = SimpleNamespace(voyage_api_key=SecretStr("test-key"), retrieval_mode="vector")
        with patch("app.services.search_service.get_settings", return_value=settings):
            with patch("app.services.search_service.embed_texts", return_value=[vector]):
                with patch.object(repository, "search_ranked_chunks", return_value=[]) as hybrid:
                    search_documents(SearchRequest(question="GP1001", top_k=3, document_id=7, retrieval_mode="hybrid"))
        hybrid.assert_called_once_with("GP1001", vector, "voyage", "voyage-4", 3, "hybrid", [7])

    def test_configured_mode_and_request_override(self):
        settings = SimpleNamespace(voyage_api_key=SecretStr("test-key"), retrieval_mode="hybrid")
        with patch("app.services.search_service.get_settings", return_value=settings):
            with patch("app.services.search_service.embed_texts", return_value=[[1.0] + [0.0] * 1023]):
                with patch.object(repository, "search_ranked_chunks", return_value=[]) as hybrid:
                    with patch.object(repository, "search_chunks", return_value=[]) as vector:
                        search_documents(SearchRequest(question="GP1001"))
                        search_documents(SearchRequest(question="GP1001", retrieval_mode="vector"))
        self.assertEqual(hybrid.call_count, 1)
        self.assertEqual(vector.call_count, 1)

    def test_punctuation_cannot_introduce_query_operators(self):
        self.assertEqual(repository.lexical_query("GP1001 -GP1002 | ! & '"), '"gp1001" OR "gp1002"')
        self.assertEqual(repository.lexical_query("KB-431 ERR_CLOCK_SKEW KB-431"), '"kb-431" OR "err_clock_skew"')
        self.assertEqual(repository.lexical_query("?!"), "")


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Set RUN_DATABASE_TESTS=1 for PostgreSQL integration tests")
class HybridDatabaseTests(unittest.TestCase):
    """Real SQL and generated columns, with every fixture rolled back."""

    def setUp(self):
        self.connection = get_connection()
        self.transaction = self.connection.transaction(force_rollback=True)
        self.transaction.__enter__()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.__exit__, None, None, None)
        self.patch = patch.object(repository, "get_connection", side_effect=self.borrow_connection)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.documents = []
        self.chunk_ids = []
        for text in (
            "GlobalProtect GP1001 disk encryption disabled. Restore encryption.",
            "GlobalProtect GP1002 endpoint protection disabled. Restore protection.",
            "GlobalProtect GP1001 unembedded document should be excluded.",
        ):
            doc_id = self.connection.execute("INSERT INTO documents (source_file, page_count) VALUES ('hybrid_test.txt', 1) RETURNING id").fetchone()[0]
            self.connection.execute("INSERT INTO document_pages(document_id, page_number, text) VALUES (%s, 1, %s)", (doc_id, text))
            chunk_id = self.connection.execute("INSERT INTO document_chunks(document_id,text,chunk_index,page_number) VALUES (%s,%s,0,1) RETURNING id", (doc_id, text)).fetchone()[0]
            self.documents.append(doc_id)
            self.chunk_ids.append(chunk_id)
        self.vector = [1.0] + [0.0] * 1023
        repository.insert_embeddings("test", "test", [(id, self.vector) for id in self.chunk_ids[:2]])

    @contextmanager
    def borrow_connection(self):
        yield self.connection

    def search(self, question, mode="hybrid", documents=None, limit=20):
        return repository.search_ranked_chunks(question, self.vector, "test", "test", limit, mode,
                                             self.documents if documents is None else documents)

    def test_identifier_ranking_and_rrf_score(self):
        hits = self.search("GP1002")
        self.assertEqual(hits[0]["chunk_id"], self.chunk_ids[1])
        self.assertEqual(hits[0]["vector_rank"], 2)
        self.assertEqual(hits[0]["lexical_rank"], 1)
        self.assertAlmostEqual(float(hits[0]["retrieval_score"]), 1 / 62 + 1 / 61)
        self.assertEqual(float(hits[0]["distance"]), 0)
        self.assertEqual(len({hit["chunk_id"] for hit in hits}), len(hits))

    def test_filters_apply_to_both_branches_before_ranking(self):
        hits = self.search("GP1002", documents=[self.documents[0]])
        self.assertEqual([hit["document_id"] for hit in hits], [self.documents[0]])
        self.assertIsNone(hits[0]["lexical_rank"])
        self.assertEqual(self.search("GP1002", documents=[]), [])
        self.assertEqual(repository.search_ranked_chunks("GP1002", self.vector, "other", "test", 20, "hybrid", self.documents), [])

    def test_scope_is_applied_before_twenty_candidate_cap(self):
        # Out-of-scope rows outrank the target in both branches if filtering is late.
        for index in range(1, 23):
            chunk_id = self.connection.execute(
                "INSERT INTO document_chunks(document_id,text,chunk_index,page_number) VALUES (%s,%s,%s,1) RETURNING id",
                (self.documents[0], "GP1002 GP1002 GP1002 repeated unrelated rule", index),
            ).fetchone()[0]
            repository.insert_embeddings("test", "test", [(chunk_id, self.vector)])
        import json
        target_vector = [0.5, 0.5] + [0.0] * 1022
        self.connection.execute("UPDATE chunk_embeddings SET embedding=%s::vector WHERE chunk_id=%s",
                                (json.dumps(target_vector), self.chunk_ids[1]))
        hits = self.search("GP1002", documents=[self.documents[1]], limit=1)
        self.assertEqual(hits[0]["chunk_id"], self.chunk_ids[1])
        self.assertEqual((hits[0]["vector_rank"], hits[0]["lexical_rank"]), (1, 1))

    def test_stopwords_and_no_matches_keep_vector_candidates(self):
        for question in ("the and", "?!", "ZZUNMATCHED987"):
            self.assertEqual(self.search(question, "lexical"), [])
            self.assertEqual([hit["chunk_id"] for hit in self.search(question)], self.chunk_ids[:2])

    def test_lexical_excludes_unembedded_rows_and_respects_limit(self):
        self.assertEqual([hit["chunk_id"] for hit in self.search("GP1001", "lexical")], self.chunk_ids[:1])
        self.assertEqual(len(self.search("GlobalProtect", limit=1)), 1)

    def test_generated_index_updates_when_text_changes(self):
        self.connection.execute("UPDATE document_chunks SET text='GP1003 replacement procedure' WHERE id=%s", (self.chunk_ids[0],))
        self.assertEqual(self.search("GP1001", "lexical"), [])
        self.assertEqual(self.search("GP1003", "lexical")[0]["chunk_id"], self.chunk_ids[0])

    def test_migration_can_be_applied_again(self):
        path = Path(__file__).resolve().parent.parent / "sql" / "007_add_chunk_text_search.sql"
        self.connection.execute(path.read_text(encoding="utf-8"))
        self.assertEqual(len(self.search("GlobalProtect", "lexical")), 2)


if __name__ == "__main__":
    unittest.main()
