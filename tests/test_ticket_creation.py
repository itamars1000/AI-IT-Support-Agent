import os
import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from unittest.mock import patch
from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from app.database import get_connection
from app.errors import IdempotencyConflict
from app.main import app
from app.repositories import ticket_repository
from app.schemas import TicketCreate, TicketResponse


class TicketCreationApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.payload = {"title": "VPN still fails", "description": "Restarted the client; error 809 persists.", "priority": "high"}
        self.key = uuid4()
        self.ticket = TicketResponse(id=431, **self.payload, status="open", created_at=datetime.now(timezone.utc))

    def test_exact_approved_details_and_key_reach_service(self):
        with patch("app.routers.tickets.ticket_service.create_ticket", return_value=self.ticket) as create:
            response = self.client.post("/tickets", json=self.payload, headers={"Idempotency-Key": str(self.key)})
        self.assertEqual(response.status_code, 201)
        create.assert_called_once_with(TicketCreate(**self.payload), self.key)
        self.assertEqual(response.json()["id"], 431)
        self.assertEqual(response.json()["description"], self.payload["description"])

    def test_invalid_key_or_fields_never_create_a_ticket(self):
        with patch("app.routers.tickets.ticket_service.create_ticket") as create:
            response = self.client.post("/tickets", json=self.payload, headers={"Idempotency-Key": "not-a-uuid"})
            self.assertEqual(response.status_code, 422)
            for change in ({"title": "  "}, {"description": ""}, {"priority": "urgent"}, {"title": "x" * 201}, {"description": "x" * 5001}):
                self.assertEqual(self.client.post("/tickets", json={**self.payload, **change}).status_code, 422)
        create.assert_not_called()

    def test_existing_clients_can_still_create_without_header(self):
        with patch("app.routers.tickets.ticket_service.create_ticket", return_value=self.ticket) as create:
            self.assertEqual(self.client.post("/tickets", json=self.payload).status_code, 201)
        create.assert_called_once_with(TicketCreate(**self.payload), None)

    def test_conflicting_request_is_reported_to_client(self):
        with patch("app.routers.tickets.ticket_service.create_ticket", side_effect=IdempotencyConflict()):
            response = self.client.post("/tickets", json=self.payload, headers={"Idempotency-Key": str(self.key)})
        self.assertEqual(response.status_code, 409)


@unittest.skipUnless(os.environ.get("RUN_DATABASE_TESTS") == "1", "Set RUN_DATABASE_TESTS=1 for PostgreSQL integration tests")
class TicketCreationDatabaseTests(unittest.TestCase):
    """The real HTTP/service/repository path; test tickets are rolled back."""

    def setUp(self):
        self.connection = get_connection()
        self.transaction = self.connection.transaction(force_rollback=True)
        self.transaction.__enter__()
        self.addCleanup(self.connection.close)
        self.addCleanup(self.transaction.__exit__, None, None, None)
        repo_patch = patch.object(ticket_repository, "get_connection", side_effect=self.borrow_connection)
        repo_patch.start()
        self.addCleanup(repo_patch.stop)
        self.client = TestClient(app)
        self.key = uuid4()
        self.headers = {"Idempotency-Key": str(self.key)}
        self.payload = {"title": "Synthetic UI test", "description": "VPN fails after restarting the client.", "priority": "medium"}

    @contextmanager
    def borrow_connection(self):
        yield self.connection

    def test_repeated_http_request_returns_one_database_ticket(self):
        first = self.client.post("/tickets", json=self.payload, headers=self.headers)
        second = self.client.post("/tickets", json=self.payload, headers=self.headers)
        self.assertEqual((first.status_code, second.status_code), (201, 201))
        self.assertEqual(first.json(), second.json())
        count = self.connection.execute("SELECT count(*) FROM tickets WHERE creation_request_id=%s", (self.key,)).fetchone()[0]
        self.assertEqual(count, 1)

    def test_changed_payload_cannot_reuse_creation_key(self):
        self.assertEqual(self.client.post("/tickets", json=self.payload, headers=self.headers).status_code, 201)
        for change in ({"title": "Changed issue"}, {"description": "Changed details"}, {"priority": "high"}):
            response = self.client.post("/tickets", json={**self.payload, **change}, headers=self.headers)
            self.assertEqual(response.status_code, 409)
        row = self.connection.execute("SELECT title,description,priority FROM tickets WHERE creation_request_id=%s", (self.key,)).fetchone()
        self.assertEqual(row, (self.payload["title"], self.payload["description"], self.payload["priority"]))


if __name__ == "__main__":
    unittest.main()
