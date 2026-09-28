from pathlib import Path

from app.database import get_connection


def initialize_database() -> None:
    schema_directory = Path(__file__).resolve().parent.parent / "sql"
    with get_connection() as connection:
        for filename in (
            "001_create_tickets.sql",
            "002_create_documents.sql",
            "003_create_document_chunks.sql",
            "004_create_chunk_embeddings.sql",
            "005_add_ticket_escalation.sql",
            "006_ticket_idempotency.sql",
        ):
            connection.execute((schema_directory / filename).read_text(encoding="utf-8"))
    print("Tickets, documents, pages, chunks and embeddings tables are ready.")


if __name__ == "__main__":
    initialize_database()
