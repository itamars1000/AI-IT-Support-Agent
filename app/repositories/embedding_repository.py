import json

from psycopg.rows import dict_row

from app.database import get_connection


def select_unembedded_chunks(provider: str, model: str, limit: int = 32) -> list[tuple[int, str]]:
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.text
            FROM document_chunks AS c
            LEFT JOIN chunk_embeddings AS e
                ON e.chunk_id = c.id AND e.provider = %s AND e.model = %s
            WHERE e.chunk_id IS NULL
            ORDER BY c.id
            LIMIT %s
            """,
            (provider, model, limit),
        ).fetchall()
        return [(row[0], row[1]) for row in rows]


def insert_embeddings(
    provider: str, model: str, items: list[tuple[int, list[float]]]
) -> None:
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.executemany(
                """
                INSERT INTO chunk_embeddings (chunk_id, provider, model, embedding)
                VALUES (%s, %s, %s, %s::vector)
                ON CONFLICT (chunk_id, provider, model) DO NOTHING
                """,
                [
                    (chunk_id, provider, model, json.dumps(vector, allow_nan=False))
                    for chunk_id, vector in items
                ],
            )


def search_chunks(
    query_vector: list[float], provider: str, model: str, limit: int, document_id: int | None
) -> list[dict]:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT c.id AS chunk_id, c.document_id, d.source_file,
                       c.page_number, c.chunk_index, c.text,
                       e.embedding <=> %s::vector AS distance
                FROM chunk_embeddings AS e
                JOIN document_chunks AS c ON c.id = e.chunk_id
                JOIN documents AS d ON d.id = c.document_id
                WHERE e.provider = %s AND e.model = %s
                  AND (%s::bigint IS NULL OR c.document_id = %s)
                ORDER BY distance, c.id
                LIMIT %s
                """,
                (json.dumps(query_vector, allow_nan=False), provider, model, document_id, document_id, limit),
            )
            return cursor.fetchall()


def search_chunks_for_documents(
    query_vector: list[float], provider: str, model: str, limit: int, document_ids: list[int]
) -> list[dict]:
    """Search only the explicitly selected evaluation corpus."""
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT c.id AS chunk_id, c.document_id, d.source_file,
                       c.page_number, c.chunk_index, c.text,
                       e.embedding <=> %s::vector AS distance
                FROM chunk_embeddings AS e
                JOIN document_chunks AS c ON c.id = e.chunk_id
                JOIN documents AS d ON d.id = c.document_id
                WHERE e.provider = %s AND e.model = %s
                  AND c.document_id = ANY(%s)
                ORDER BY distance, c.id
                LIMIT %s
                """,
                (json.dumps(query_vector, allow_nan=False), provider, model, document_ids, limit),
            )
            return cursor.fetchall()
