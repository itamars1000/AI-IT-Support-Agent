import json
import re

from psycopg.rows import dict_row

from app.database import get_connection


CANDIDATE_LIMIT = 20
RRF_CONSTANT = 60


def lexical_query(question: str) -> str:
    """OR normalized words for recall; quote compound identifiers as phrases.

    Only literal word tokens reach websearch_to_tsquery, so user punctuation
    cannot introduce negation, tsquery operators or SQL syntax.
    """
    tokens = dict.fromkeys(re.findall(r"\w+(?:[.-]\w+)*", question.casefold()))
    return " OR ".join(f'"{token}"' for token in tokens)


def search_ranked_chunks(
    question: str, query_vector: list[float], provider: str, model: str,
    limit: int, mode: str, document_ids: list[int] | None = None,
) -> list[dict]:
    """Rank the same embedded corpus by lexical search or equal-weight RRF.

    distance remains cosine distance even when it is not the sorting score.
    The filters apply to both branches before either candidate limit.
    """
    if mode not in ("lexical", "hybrid"):
        raise ValueError("Expected lexical or hybrid retrieval")
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                WITH query AS (
                    SELECT websearch_to_tsquery('english', %s) AS terms
                ), eligible AS NOT MATERIALIZED (
                    SELECT c.id AS chunk_id, c.document_id, d.source_file,
                           c.page_number, c.chunk_index, c.text, c.search_vector,
                           e.embedding <=> %s::vector AS distance
                    FROM chunk_embeddings e
                    JOIN document_chunks c ON c.id = e.chunk_id
                    JOIN documents d ON d.id = c.document_id
                    WHERE e.provider = %s AND e.model = %s
                      AND (%s::bigint[] IS NULL OR c.document_id = ANY(%s))
                ), vector_candidates AS (
                    SELECT chunk_id, row_number() OVER (ORDER BY distance, chunk_id) AS vector_rank
                    FROM eligible ORDER BY distance, chunk_id LIMIT %s
                ), lexical_candidates AS (
                    SELECT chunk_id, ts_rank_cd(search_vector, query.terms) AS lexical_score,
                           row_number() OVER (
                               ORDER BY ts_rank_cd(search_vector, query.terms) DESC, chunk_id
                           ) AS lexical_rank
                    FROM eligible CROSS JOIN query
                    WHERE search_vector @@ query.terms
                    ORDER BY lexical_score DESC, chunk_id LIMIT %s
                ), fused AS (
                    SELECT coalesce(v.chunk_id, l.chunk_id) AS chunk_id,
                           v.vector_rank, l.lexical_rank, l.lexical_score,
                           coalesce(1.0 / (%s + v.vector_rank), 0) +
                           coalesce(1.0 / (%s + l.lexical_rank), 0) AS rrf_score
                    FROM vector_candidates v FULL OUTER JOIN lexical_candidates l USING (chunk_id)
                )
                SELECT e.chunk_id, e.document_id, e.source_file, e.page_number,
                       e.chunk_index, e.text, e.distance, f.vector_rank, f.lexical_rank,
                       CASE WHEN %s = 'hybrid' THEN f.rrf_score ELSE f.lexical_score END AS retrieval_score
                FROM fused f JOIN eligible e USING (chunk_id)
                WHERE %s = 'hybrid' OR f.lexical_rank IS NOT NULL
                ORDER BY retrieval_score DESC, e.chunk_id LIMIT %s
                """,
                (lexical_query(question), json.dumps(query_vector, allow_nan=False), provider, model,
                 document_ids, document_ids, CANDIDATE_LIMIT, CANDIDATE_LIMIT,
                 RRF_CONSTANT, RRF_CONSTANT, mode, mode, limit),
            )
            return cursor.fetchall()


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
