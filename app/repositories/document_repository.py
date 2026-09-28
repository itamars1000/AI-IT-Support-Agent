from psycopg.rows import dict_row

from app.database import get_connection
from app.schemas import DocumentChunk, DocumentResponse, ExtractedDocument


def insert_document(
    document: ExtractedDocument, chunks: list[DocumentChunk]
) -> DocumentResponse:
    # One transaction saves the document, its pages and its chunks.
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                INSERT INTO documents (source_file, page_count)
                VALUES (%s, %s)
                RETURNING id, source_file, page_count, created_at
                """,
                (document.source_file, len(document.pages)),
            )
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("INSERT did not return a document")
            saved_document = DocumentResponse.model_validate({**row, "chunk_count": len(chunks)})
            cursor.executemany(
                """
                INSERT INTO document_pages (document_id, page_number, text)
                VALUES (%s, %s, %s)
                """,
                [
                    (saved_document.id, page.page_number, page.text)
                    for page in document.pages
                ],
            )
            cursor.executemany(
                """
                INSERT INTO document_chunks (document_id, text, chunk_index, page_number)
                VALUES (%s, %s, %s, %s)
                """,
                [
                    (saved_document.id, chunk.text, chunk.chunk_index, chunk.page_number)
                    for chunk in chunks
                ],
            )
    return saved_document
