from math import isfinite

import psycopg

from app.config import get_settings
from app.errors import DatabaseUnavailable
from app.repositories import embedding_repository
from app.services.voyage_client import embed_texts


EMBEDDING_PROVIDER = "voyage"
EMBEDDING_MODEL = "voyage-4"
EMBEDDING_DIMENSIONS = 1024
BATCH_SIZE = 32


def embed_pending_chunks() -> int:
    settings = get_settings()
    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else ""
    if not api_key:
        raise ValueError("Set VOYAGE_API_KEY in .env before generating embeddings")

    embedded_count = 0
    while True:
        try:
            chunks = embedding_repository.select_unembedded_chunks(
                EMBEDDING_PROVIDER, EMBEDDING_MODEL, BATCH_SIZE
            )
        except psycopg.OperationalError as error:
            raise DatabaseUnavailable() from error
        if not chunks:
            return embedded_count

        vectors = embed_texts(
            [text for _, text in chunks],
            api_key=api_key,
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
        )
        if len(vectors) != len(chunks):
            raise ValueError("Embedding provider returned the wrong number of vectors")
        for vector in vectors:
            if (
                not isinstance(vector, list)
                or len(vector) != EMBEDDING_DIMENSIONS
                or not all(type(value) in (int, float) and isfinite(value) for value in vector)
            ):
                raise ValueError("Embedding provider returned an invalid vector")

        try:
            embedding_repository.insert_embeddings(
                EMBEDDING_PROVIDER,
                EMBEDDING_MODEL,
                [(chunk_id, vector) for (chunk_id, _), vector in zip(chunks, vectors)],
            )
        except psycopg.OperationalError as error:
            raise DatabaseUnavailable() from error
        embedded_count += len(chunks)
