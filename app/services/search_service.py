"""Embed a question and retrieve vector or fused vector/text candidates."""

from math import isfinite
from time import perf_counter

import psycopg

from app.config import get_settings
from app.errors import DatabaseUnavailable, EmbeddingProviderUnavailable
from app.repositories import embedding_repository
from app.schemas import SearchHit, SearchRequest
from app.services.embedding_service import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, EMBEDDING_PROVIDER
from app.services.voyage_client import VoyageRequestError, embed_texts
from app.observability import current_trace, elapsed_ms


def search_documents(request: SearchRequest) -> list[SearchHit]:
    trace = current_trace.get()
    if trace is not None:
        trace.embedding_model = EMBEDDING_MODEL
    started = perf_counter()
    try:
        return _search_documents(request)
    finally:
        if trace is not None:
            trace.retrieval_latency_ms += elapsed_ms(started)


def _search_documents(request: SearchRequest) -> list[SearchHit]:
    settings = get_settings()
    api_key = settings.voyage_api_key.get_secret_value() if settings.voyage_api_key else ""
    if not api_key:
        raise EmbeddingProviderUnavailable()

    vectors = embed_texts(
        [request.question],
        api_key=api_key,
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIMENSIONS,
        input_type="query",
    )
    if len(vectors) != 1 or not isinstance(vectors[0], list):
        raise VoyageRequestError("Voyage API returned an invalid vector")
    vector = vectors[0]
    if len(vector) != EMBEDDING_DIMENSIONS or not all(
        type(value) in (int, float) and isfinite(value) for value in vector
    ):
        raise VoyageRequestError("Voyage API returned an invalid vector")

    try:
        mode = request.retrieval_mode or getattr(settings, "retrieval_mode", "vector")
        if mode == "hybrid":
            rows = embedding_repository.search_ranked_chunks(
                request.question, vector, EMBEDDING_PROVIDER, EMBEDDING_MODEL,
                request.top_k, "hybrid", [request.document_id] if request.document_id else None,
            )
        else:
            rows = embedding_repository.search_chunks(
                vector, EMBEDDING_PROVIDER, EMBEDDING_MODEL, request.top_k, request.document_id
            )
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error
    return [SearchHit.model_validate(row) for row in rows]
