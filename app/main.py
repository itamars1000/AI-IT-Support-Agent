import psycopg
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.database import check_database_connection
from app.errors import AnswerProviderUnavailable, DatabaseUnavailable, EmbeddingProviderUnavailable, IdempotencyConflict, TicketNotFound
from app.routers.tickets import router as tickets_router
from app.routers.documents import router as documents_router
from app.routers.search import router as search_router
from app.routers.rag import router as rag_router
from app.routers.tool_calling import router as tool_calling_router
from app.routers.agent import router as agent_router
from app.services.anthropic_client import AnthropicRequestError
from app.services.voyage_client import VoyageRequestError
from app.observability import current_trace, new_trace


app = FastAPI(title="AI IT Support Agent", version="1.0.0")
app.include_router(tickets_router)
app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(tool_calling_router)
app.include_router(agent_router)


@app.middleware("http")
async def trace_requests(request: Request, call_next):
    trace = new_trace(request.url.path)
    token = current_trace.set(trace)
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        response.headers["X-Request-ID"] = trace.request_id
        return response
    finally:
        trace.finish(status_code)
        current_trace.reset(token)


@app.exception_handler(TicketNotFound)
async def ticket_not_found_handler(request: Request, error: TicketNotFound) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": "Ticket not found"})


@app.exception_handler(DatabaseUnavailable)
async def database_unavailable_handler(request: Request, error: DatabaseUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Database unavailable"})


@app.exception_handler(EmbeddingProviderUnavailable)
async def embedding_provider_unavailable_handler(request: Request, error: EmbeddingProviderUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Embedding provider is not configured"})


@app.exception_handler(VoyageRequestError)
async def voyage_request_error_handler(request: Request, error: VoyageRequestError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "Embedding provider request failed"})


@app.exception_handler(AnswerProviderUnavailable)
async def answer_provider_unavailable_handler(request: Request, error: AnswerProviderUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": "Answer provider is not configured"})


@app.exception_handler(AnthropicRequestError)
async def anthropic_request_error_handler(request: Request, error: AnthropicRequestError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": "Answer provider request failed"})


@app.exception_handler(IdempotencyConflict)
async def idempotency_conflict_handler(request: Request, error: IdempotencyConflict) -> JSONResponse:
    return JSONResponse(status_code=409, content={"detail": "Request ID was already used for different ticket details"})


class HealthResponse(BaseModel):
    status: str


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/health/db")
def database_health() -> HealthResponse:
    try:
        available = check_database_connection()
    except psycopg.Error:
        raise HTTPException(status_code=503, detail="Database unavailable") from None
    if not available:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return HealthResponse(status="ok")
