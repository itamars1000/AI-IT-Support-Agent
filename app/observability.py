"""Request-scoped, content-free timing and usage logs."""

import json
import logging
from contextvars import ContextVar
from dataclasses import dataclass, field
from time import perf_counter
from uuid import uuid4


logger = logging.getLogger("app.observability")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


@dataclass
class RequestTrace:
    request_id: str
    endpoint: str
    started: float = field(default_factory=perf_counter)
    llm_model: str | None = None
    embedding_model: str | None = None
    tool_names: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0.0
    llm_latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    token_usage_available: bool = False
    failed: bool = False
    events: list[dict] = field(default_factory=list)

    def snapshot(self, status_code: int) -> dict:
        """Measured processing so far; no prompts, arguments, credentials or results."""
        return {
            "request_id": self.request_id,
            "endpoint": self.endpoint,
            "latency_ms": elapsed_ms(self.started),
            "llm_model": self.llm_model,
            "embedding_model": self.embedding_model,
            "number_of_tool_calls": len(self.tool_names),
            "tool_names": self.tool_names,
            "retrieval_latency_ms": round(self.retrieval_latency_ms, 3),
            "llm_latency_ms": round(self.llm_latency_ms, 3),
            "total_tokens": self.input_tokens + self.output_tokens if self.token_usage_available else None,
            "status": "failure" if self.failed or status_code >= 400 else "success",
            "status_code": status_code,
            "events": list(self.events),
        }

    def finish(self, status_code: int) -> None:
        summary = self.snapshot(status_code)
        summary.pop("events")
        logger.info(json.dumps({"event": "request", **summary}))


current_trace: ContextVar[RequestTrace | None] = ContextVar("request_trace", default=None)


def elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 3)


def record_tool(name: str, started: float, status: str) -> None:
    trace = current_trace.get()
    if trace is None:
        return
    trace.tool_names.append(name)
    duration = elapsed_ms(started)
    trace.events.append({"kind": "tool", "name": name, "latency_ms": duration, "status": status})
    logger.info(json.dumps({
        "event": "tool_call", "request_id": trace.request_id,
        "endpoint": trace.endpoint, "tool": name,
        "latency_ms": duration, "status": status,
    }))


def new_trace(endpoint: str) -> RequestTrace:
    return RequestTrace(request_id=uuid4().hex, endpoint=endpoint)
