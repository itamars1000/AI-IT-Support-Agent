from datetime import datetime
from enum import Enum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TicketPriority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TicketStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    closed = "closed"


class TicketCreate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1, max_length=5000)
    priority: TicketPriority = TicketPriority.medium


class TicketUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    priority: TicketPriority | None = None
    status: TicketStatus | None = None

    @model_validator(mode="after")
    def require_changes(self) -> "TicketUpdate":
        if not self.model_fields_set:
            raise ValueError("Provide at least one field to update")
        for field in self.model_fields_set:
            if getattr(self, field) is None:
                raise ValueError(f"{field} cannot be null")
        return self


class TicketResponse(BaseModel):
    id: int
    title: str
    description: str
    priority: TicketPriority
    status: TicketStatus
    created_at: datetime
    escalated_at: datetime | None = None


class EscalationResult(BaseModel):
    ticket: TicketResponse
    escalated: bool
    reason: str


class DocumentPage(BaseModel):
    page_number: int = Field(ge=1)
    text: str


class ExtractedDocument(BaseModel):
    source_file: str
    pages: list[DocumentPage]


class DocumentResponse(BaseModel):
    id: int
    source_file: str
    page_count: int
    created_at: datetime
    chunk_count: int = Field(ge=0)


class DocumentChunk(BaseModel):
    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_number: int = Field(ge=1)


class SearchRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: int | None = Field(default=None, gt=0)


class SearchHit(BaseModel):
    chunk_id: int
    document_id: int
    source_file: str
    page_number: int
    chunk_index: int
    text: str
    distance: float


class RagRequest(SearchRequest):
    top_k: int = Field(default=5, ge=1, le=5)


class RagSource(SearchHit):
    reference: str


class RagResponse(BaseModel):
    answer: str
    sources: list[RagSource]


class ToolCallRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    allowed_actions: list[Literal["create_ticket"]] = Field(default_factory=list)
    idempotency_key: UUID | None = None

    @model_validator(mode="after")
    def require_idempotency_for_create(self) -> "ToolCallRequest":
        if "create_ticket" in self.allowed_actions and self.idempotency_key is None:
            raise ValueError("idempotency_key is required when create_ticket is allowed")
        return self


class ToolExecution(BaseModel):
    name: str
    input: dict
    result: TicketResponse


class ToolCallResponse(BaseModel):
    answer: str
    tool_call: ToolExecution | None = None


class AgentRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    message: str = Field(min_length=1, max_length=2000)
    allowed_actions: list[Literal["create_ticket", "escalate_ticket"]] = Field(default_factory=list)
    idempotency_key: UUID | None = None

    @model_validator(mode="after")
    def require_idempotency_for_create(self) -> "AgentRequest":
        if len(self.allowed_actions) != len(set(self.allowed_actions)):
            raise ValueError("allowed_actions cannot contain duplicates")
        if "create_ticket" in self.allowed_actions and self.idempotency_key is None:
            raise ValueError("idempotency_key is required when create_ticket is allowed")
        return self


class AgentStep(BaseModel):
    name: str
    input: dict
    result: dict


class AgentResponse(BaseModel):
    answer: str
    steps: list[AgentStep]
    completed: bool
