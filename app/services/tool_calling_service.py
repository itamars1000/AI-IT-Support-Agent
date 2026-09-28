"""One Claude tool call at most; the agent loop is a later step."""

import json
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.config import get_settings
from app.errors import AnswerProviderUnavailable
from app.schemas import TicketCreate, ToolCallRequest, ToolCallResponse, ToolExecution
from app.services import ticket_service
from app.services.anthropic_client import AnthropicRequestError, create_message, extract_final_text
from app.observability import record_tool


SYSTEM_PROMPT = (
    "You are an IT support assistant. Use create_ticket only when the user asks to open a ticket. "
    "Use get_ticket only when the user asks to inspect a ticket by ID. "
    "If required details are missing, ask a short question instead of inventing them. "
    "Never claim a ticket was created or retrieved unless a tool result confirms it. "
    "Do not request a second tool call in this turn. Respond in the user's language."
)

TOOLS = [
    {
        "name": "create_ticket",
        "description": "Create one new IT support ticket when the user explicitly asks to open or create one.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short summary, 1 to 200 characters"},
                "description": {"type": "string", "description": "Problem details supplied by the user, 1 to 5000 characters"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "description": "Use medium unless the user specifies another priority"},
            },
            "required": ["title", "description"],
            "additionalProperties": False,
        },
    },
    {
        "name": "get_ticket",
        "description": "Retrieve an existing IT support ticket by its numeric ID.",
        "input_schema": {
            "type": "object",
            "properties": {"ticket_id": {"type": "integer", "minimum": 1}},
            "required": ["ticket_id"],
            "additionalProperties": False,
        },
    },
]


class GetTicketInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    ticket_id: int = Field(gt=0)


def run_tool_call(request: ToolCallRequest) -> ToolCallResponse:
    settings = get_settings()
    api_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
    if not api_key:
        raise AnswerProviderUnavailable()

    offered_tools = [
        tool for tool in TOOLS
        if tool["name"] != "create_ticket" or "create_ticket" in request.allowed_actions
    ]
    offered_names = {tool["name"] for tool in offered_tools}
    messages = [{"role": "user", "content": request.message}]
    first = create_message(
        SYSTEM_PROMPT, messages, api_key, settings.anthropic_model,
        tools=offered_tools, tool_choice={"type": "auto", "disable_parallel_tool_use": True},
    )
    blocks = first["content"]
    calls = [block for block in blocks if isinstance(block, dict) and block.get("type") == "tool_use"]
    if first.get("stop_reason") == "end_turn" and not calls:
        return ToolCallResponse(answer=extract_final_text(first))
    if first.get("stop_reason") != "tool_use" or len(calls) != 1:
        raise AnthropicRequestError("Anthropic returned an unsupported tool response")

    call = calls[0]
    if (
        not isinstance(call.get("id"), str) or not call["id"]
        or call.get("name") not in offered_names
        or not isinstance(call.get("input"), dict)
    ):
        raise AnthropicRequestError("Anthropic returned an invalid tool call")
    tool_started = perf_counter()
    try:
        if call.get("name") == "create_ticket":
            payload = TicketCreate.model_validate(call["input"])
            tool_input = payload.model_dump(mode="json")
            ticket = ticket_service.create_ticket(payload, request.idempotency_key)
        elif call.get("name") == "get_ticket":
            payload = GetTicketInput.model_validate(call["input"])
            tool_input = payload.model_dump()
            ticket = ticket_service.get_ticket(payload.ticket_id)
        else:
            raise AnthropicRequestError("Anthropic requested an unknown tool")
    except ValidationError:
        record_tool(call["name"], tool_started, "failure")
        raise AnthropicRequestError("Anthropic returned invalid tool arguments") from None
    except Exception:
        record_tool(call["name"], tool_started, "failure")
        raise

    record_tool(call["name"], tool_started, "success")

    execution = ToolExecution(name=call["name"], input=tool_input, result=ticket)
    tool_result = {
        "type": "tool_result",
        "tool_use_id": call["id"],
        "content": json.dumps(ticket.model_dump(mode="json"), ensure_ascii=False),
    }
    messages.extend([
        {"role": "assistant", "content": blocks},
        {"role": "user", "content": [tool_result]},
    ])
    try:
        final = create_message(
            SYSTEM_PROMPT, messages, api_key, settings.anthropic_model,
            tools=offered_tools, tool_choice={"type": "none"},
        )
        answer = extract_final_text(final)
    except AnthropicRequestError:
        # The database operation already succeeded; keep its result visible to the caller.
        answer = f"Ticket #{ticket.id}: {ticket.status.value}."
    return ToolCallResponse(answer=answer, tool_call=execution)
