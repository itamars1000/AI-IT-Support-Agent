from fastapi import APIRouter

from app.schemas import ToolCallRequest, ToolCallResponse
from app.services.tool_calling_service import run_tool_call


router = APIRouter(tags=["tool calling"])


@router.post("/tool-call", response_model=ToolCallResponse)
def tool_call(request: ToolCallRequest) -> ToolCallResponse:
    return run_tool_call(request)
