from fastapi import APIRouter

from app.schemas import AgentRequest, AgentResponse, RequestActivity
from app.observability import current_trace
from app.services.agent_service import run_agent


router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    response = run_agent(request)
    trace = current_trace.get()
    if trace is not None:
        trace.failed |= not response.completed
        response.trace = RequestActivity.model_validate(trace.snapshot(200))
    return response
