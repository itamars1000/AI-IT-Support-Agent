from fastapi import APIRouter

from app.schemas import AgentRequest, AgentResponse
from app.services.agent_service import run_agent


router = APIRouter(tags=["agent"])


@router.post("/agent", response_model=AgentResponse)
def agent(request: AgentRequest) -> AgentResponse:
    return run_agent(request)
