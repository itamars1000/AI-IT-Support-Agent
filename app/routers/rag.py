from fastapi import APIRouter

from app.schemas import RagRequest, RagResponse
from app.services.rag_service import answer_question


router = APIRouter(tags=["rag"])


@router.post("/rag", response_model=RagResponse)
def rag(request: RagRequest) -> RagResponse:
    return answer_question(request)
