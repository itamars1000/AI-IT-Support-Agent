from fastapi import APIRouter

from app.schemas import SearchHit, SearchRequest
from app.services.search_service import search_documents


router = APIRouter(tags=["search"])


@router.post("/search", response_model=list[SearchHit])
def search(request: SearchRequest) -> list[SearchHit]:
    return search_documents(request)
