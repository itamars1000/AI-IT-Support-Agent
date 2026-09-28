from fastapi import APIRouter

from app.schemas import TicketCreate, TicketResponse, TicketUpdate
from app.services import ticket_service


router = APIRouter(prefix="/tickets", tags=["tickets"])


@router.post("", response_model=TicketResponse, status_code=201)
def create_ticket(payload: TicketCreate) -> TicketResponse:
    return ticket_service.create_ticket(payload)


@router.get("", response_model=list[TicketResponse])
def list_tickets() -> list[TicketResponse]:
    return ticket_service.list_tickets()


@router.get("/{ticket_id}", response_model=TicketResponse)
def get_ticket(ticket_id: int) -> TicketResponse:
    return ticket_service.get_ticket(ticket_id)


@router.patch("/{ticket_id}", response_model=TicketResponse)
def update_ticket(ticket_id: int, payload: TicketUpdate) -> TicketResponse:
    return ticket_service.update_ticket(ticket_id, payload)
