import psycopg
from uuid import UUID

from app.errors import DatabaseUnavailable, TicketNotFound
from app.repositories import ticket_repository
from app.schemas import EscalationResult, TicketCreate, TicketResponse, TicketUpdate


def create_ticket(payload: TicketCreate, idempotency_key: UUID | None = None) -> TicketResponse:
    try:
        return ticket_repository.insert_ticket(payload, idempotency_key)
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error


def list_tickets() -> list[TicketResponse]:
    try:
        return ticket_repository.select_tickets()
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error


def get_ticket(ticket_id: int) -> TicketResponse:
    try:
        ticket = ticket_repository.select_ticket(ticket_id)
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error
    if ticket is None:
        raise TicketNotFound()
    return ticket


def update_ticket(ticket_id: int, payload: TicketUpdate) -> TicketResponse:
    try:
        ticket = ticket_repository.modify_ticket(ticket_id, payload)
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error
    if ticket is None:
        raise TicketNotFound()
    return ticket


def escalate_ticket(ticket_id: int) -> EscalationResult:
    try:
        result = ticket_repository.escalate_ticket_if_overdue(ticket_id)
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error
    if result is None:
        raise TicketNotFound()
    return result
