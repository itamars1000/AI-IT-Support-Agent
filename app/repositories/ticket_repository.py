from psycopg.rows import dict_row
from uuid import UUID

from app.database import get_connection
from app.errors import IdempotencyConflict
from app.schemas import EscalationResult, TicketCreate, TicketResponse, TicketUpdate


def insert_ticket(payload: TicketCreate, idempotency_key: UUID | None = None) -> TicketResponse:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                INSERT INTO tickets (title, description, priority, creation_request_id)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (creation_request_id) DO NOTHING
                RETURNING id, title, description, priority, status, created_at, escalated_at
                """,
                (payload.title, payload.description, payload.priority.value, idempotency_key),
            )
            row = cursor.fetchone()
            if row is None:
                if idempotency_key is None:
                    raise RuntimeError("INSERT did not return a ticket")
                cursor.execute(
                    """
                    SELECT id, title, description, priority, status, created_at, escalated_at
                    FROM tickets
                    WHERE creation_request_id = %s
                    """,
                    (idempotency_key,),
                )
                row = cursor.fetchone()
                if row is None:
                    raise RuntimeError("Idempotent ticket insert returned no ticket")
                if (row["title"], row["description"], row["priority"]) != (
                    payload.title, payload.description, payload.priority.value
                ):
                    raise IdempotencyConflict()
            ticket = TicketResponse.model_validate(row)
    return ticket


def select_tickets() -> list[TicketResponse]:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, title, description, priority, status, created_at, escalated_at
                FROM tickets
                ORDER BY id
                """
            )
            return [TicketResponse.model_validate(row) for row in cursor.fetchall()]


def select_ticket(ticket_id: int) -> TicketResponse | None:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                SELECT id, title, description, priority, status, created_at, escalated_at
                FROM tickets
                WHERE id = %s
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return TicketResponse.model_validate(row)


def modify_ticket(ticket_id: int, payload: TicketUpdate) -> TicketResponse | None:
    changes = payload.model_dump(exclude_unset=True, mode="json")
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE tickets
                SET priority = COALESCE(%s, priority),
                    status = COALESCE(%s, status)
                WHERE id = %s
                RETURNING id, title, description, priority, status, created_at, escalated_at
                """,
                (changes.get("priority"), changes.get("status"), ticket_id),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return TicketResponse.model_validate(row)


def escalate_ticket_if_overdue(ticket_id: int) -> EscalationResult | None:
    with get_connection() as connection:
        with connection.cursor(row_factory=dict_row) as cursor:
            cursor.execute(
                """
                UPDATE tickets
                SET priority = 'high', escalated_at = CURRENT_TIMESTAMP
                WHERE id = %s
                  AND status = 'open'
                  AND created_at < CURRENT_TIMESTAMP - INTERVAL '7 days'
                  AND escalated_at IS NULL
                RETURNING id, title, description, priority, status, created_at, escalated_at
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                return EscalationResult(
                    ticket=TicketResponse.model_validate(row), escalated=True, reason="escalated"
                )

            cursor.execute(
                """
                SELECT id, title, description, priority, status, created_at, escalated_at,
                       created_at < CURRENT_TIMESTAMP - INTERVAL '7 days' AS overdue
                FROM tickets
                WHERE id = %s
                """,
                (ticket_id,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            reason = (
                "already_escalated" if row["escalated_at"] is not None else
                "not_open" if row["status"] != "open" else
                "too_recent" if not row["overdue"] else
                "not_escalated"
            )
            return EscalationResult(
                ticket=TicketResponse.model_validate(row), escalated=False, reason=reason
            )
