import psycopg

from app.config import get_settings


def get_connection() -> psycopg.Connection:
    settings = get_settings()
    address_override = {}
    if settings.database_host:
        address_override["host"] = settings.database_host
    if settings.database_port is not None:
        address_override["port"] = settings.database_port
    return psycopg.connect(
        settings.database_url.get_secret_value(),
        connect_timeout=3,
        options="-c statement_timeout=3000",
        **address_override,
    )


def check_database_connection() -> bool:
    with get_connection() as connection:
        result = connection.execute("SELECT 1").fetchone()
        return result == (1,)
