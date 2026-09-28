import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import SecretStr

from app.database import get_connection


class DatabaseAddressTests(unittest.TestCase):
    def test_local_connection_keeps_address_from_database_url(self):
        settings = SimpleNamespace(
            database_url=SecretStr("postgresql://support:test@127.0.0.1:5433/support_agent"),
            database_host=None,
            database_port=None,
        )
        with patch("app.database.get_settings", return_value=settings):
            with patch("app.database.psycopg.connect") as connect:
                get_connection()
        self.assertNotIn("host", connect.call_args.kwargs)
        self.assertNotIn("port", connect.call_args.kwargs)

    def test_container_connection_overrides_only_address(self):
        settings = SimpleNamespace(
            database_url=SecretStr("postgresql://support:test@127.0.0.1:5433/support_agent"),
            database_host="db",
            database_port=5432,
        )
        with patch("app.database.get_settings", return_value=settings):
            with patch("app.database.psycopg.connect") as connect:
                get_connection()
        self.assertEqual(connect.call_args.kwargs["host"], "db")
        self.assertEqual(connect.call_args.kwargs["port"], 5432)
        self.assertEqual(connect.call_args.args[0], settings.database_url.get_secret_value())


if __name__ == "__main__":
    unittest.main()
