"""Run with: python -m app.ingest_pdf PATH."""

import argparse
import sys
from pathlib import Path

from app.errors import DatabaseUnavailable
from app.services.document_service import ingest_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a PDF and save its pages to PostgreSQL")
    parser.add_argument("path", type=Path, help="Path to the source PDF")
    args = parser.parse_args()
    try:
        document = ingest_pdf(args.path)
    except DatabaseUnavailable:
        parser.error("Database unavailable")
    except (OSError, ValueError) as error:
        parser.error(str(error))
    sys.stdout.reconfigure(encoding="utf-8")
    print(document.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
