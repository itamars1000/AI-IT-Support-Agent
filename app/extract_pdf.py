"""Run with: python -m app.extract_pdf PATH [--output OUTPUT.json]."""

import argparse
import sys
from pathlib import Path

from app.services.document_service import extract_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PDF text by page")
    parser.add_argument("path", type=Path, help="Path to the source PDF")
    parser.add_argument("--output", type=Path, help="Write UTF-8 JSON to this file")
    args = parser.parse_args()

    if args.output is not None and args.output.resolve() == args.path.resolve():
        parser.error("Output must be different from the source PDF")

    try:
        document = extract_pdf(args.path)
        result = document.model_dump_json(indent=2)
        if args.output is not None:
            args.output.write_text(result + "\n", encoding="utf-8")
        else:
            sys.stdout.reconfigure(encoding="utf-8")
            print(result)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    for page in document.pages:
        if not page.text:
            print(
                f"Warning: page {page.page_number} has no extracted text; "
                "it may be blank or require OCR.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
