"""Run with: python -m app.embed_chunks."""

import argparse

from app.errors import DatabaseUnavailable
from app.services.embedding_service import EMBEDDING_MODEL, embed_pending_chunks
from app.services.voyage_client import VoyageRequestError


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for document chunks")
    try:
        embedded_count = embed_pending_chunks()
    except DatabaseUnavailable:
        parser.error("Database unavailable")
    except ValueError as error:
        parser.error(str(error))
    except VoyageRequestError as error:
        parser.error(str(error))
    print(f"Embedded {embedded_count} chunks with {EMBEDDING_MODEL}.")


if __name__ == "__main__":
    main()
