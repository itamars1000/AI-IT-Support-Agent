"""Extract PDF text and orchestrate storage of the document and its pages."""

from pathlib import Path
from typing import BinaryIO

import psycopg
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from app.errors import DatabaseUnavailable
from app.repositories import document_repository
from app.schemas import DocumentPage, DocumentResponse, ExtractedDocument
from app.services.chunking_service import chunk_document


def extract_pdf(path: str | Path) -> ExtractedDocument:
    source = Path(path)
    with source.open("rb") as stream:
        return extract_pdf_stream(stream, source.name)


def extract_pdf_stream(stream: BinaryIO, source_file: str) -> ExtractedDocument:
    try:
        reader = PdfReader(stream)
        if reader.is_encrypted:
            raise ValueError("Encrypted PDFs are not supported in this step")

        pages = []
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            pages.append(DocumentPage(page_number=page_number, text=text))
    except PyPdfError as error:
        raise ValueError("Could not read or extract text from this PDF") from error

    return ExtractedDocument(source_file=source_file, pages=pages)


def ingest_pdf(path: str | Path) -> DocumentResponse:
    return save_extracted_document(extract_pdf(path))


def ingest_pdf_stream(stream: BinaryIO, source_file: str) -> DocumentResponse:
    return save_extracted_document(extract_pdf_stream(stream, source_file))


def save_extracted_document(document: ExtractedDocument) -> DocumentResponse:
    if not document.pages or not any(page.text.strip() for page in document.pages):
        raise ValueError("PDF contains no extractable text; it may require OCR")
    try:
        return document_repository.insert_document(document, chunk_document(document))
    except psycopg.OperationalError as error:
        raise DatabaseUnavailable() from error
