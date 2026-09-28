from pathlib import PurePosixPath
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.schemas import DocumentResponse
from app.services.document_service import ingest_pdf_stream


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("", response_model=DocumentResponse, status_code=201)
def upload_document(file: Annotated[UploadFile, File()]) -> DocumentResponse:
    filename = PurePosixPath((file.filename or "").replace("\\", "/")).name
    if not filename or not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Upload a file with a .pdf filename")
    try:
        return ingest_pdf_stream(file.file, filename)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
