from app.schemas import DocumentChunk, ExtractedDocument


def chunk_document(
    document: ExtractedDocument, chunk_size: int = 1000, overlap: int = 200
) -> list[DocumentChunk]:
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= overlap < chunk_size")

    chunks = []
    for page in document.pages:
        for start in range(0, len(page.text), chunk_size - overlap):
            end = min(start + chunk_size, len(page.text))
            text = page.text[start:end]
            if text.strip():
                chunks.append(
                    DocumentChunk(
                        text=text,
                        chunk_index=len(chunks),
                        page_number=page.page_number,
                    )
                )
            if end == len(page.text):
                break
    return chunks
