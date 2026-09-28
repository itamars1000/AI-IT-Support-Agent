CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    text TEXT NOT NULL CHECK (char_length(btrim(text)) > 0),
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    page_number INTEGER NOT NULL CHECK (page_number > 0),
    UNIQUE (document_id, chunk_index),
    FOREIGN KEY (document_id, page_number)
        REFERENCES document_pages(document_id, page_number) ON DELETE CASCADE
);
