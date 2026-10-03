-- Stored generated values cover existing rows and future ingestion/text changes.
ALTER TABLE document_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', text)) STORED;

CREATE INDEX IF NOT EXISTS document_chunks_search_vector_idx
    ON document_chunks USING GIN (search_vector);
