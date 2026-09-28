ALTER TABLE tickets
ADD COLUMN IF NOT EXISTS creation_request_id UUID;

CREATE UNIQUE INDEX IF NOT EXISTS tickets_creation_request_id_unique
ON tickets (creation_request_id);
