class TicketNotFound(Exception):
    """The requested ticket does not exist."""


class DatabaseUnavailable(Exception):
    """A database operation could not connect or complete."""


class EmbeddingProviderUnavailable(Exception):
    """The embedding provider is not configured for search."""


class AnswerProviderUnavailable(Exception):
    """The answer provider is not configured for RAG."""


class IdempotencyConflict(Exception):
    """A request ID was already used for different ticket details."""
