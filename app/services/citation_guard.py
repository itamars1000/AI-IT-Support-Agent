"""Check that numbered citations refer to sources actually returned by search."""

import re

from app.services.anthropic_client import AnthropicRequestError


def validate_citations(answer: str, source_count: int) -> None:
    referenced = {int(number) for number in re.findall(r"\[(\d+)\]", answer)}
    if any(number < 1 or number > source_count for number in referenced):
        raise AnthropicRequestError("The answer cited a source that was not retrieved")
