"""Small HTTP adapter for Voyage's embeddings API."""

import json
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VoyageRequestError(RuntimeError):
    """The embedding request failed without exposing credentials or document text."""


def embed_texts(
    texts: list[str],
    api_key: str,
    model: str,
    dimensions: int,
    input_type: Literal["document", "query"] = "document",
) -> list[list[float]]:
    payload = json.dumps(
        {
            "input": texts,
            "model": model,
            "input_type": input_type,
            "output_dimension": dimensions,
            "truncation": False,
        }
    ).encode("utf-8")
    request = Request(
        "https://api.voyageai.com/v1/embeddings",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
        entries = data["data"]
        if sorted(entry["index"] for entry in entries) != list(range(len(texts))):
            raise VoyageRequestError("Voyage returned invalid embedding indices")
        return [entry["embedding"] for entry in sorted(entries, key=lambda item: item["index"])]
    except HTTPError as error:
        raise VoyageRequestError(f"Voyage API returned HTTP {error.code}") from None
    except URLError:
        raise VoyageRequestError("Could not connect to Voyage API") from None
    except (KeyError, TypeError, ValueError):
        raise VoyageRequestError("Voyage API returned an invalid response") from None
