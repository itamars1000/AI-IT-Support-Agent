"""Small HTTP adapter for Anthropic's Messages API."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.observability import current_trace, elapsed_ms
from time import perf_counter


class AnthropicRequestError(RuntimeError):
    """The answer request failed without exposing credentials or source text."""


def create_message(
    system: str,
    messages: list[dict],
    api_key: str,
    model: str,
    tools: list[dict] | None = None,
    tool_choice: dict | None = None,
) -> dict:
    trace = current_trace.get()
    if trace is not None:
        trace.llm_model = model
    started = perf_counter()
    body = {
        "model": model,
        "max_tokens": 800,
        "system": system,
        "messages": messages,
    }
    if tools is not None:
        body["tools"] = tools
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    payload = json.dumps(
        body,
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    call_status = "failure"
    try:
        with urlopen(request, timeout=60) as response:
            data = json.load(response)
        if not isinstance(data, dict) or not isinstance(data.get("content"), list):
            raise AnthropicRequestError("Anthropic API returned an invalid response")
        if trace is not None:
            usage = data.get("usage")
            if isinstance(usage, dict):
                input_tokens = usage.get("input_tokens")
                output_tokens = usage.get("output_tokens")
                if type(input_tokens) is int and type(output_tokens) is int:
                    trace.input_tokens += input_tokens
                    trace.output_tokens += output_tokens
                    trace.token_usage_available = True
        call_status = "success"
        return data
    except HTTPError as error:
        raise AnthropicRequestError(f"Anthropic API returned HTTP {error.code}") from None
    except URLError:
        raise AnthropicRequestError("Could not connect to Anthropic API") from None
    except (TypeError, ValueError):
        raise AnthropicRequestError("Anthropic API returned an invalid response") from None
    finally:
        if trace is not None:
            duration = elapsed_ms(started)
            trace.llm_latency_ms += duration
            trace.events.append({"kind": "llm", "name": model, "latency_ms": duration, "status": call_status})


def extract_final_text(data: dict) -> str:
    try:
        if data["stop_reason"] != "end_turn":
            raise AnthropicRequestError("Anthropic did not finish the answer")
        blocks = data["content"]
        if not isinstance(blocks, list) or not blocks or any(
            block.get("type") != "text" or not isinstance(block.get("text"), str)
            for block in blocks
        ):
            raise AnthropicRequestError("Anthropic returned an invalid answer")
        answer = "\n".join(block["text"] for block in blocks).strip()
        if not answer:
            raise AnthropicRequestError("Anthropic returned an empty answer")
        return answer
    except (KeyError, TypeError, ValueError, AttributeError):
        raise AnthropicRequestError("Anthropic API returned an invalid response") from None


def generate_text(system: str, user: str, api_key: str, model: str) -> str:
    data = create_message(system, [{"role": "user", "content": user}], api_key, model)
    return extract_final_text(data)
