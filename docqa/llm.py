"""Thin wrapper around the Groq chat API."""
from collections.abc import Iterator
from functools import lru_cache

from groq import (
    APIConnectionError,
    APITimeoutError,
    Groq,
    InternalServerError,
    RateLimitError,
)

from .config import settings

# Failures where another model may still work: rate limits, timeouts, provider errors.
_TRY_NEXT_MODEL = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)


class LLMUnavailable(RuntimeError):
    """Every model that was tried is rate-limited or down."""


@lru_cache
def client() -> Groq:
    # The SDK retries rate limits (429) and server errors with exponential backoff.
    return Groq(api_key=settings.groq_api_key or None, max_retries=6, timeout=60)


def _kwargs(model: str, max_tokens: int) -> dict:
    kwargs = {"model": model, "temperature": 0, "max_completion_tokens": max_tokens}
    if "gpt-oss" in model:
        # Reasoning models: keep the hidden thinking short, answers here are extractive.
        kwargs["reasoning_effort"] = "low"
    return kwargs


def complete(messages: list[dict], model: str | None = None, max_tokens: int = 1024) -> str:
    """Patient call for batch work (evaluation, summaries): waits out rate limits, one model."""
    r = client().chat.completions.create(messages=messages, **_kwargs(model or settings.llm_model, max_tokens))
    return (r.choices[0].message.content or "").strip()


def stream_first_available(messages: list[dict], models: list[str], info: dict, max_tokens: int = 1024) -> Iterator[str]:
    """Interactive call: yield tokens from the first model in `models` that responds.
    A user is waiting, so a rate-limited model is skipped at once instead of retried.
    `info` is filled in with the model that answered and, at the end, its token counts."""
    impatient = client().with_options(max_retries=0, timeout=30)
    last_error: Exception | None = None
    for model in models:
        try:
            events = impatient.chat.completions.create(messages=messages, stream=True, **_kwargs(model, max_tokens))
        except _TRY_NEXT_MODEL as error:
            last_error = error
            continue
        info["model"] = model
        for event in events:
            if event.choices and event.choices[0].delta.content:
                yield event.choices[0].delta.content
            if getattr(event, "usage", None):  # Groq sends token counts on the last chunk
                info["prompt_tokens"] = event.usage.prompt_tokens
                info["completion_tokens"] = event.usage.completion_tokens
        return
    raise LLMUnavailable("no language model is available right now") from last_error
