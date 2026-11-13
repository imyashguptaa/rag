"""The question-answering pipeline. One generator, `answer_events`, serves both the streaming
API and the plain `answer()` call, so there is a single code path to test and measure."""
import json
import logging
import re
import time
import uuid
from collections.abc import Iterator

from . import cache, llm, memory, prompts, store
from .config import settings
from .retrieve import Hit, retrieve

# One JSON object per answered request, so a bad answer can be traced from its request id.
request_log = logging.getLogger("docqa.request")

_HARD_WORDS = re.compile(r"\b(why|how(?! many| much| long)|compare|comparison|difference|differ|versus|vs|explain|trade-?offs?|advantages?|limitations?|summari[sz]e)\b", re.I)


def pick_model(question: str, confidence: float) -> str:
    """Send easy queries to the small model: short, a single fact, and retrieval is confident.
    Anything asking for reasoning or comparison goes to the large model."""
    easy = len(question.split()) <= 14 and confidence >= 0.7 and not _HARD_WORDS.search(question)
    return settings.llm_model_small if settings.route_easy_queries and easy else settings.llm_model


def plain_brackets(text: str) -> str:
    """Some models write citations as full-width brackets. Make them plain [n], so the
    citation parser, the page and the evaluation all see the same thing."""
    return text.replace("【", "[").replace("】", "]")


def cited_sources(answer: str, hits: list[Hit]) -> list[dict]:
    """The sources the answer actually cited, in citation order. Numbers that do not match
    a retrieved excerpt are ignored rather than trusted."""
    seen, sources = set(), []
    for n in re.findall(r"\[(\d+)\]", answer):
        i = int(n)
        if 1 <= i <= len(hits) and i not in seen:
            seen.add(i)
            h = hits[i - 1]
            sources.append({"id": i, "source": h.source, "page": h.page, "date": h.doc_date, "text": h.text})
    return sources


def _record(request_id: str, session_id: str | None, question: str, result: dict, mode: str, cached: bool,
            total_ms: float, retrieve_ms: float = 0.0, first_token_ms: float | None = None, usage: dict | None = None) -> None:
    """Write the request to the database (latency report, feedback) and to the structured log."""
    usage = usage or {}
    store.log_request(request_id, question, result["answer"], total_ms, retrieve_ms, first_token_ms, result["model"],
                      cached, result["refused"], mode, usage.get("prompt_tokens"), usage.get("completion_tokens"))
    request_log.info(json.dumps({
        "request_id": request_id, "session_id": session_id, "question": question,
        "search_query": result["search_query"], "mode": mode, "confidence": result["confidence"],
        "retrieved": [{"source": r["source"], "page": r["page"], "score": round(r["score"], 4)} for r in result["retrieved"]],
        "cited": [s["id"] for s in result["sources"]], "model": result["model"], "cached": cached,
        "refused": result["refused"], "total_ms": round(total_ms, 1), "retrieve_ms": round(retrieve_ms, 1),
        "first_token_ms": round(first_token_ms, 1) if first_token_ms else None,
        "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"),
    }))


def answer_events(question: str, session_id: str | None = None, mode: str | None = None,
                  request_id: str | None = None) -> Iterator[dict]:
    """Yield {"type": "token", "text": ...} while generating, then one {"type": "done", ...}."""
    started = time.perf_counter()
    request_id = request_id or uuid.uuid4().hex
    mode = mode or settings.retrieval_mode
    summary, turns = store.load_session(session_id) if session_id else ("", [])
    has_history = bool(turns or summary)
    # Only context-free questions are cacheable: with history, the same words can mean something else.
    cacheable = settings.cache_enabled and not has_history

    if cacheable and (hit := cache.get(question, mode)):
        yield {"type": "token", "text": hit["answer"]}
        total_ms = (time.perf_counter() - started) * 1000
        _record(request_id, session_id, question, hit, mode, True, total_ms, first_token_ms=total_ms)
        if session_id:
            memory.remember(session_id, question, hit["answer"])
        yield {**hit, "type": "done", "request_id": request_id, "cached": True, "timings": {"total_ms": round(total_ms, 1)}}
        return

    search_query = memory.standalone_question(question, summary, turns)
    t0 = time.perf_counter()
    hits, confidence = retrieve(search_query, mode)
    retrieve_ms = (time.perf_counter() - t0) * 1000

    first_token_ms = None
    info: dict = {"model": "none"}
    if confidence < settings.min_score:
        # Retrieval found nothing close enough: refuse without spending an LLM call.
        text, refused = prompts.REFUSAL, True
        yield {"type": "token", "text": text}
    else:
        model = pick_model(search_query, confidence)
        # If the chosen model is rate-limited or down, the other one answers instead.
        fallback = settings.llm_model_small if model == settings.llm_model else settings.llm_model
        parts = []
        messages = prompts.build_messages(question, hits, summary, turns)
        for token in llm.stream_first_available(messages, [model, fallback], info):
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            token = plain_brackets(token)
            parts.append(token)
            yield {"type": "token", "text": token}
        text = "".join(parts).strip()
        refused = text.startswith("I do not know")

    total_ms = (time.perf_counter() - started) * 1000
    result = {
        "answer": text,
        "sources": [] if refused else cited_sources(text, hits),
        "refused": refused,
        "model": info["model"],
        "confidence": round(confidence, 3),
        "search_query": search_query,
        "retrieved": [h.public() for h in hits],
    }
    _record(request_id, session_id, question, result, mode, False, total_ms, retrieve_ms, first_token_ms, info)
    if cacheable:
        cache.put(question, mode, result)
    if session_id:
        memory.remember(session_id, question, text)
    yield {
        **result, "type": "done", "request_id": request_id, "cached": False,
        "timings": {"total_ms": round(total_ms, 1), "retrieve_ms": round(retrieve_ms, 1),
                    "first_token_ms": round(first_token_ms, 1) if first_token_ms else None},
        "tokens": {"prompt": info.get("prompt_tokens"), "completion": info.get("completion_tokens")},
    }


def answer(question: str, session_id: str | None = None, mode: str | None = None, request_id: str | None = None) -> dict:
    for event in answer_events(question, session_id, mode, request_id):
        if event["type"] == "done":
            return event
    raise RuntimeError("pipeline ended without a result")


if __name__ == "__main__":
    import sys

    result = answer(" ".join(sys.argv[1:]))
    result.pop("retrieved")
    print(json.dumps(result, indent=2))
