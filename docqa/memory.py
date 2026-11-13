"""Chat memory. A session keeps its recent turns word for word. When the stored turns pass
the token limit, the oldest ones are folded into a running summary by the small model, so the
prompt stays a bounded size however long the conversation gets."""
from . import llm, prompts, store
from .config import settings


def approx_tokens(text: str) -> int:
    # About 4 characters per token for English. Exact counts are not needed for a soft limit.
    return len(text) // 4


def _turn_tokens(turns: list[dict]) -> int:
    return sum(approx_tokens(t["content"]) for t in turns)


def _render(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def standalone_question(question: str, summary: str, turns: list[dict]) -> str:
    """Follow-ups like "and how was it trained?" retrieve badly on their own. Rewrite them
    into a self-contained query before searching."""
    if not turns and not summary:
        return question
    history = (f"Summary: {summary}\n" if summary else "") + _render(turns[-6:])
    rewritten = llm.complete(
        [{"role": "system", "content": prompts.REWRITE},
         {"role": "user", "content": f"Conversation:\n{history}\n\nLatest question: {question}"}],
        model=settings.llm_model_small, max_tokens=200,
    )
    return rewritten or question


def compact(summary: str, turns: list[dict]) -> tuple[str, list[dict]]:
    """Fold old turns into the summary once the token limit is passed."""
    keep = settings.memory_keep_turns * 2  # a turn is a user message plus an assistant message
    if _turn_tokens(turns) <= settings.memory_token_limit or len(turns) <= keep:
        return summary, turns
    old, recent = turns[:-keep], turns[-keep:]
    text = (f"Earlier summary: {summary}\n\n" if summary else "") + _render(old)
    new_summary = llm.complete(
        [{"role": "system", "content": prompts.SUMMARISE}, {"role": "user", "content": text}],
        model=settings.llm_model_small, max_tokens=300,
    )
    return new_summary or summary, recent


def remember(session_id: str, question: str, answer: str) -> None:
    summary, turns = store.load_session(session_id)
    turns += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    summary, turns = compact(summary, turns)
    store.save_session(session_id, summary, turns)
