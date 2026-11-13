"""In-process answer cache with a time limit. Keys include the index version, so adding,
updating or deleting a document makes every older cached answer unreachable."""
import re
import time

from . import store
from .config import settings

_cache: dict[tuple, tuple[float, dict]] = {}
MAX_ENTRIES = 5000


def _key(question: str, mode: str) -> tuple:
    normalised = re.sub(r"[^\w\s]", "", question.lower())
    return (" ".join(normalised.split()), mode, store.index_version())


def get(question: str, mode: str) -> dict | None:
    entry = _cache.get(_key(question, mode))
    if entry and time.time() - entry[0] < settings.cache_ttl_seconds:
        return entry[1]
    return None


def put(question: str, mode: str, result: dict) -> None:
    if len(_cache) >= MAX_ENTRIES:
        _cache.pop(next(iter(_cache)))  # drop the oldest entry
    _cache[_key(question, mode)] = (time.time(), result)
