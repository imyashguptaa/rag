import pymupdf
import pytest

from docqa import cache, index, store
from docqa.config import settings


def make_pdf(path, pages: list[str]) -> None:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 800), text, fontsize=10)
    doc.save(path)
    doc.close()


@pytest.fixture
def tmp_index(tmp_path, monkeypatch):
    """A fresh, empty index, database and PDF folder for one test."""
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    monkeypatch.setattr(settings, "pdf_dir", pdf_dir)
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "qdrant")
    monkeypatch.setattr(settings, "qdrant_url", "")
    monkeypatch.setattr(settings, "db_path", tmp_path / "app.db")
    index.client.cache_clear()
    store._conn = None
    cache._cache.clear()
    yield pdf_dir
    index.client().close()
    index.client.cache_clear()
    store._conn.close()
    store._conn = None


class FakeLLM:
    """Stands in for Groq so pipeline tests are free, fast and deterministic."""

    def __init__(self, reply: str = "The answer is 42 [1]."):
        self.reply = reply
        self.calls: list[list[dict]] = []

    def stream_first_available(self, messages, models, info, max_tokens=1024):
        self.calls.append(messages)
        info.update(model=models[0], prompt_tokens=100, completion_tokens=7)
        for token in self.reply.split(" "):
            yield token + " "

    def complete(self, messages, model=None, max_tokens=1024):
        self.calls.append(messages)
        return self.reply


@pytest.fixture
def fake_llm(monkeypatch):
    from docqa import llm

    fake = FakeLLM()
    monkeypatch.setattr(llm, "stream_first_available", fake.stream_first_available)
    monkeypatch.setattr(llm, "complete", fake.complete)
    return fake
