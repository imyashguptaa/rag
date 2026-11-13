"""Ingestion: parse PDFs, chunk with overlap, embed, store.

`sync()` makes the index match the PDF folder incrementally: only new or changed files
are embedded, and files that disappeared are removed. Nothing else is touched.

Run: python -m docqa.ingest
"""
import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pymupdf
from qdrant_client import models

from . import index, store
from .config import settings

_ID_NAMESPACE = uuid.UUID("6f1d3c1e-0b7a-4c57-9d0e-5a1c2b3d4e5f")


def clean(text: str) -> str:
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # re-join words hyphenated across lines
    return re.sub(r"\s+", " ", text).strip()


class UnreadablePDF(ValueError):
    """The file cannot be indexed: it is not a valid PDF, or it has no text layer."""


def parse_pdf(path: Path) -> list[tuple[int, str]]:
    """Return (page number starting at 1, cleaned text) for every page that has text.
    Raises UnreadablePDF instead of indexing a document as nothing."""
    try:
        with pymupdf.open(path) as doc:
            pages = [(i + 1, clean(page.get_text())) for i, page in enumerate(doc)]
    except Exception as error:  # PyMuPDF raises several unrelated types for damaged files
        raise UnreadablePDF("the file is not a readable PDF") from error
    pages = [(n, t) for n, t in pages if t]
    if not pages:
        raise UnreadablePDF("the PDF has no extractable text. It is probably a scan, which needs OCR first")
    return pages


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Sliding window over words. Consecutive chunks share `overlap` words so a fact that
    straddles a boundary is still whole in one of them."""
    words = text.split()
    if len(words) <= size:
        return [text] if words else []
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def document_meta(path: Path) -> tuple[str, str]:
    """(title, publication date). Each comes from the folder's manifest.json if it lists the
    file, else from the PDF's own metadata, else from the file name / modified time."""
    manifest = path.parent / "manifest.json"
    entry = json.loads(manifest.read_text(encoding="utf-8")).get(path.name, {}) if manifest.exists() else {}
    with pymupdf.open(path) as doc:
        meta = doc.metadata or {}
    title = entry.get("title") or (meta.get("title") or "").strip() or path.stem.replace("_", " ")
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})", meta.get("creationDate") or "")
    date = entry.get("date") or ("-".join(m.groups()) if m else
                                 datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).date().isoformat())
    return title, date


def delete_document(name: str) -> bool:
    """Remove one document's chunks. Returns False if it was not indexed."""
    if name not in store.get_documents(settings.collection):
        return False
    with index.lock:
        index.client().delete(settings.collection, points_selector=models.FilterSelector(filter=index.source_filter(name)))
    store.remove_document(settings.collection, name)
    return True


def add_document(path: Path, sha256: str, pages: list[tuple[int, str]]) -> int:
    """Chunk, embed and store one parsed PDF. Returns the number of chunks written."""
    title, doc_date = document_meta(path)
    ingested_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payloads = []
    for page_no, text in pages:
        for i, chunk in enumerate(chunk_words(text, settings.chunk_words, settings.chunk_overlap)):
            payloads.append({
                "source": path.name, "title": title, "page": page_no, "chunk_index": i,
                "doc_date": doc_date, "ingested_at": ingested_at, "text": chunk,
            })
    # A chunk from the middle of a document rarely names the document. Putting the title in
    # front of the text that is embedded lets "... in the Self-RAG paper" match every chunk of
    # that paper, not only the pages that happen to repeat its name. The stored text is unchanged.
    prefix = f"{title}. " if settings.chunk_title_prefix else ""
    vectors = index.embed_passages([prefix + p["text"] for p in payloads])
    points = [
        models.PointStruct(
            # Deterministic id: re-ingesting the same chunk overwrites it instead of duplicating.
            id=str(uuid.uuid5(_ID_NAMESPACE, f"{p['source']}:{p['page']}:{p['chunk_index']}")),
            vector=v, payload=p,
        )
        for p, v in zip(payloads, vectors)
    ]
    for start in range(0, len(points), 128):
        with index.lock:
            index.client().upsert(settings.collection, points[start:start + 128])
    store.put_document(settings.collection, path.name, sha256, doc_date, len(pages), len(points), ingested_at)
    return len(points)


def upsert_document(path: Path) -> str:
    """Add a new PDF or replace a changed one. Returns 'added', 'updated' or 'unchanged'."""
    index.ensure_collection()
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    known = store.get_documents(settings.collection).get(path.name)
    if known and known["sha256"] == sha256:
        return "unchanged"
    pages = parse_pdf(path)  # before deleting anything: an unreadable new version must not wipe the old one
    if known:
        # Delete the old chunks first: the new version may have fewer pages or chunks,
        # and leftovers from the old one would otherwise keep being retrieved.
        delete_document(path.name)
    add_document(path, sha256, pages)
    return "updated" if known else "added"


def sync(pdf_dir: Path | None = None) -> dict[str, list[str]]:
    """Make the index match the folder. Only differences are processed."""
    pdf_dir = pdf_dir or settings.pdf_dir
    index.ensure_collection()
    report: dict[str, list[str]] = {"added": [], "updated": [], "unchanged": [], "deleted": [], "skipped": []}
    on_disk = {p.name: p for p in sorted(pdf_dir.glob("*.pdf"))}
    for name in store.get_documents(settings.collection):
        if name not in on_disk:
            delete_document(name)
            report["deleted"].append(name)
    for name, path in on_disk.items():
        try:
            report[upsert_document(path)].append(name)
        except UnreadablePDF as error:
            report["skipped"].append(f"{name} ({error})")
    return report


if __name__ == "__main__":
    result = sync()
    for action, names in result.items():
        print(f"{action}: {len(names)} {names if action != 'unchanged' else ''}")
    total = sum(d["chunks"] for d in store.get_documents(settings.collection).values())
    print(f"collection {settings.collection}: {total} chunks")
