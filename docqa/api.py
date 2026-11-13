"""HTTP API.

POST   /ask                     answer a question; streams server-sent events by default
POST   /feedback                rate an answer (thumbs up / down) by its request id
GET    /documents               list indexed documents
POST   /documents               upload a PDF; it is checked, then indexed in the background
GET    /documents/jobs/{id}     progress of one upload
DELETE /documents/{name}        remove one document from the index
GET    /metrics                 latency percentiles (p5 / p50 / p95), cache hit rate, tokens, feedback
GET    /health
GET    /                        the chat web page

Run: uvicorn docqa.api:app --port 8000
"""
import asyncio
import json
import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import index, ingest, store
from .answer import answer, answer_events, request_log
from .config import settings
from .llm import LLMUnavailable

log = logging.getLogger("docqa")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
# Request records go out as bare JSON lines, one per request, so log tools can parse them.
_json_lines = logging.StreamHandler()
_json_lines.setFormatter(logging.Formatter("%(message)s"))
request_log.addHandler(_json_lines)
request_log.setLevel(logging.INFO)
request_log.propagate = False

MAX_UPLOAD_BYTES = 50 * 1024 * 1024
BUSY = "The language model is rate-limited right now. Please try again in a minute."
STATIC = Path(__file__).parent / "static"


async def _housekeeping() -> None:
    while True:
        removed = await asyncio.to_thread(store.cleanup)
        if any(removed.values()):
            log.info("housekeeping %s", removed)
        await asyncio.sleep(3600)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the models before the first request, so no user pays the cold-start cost.
    t0 = time.perf_counter()
    index.ensure_collection()
    index.embed_query_dense("warm up")
    index.embed_query_sparse("warm up")
    if settings.retrieval_mode == "hybrid_rerank":
        list(index.rerank_model().rerank("warm up", ["warm up"]))
    log.info("models ready in %.1fs, collection=%s", time.perf_counter() - t0, settings.collection)
    housekeeping = asyncio.create_task(_housekeeping())
    yield
    housekeeping.cancel()


def require_key(request: Request, x_api_key: str = Header(default="")) -> None:
    # /health stays open for the host's health check, and / serves the web page, which then
    # asks the visitor for the key. Neither reveals anything private.
    if settings.api_key and request.url.path not in ("/", "/health") and x_api_key != settings.api_key:
        raise HTTPException(401, "missing or wrong X-API-Key")


_recent_asks: dict[str, deque[float]] = defaultdict(deque)


def rate_limit(request: Request, x_api_key: str = Header(default="")) -> None:
    """Sliding one-minute window per caller, so one caller cannot spend the whole LLM quota.
    Counted in this process only: correct for the single-process deployment this app uses."""
    caller = x_api_key or (request.client.host if request.client else "unknown")
    now = time.monotonic()
    asks = _recent_asks[caller]
    while asks and now - asks[0] >= 60:
        asks.popleft()
    if len(asks) >= settings.ask_rate_limit_per_minute:
        wait = int(60 - (now - asks[0])) + 1
        raise HTTPException(429, f"Too many questions. Try again in {wait} seconds.", headers={"Retry-After": str(wait)})
    asks.append(now)


app = FastAPI(title="Document Q&A", lifespan=lifespan, dependencies=[Depends(require_key)])


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, max_length=64, pattern=r"^[\w-]+$")
    stream: bool = True


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    rating: Literal["up", "down"]
    comment: str = Field(default="", max_length=1000)


def _sse_event(event: dict) -> str:
    return f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"


def _sse(events, request_id: str):
    try:
        for event in events:
            yield _sse_event(event)
    except LLMUnavailable:
        log.warning("request %s: every language model is rate-limited or down", request_id)
        yield _sse_event({"type": "error", "detail": BUSY, "request_id": request_id})
    except Exception:
        log.exception("request %s failed mid-stream", request_id)
        yield _sse_event({"type": "error", "detail": "The answer could not be completed.", "request_id": request_id})


@app.get("/", include_in_schema=False)
def home():
    # no-cache: the browser must check for a newer page each time, so an updated page is never
    # hidden behind a stale cached copy. Unchanged pages still answer with a cheap 304.
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.post("/ask", dependencies=[Depends(rate_limit)])
def ask(req: AskRequest, response: Response):
    request_id = uuid.uuid4().hex  # also returned in the body and written to every log line
    if not req.stream:
        response.headers["X-Request-ID"] = request_id
        try:
            return answer(req.question, req.session_id, request_id=request_id)
        except LLMUnavailable:
            raise HTTPException(503, BUSY, headers={"Retry-After": "60", "X-Request-ID": request_id}) from None
    return StreamingResponse(
        _sse(answer_events(req.question, req.session_id, request_id=request_id), request_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "X-Request-ID": request_id},
    )


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    if not store.add_feedback(req.request_id, req.rating, req.comment):
        raise HTTPException(404, "unknown request_id")
    return {"status": "recorded"}


@app.get("/documents")
def list_documents():
    return sorted(store.get_documents(settings.collection).values(), key=lambda d: d["name"])


# Upload jobs live in memory: they describe work in this process and mean nothing after a restart.
_jobs: dict[str, dict] = {}
_one_ingest_at_a_time = threading.Lock()


def _run_ingest(job_id: str, path: Path) -> None:
    job = _jobs[job_id]
    job["status"] = "processing"
    try:
        with _one_ingest_at_a_time:
            job["result"] = ingest.upsert_document(path)
        job["chunks"] = store.get_documents(settings.collection)[path.name]["chunks"]
        job["status"] = "done"
    except Exception:
        log.exception("ingest job %s failed for %s", job_id, path.name)
        job.update(status="failed", error="The document could not be indexed.")


@app.post("/documents", status_code=202)
def upload_document(file: UploadFile, background: BackgroundTasks):
    name = Path(file.filename or "").name  # drop any directory part sent by the client
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "only .pdf files are accepted")
    data = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file is larger than 50 MB")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "file is not a PDF")

    # Check the file before it replaces anything: a scan with no text would otherwise be
    # accepted and indexed as nothing.
    job_id = uuid.uuid4().hex
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    incoming = settings.pdf_dir / f".incoming-{job_id}"
    incoming.write_bytes(data)
    try:
        ingest.parse_pdf(incoming)
    except ingest.UnreadablePDF as error:
        incoming.unlink()
        raise HTTPException(422, f"{name}: {error}") from None
    path = settings.pdf_dir / name
    os.replace(incoming, path)

    if len(_jobs) >= 200:
        _jobs.pop(next(iter(_jobs)))  # forget the oldest job
    _jobs[job_id] = {"job_id": job_id, "name": name, "status": "queued"}
    background.add_task(_run_ingest, job_id, path)  # embedding takes seconds to minutes
    return _jobs[job_id]


@app.get("/documents/jobs/{job_id}")
def upload_status(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


@app.delete("/documents/{name}")
def delete_document(name: str):
    name = Path(name).name
    with _one_ingest_at_a_time:
        if not ingest.delete_document(name):
            raise HTTPException(404, "document is not indexed")
    (settings.pdf_dir / name).unlink(missing_ok=True)
    return {"name": name, "status": "deleted"}


@app.get("/metrics")
def metrics():
    return store.latency_report()


@app.get("/health")
def health():
    docs = store.get_documents(settings.collection)
    return {"status": "ok", "documents": len(docs), "chunks": sum(d["chunks"] for d in docs.values()),
            "retrieval_mode": settings.retrieval_mode, "index_version": store.index_version()}
