"""The HTTP API and the production features around it: uploads, rate limiting, request
logging, feedback and housekeeping."""
import json
import logging
import time

from fastapi.testclient import TestClient

from docqa import api, ingest, store
from docqa.answer import answer
from docqa.config import settings

from .conftest import make_pdf

SOLAR = "Solar panels convert sunlight into electricity using photovoltaic cells made of silicon."


def done_event(sse_body: str) -> dict:
    lines = sse_body.split("event: done")[1].splitlines()
    return json.loads(next(line for line in lines if line.startswith("data: "))[6:])


def test_api_streams_answers_and_manages_documents(tmp_index, fake_llm, monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "min_score", 0.3)
    api._recent_asks.clear()
    make_pdf(tmp_path / "solar.pdf", [SOLAR])
    pdf = (tmp_path / "solar.pdf").read_bytes()
    with TestClient(api.app) as client:
        assert client.post("/documents", files={"file": ("notes.txt", b"hello")}).status_code == 400

        # Upload is accepted at once and indexed in the background; the job reports the outcome.
        accepted = client.post("/documents", files={"file": ("solar.pdf", pdf)})
        assert accepted.status_code == 202
        job = client.get("/documents/jobs/" + accepted.json()["job_id"]).json()
        assert (job["status"], job["result"], job["chunks"]) == ("done", "added", 1)
        again = client.post("/documents", files={"file": ("solar.pdf", pdf)}).json()
        assert client.get("/documents/jobs/" + again["job_id"]).json()["result"] == "unchanged"
        assert client.get("/documents/jobs/nope").status_code == 404

        with client.stream("POST", "/ask", json={"question": "What do solar panels convert sunlight into?"}) as r:
            body = "".join(r.iter_text())
        done = done_event(body)
        assert "event: token" in body and done["sources"][0]["source"] == "solar.pdf"
        assert done["request_id"] == r.headers["X-Request-ID"]
        assert done["tokens"] == {"prompt": 100, "completion": 7}

        # Feedback is tied to the request id and shows up in the metrics.
        assert client.post("/feedback", json={"request_id": done["request_id"], "rating": "up"}).status_code == 200
        assert client.post("/feedback", json={"request_id": "unknown", "rating": "down"}).status_code == 404
        assert client.post("/feedback", json={"request_id": done["request_id"], "rating": "meh"}).status_code == 422

        metrics = client.get("/metrics").json()
        assert metrics["count"] == 1 and {"p5_ms", "p50_ms", "p95_ms"} <= metrics.keys()
        assert (metrics["feedback_up"], metrics["feedback_down"], metrics["prompt_tokens"]) == (1, 0, 100)

        assert client.delete("/documents/solar.pdf").json()["status"] == "deleted"
        assert client.delete("/documents/solar.pdf").status_code == 404
        assert client.get("/health").json()["documents"] == 0


def test_api_key_protects_everything_except_health_and_the_page(tmp_index, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret")
    with TestClient(api.app) as client:
        assert client.get("/health").status_code == 200
        page = client.get("/")
        assert page.status_code == 200 and "Document Q&amp;A" in page.text
        assert page.headers["Cache-Control"] == "no-cache"  # an updated page must not be served stale
        assert client.get("/documents").status_code == 401
        assert client.get("/documents", headers={"X-API-Key": "secret"}).status_code == 200


def test_upload_of_a_pdf_without_text_is_rejected_and_changes_nothing(tmp_index, tmp_path):
    make_pdf(tmp_path / "scan.pdf", [""])  # a page with no text layer, like a scan
    with TestClient(api.app) as client:
        r = client.post("/documents", files={"file": ("scan.pdf", (tmp_path / "scan.pdf").read_bytes())})
        assert r.status_code == 422 and "no extractable text" in r.json()["detail"]
        assert list(tmp_index.iterdir()) == []  # nothing was left in the PDF folder
        assert client.get("/documents").json() == []


def test_unreadable_new_version_does_not_wipe_the_indexed_one(tmp_index):
    make_pdf(tmp_index / "a.pdf", [SOLAR])
    ingest.sync()
    (tmp_index / "a.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    report = ingest.sync()
    assert len(report["skipped"]) == 1 and report["skipped"][0].startswith("a.pdf")
    assert store.get_documents(settings.collection)["a.pdf"]["chunks"] == 1  # old version still indexed


def test_ask_is_rate_limited_per_caller(tmp_index, fake_llm, monkeypatch):
    monkeypatch.setattr(settings, "ask_rate_limit_per_minute", 2)
    api._recent_asks.clear()
    make_pdf(tmp_index / "a.pdf", [SOLAR])
    ingest.sync()
    with TestClient(api.app) as client:
        def ask(key):
            return client.post("/ask", json={"question": "anything", "stream": False}, headers={"X-API-Key": key})

        assert [ask("alice").status_code for _ in range(3)] == [200, 200, 429]
        assert int(ask("alice").headers["Retry-After"]) > 0
        assert ask("bob").status_code == 200  # another caller has their own allowance
    api._recent_asks.clear()


def test_every_request_writes_one_structured_log_line(tmp_index, fake_llm, monkeypatch, caplog):
    monkeypatch.setattr(settings, "min_score", 0.3)
    make_pdf(tmp_index / "a.pdf", [SOLAR])
    ingest.sync()
    logger = logging.getLogger("docqa.request")
    logger.addHandler(caplog.handler)
    result = answer("What do solar panels convert sunlight into?")
    logger.removeHandler(caplog.handler)
    record = json.loads(caplog.records[-1].getMessage())
    assert record["request_id"] == result["request_id"] and record["model"] == result["model"]
    assert record["retrieved"][0]["source"] == "a.pdf" and record["cited"] == [1]
    assert (record["prompt_tokens"], record["completion_tokens"]) == (100, 7)


def test_cleanup_removes_only_expired_sessions_and_log_rows(tmp_index):
    old = time.time() - 40 * 86400
    store.save_session("old", "", [])
    store.save_session("fresh", "", [])
    store.execute("UPDATE sessions SET updated_at = ? WHERE id = 'old'", (old,))
    store.log_request("r-old", "q", "a", 1, 0, None, "m", False, False, "dense")
    store.log_request("r-new", "q", "a", 1, 0, None, "m", False, False, "dense")
    store.execute("UPDATE request_log SET ts = ? WHERE request_id = 'r-old'", (old,))
    store.add_feedback("r-old", "up", "")
    assert store.cleanup() == {"sessions_deleted": 1, "requests_deleted": 1}
    assert [r["id"] for r in store.execute("SELECT id FROM sessions")] == ["fresh"]
    assert [r["request_id"] for r in store.execute("SELECT request_id FROM request_log")] == ["r-new"]
    assert len(store.execute("SELECT * FROM feedback")) == 1  # feedback is kept
