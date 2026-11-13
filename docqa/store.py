"""Small SQLite store for everything that is not a vector: the document registry
(which file versions are indexed), chat sessions, the per-request log, and user feedback."""
import json
import sqlite3
import threading
import time

from .config import settings

_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    collection TEXT, name TEXT, sha256 TEXT, doc_date TEXT,
    pages INTEGER, chunks INTEGER, ingested_at TEXT,
    PRIMARY KEY (collection, name)
);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY, summary TEXT, turns TEXT, updated_at REAL
);
CREATE TABLE IF NOT EXISTS request_log (
    request_id TEXT PRIMARY KEY, ts REAL, question TEXT, answer TEXT,
    total_ms REAL, retrieve_ms REAL, first_token_ms REAL,
    model TEXT, cached INTEGER, refused INTEGER, mode TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS feedback (
    request_id TEXT PRIMARY KEY, ts REAL, rating TEXT, comment TEXT, question TEXT, answer TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


def db() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            settings.db_path.parent.mkdir(parents=True, exist_ok=True)
            _conn = sqlite3.connect(settings.db_path, check_same_thread=False, isolation_level=None)
            _conn.row_factory = sqlite3.Row
            _conn.executescript(SCHEMA)
        return _conn


def execute(sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = db()
    with _lock:
        return conn.execute(sql, params).fetchall()


# --- document registry -------------------------------------------------------

def get_documents(collection: str) -> dict[str, dict]:
    rows = execute("SELECT * FROM documents WHERE collection = ?", (collection,))
    return {r["name"]: dict(r) for r in rows}


def put_document(collection: str, name: str, sha256: str, doc_date: str, pages: int, chunks: int, ingested_at: str) -> None:
    execute(
        "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?, ?, ?)",
        (collection, name, sha256, doc_date, pages, chunks, ingested_at),
    )
    bump_index_version()


def remove_document(collection: str, name: str) -> None:
    execute("DELETE FROM documents WHERE collection = ? AND name = ?", (collection, name))
    bump_index_version()


def index_version() -> int:
    """Changes whenever a document is added, updated or deleted. Cached answers are keyed
    on it, so an index change invalidates them automatically."""
    rows = execute("SELECT value FROM meta WHERE key = 'index_version'")
    return int(rows[0]["value"]) if rows else 0


def bump_index_version() -> None:
    execute("INSERT OR REPLACE INTO meta VALUES ('index_version', ?)", (str(index_version() + 1),))


# --- chat sessions -----------------------------------------------------------

def load_session(session_id: str) -> tuple[str, list[dict]]:
    rows = execute("SELECT summary, turns FROM sessions WHERE id = ?", (session_id,))
    if not rows:
        return "", []
    return rows[0]["summary"], json.loads(rows[0]["turns"])


def save_session(session_id: str, summary: str, turns: list[dict]) -> None:
    execute(
        "INSERT OR REPLACE INTO sessions VALUES (?, ?, ?, ?)",
        (session_id, summary, json.dumps(turns), time.time()),
    )


# --- request log and feedback ------------------------------------------------

def log_request(request_id: str, question: str, answer: str, total_ms: float, retrieve_ms: float,
                first_token_ms: float | None, model: str, cached: bool, refused: bool, mode: str,
                prompt_tokens: int | None = None, completion_tokens: int | None = None) -> None:
    execute(
        "INSERT OR REPLACE INTO request_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (request_id, time.time(), question, answer, total_ms, retrieve_ms, first_token_ms, model,
         int(cached), int(refused), mode, prompt_tokens, completion_tokens),
    )


def add_feedback(request_id: str, rating: str, comment: str) -> bool:
    """Store a rating for one answered request. Returns False if the request is unknown.
    The question and answer are copied in, so feedback outlives the request log's retention."""
    rows = execute("SELECT question, answer FROM request_log WHERE request_id = ?", (request_id,))
    if not rows:
        return False
    execute("INSERT OR REPLACE INTO feedback VALUES (?, ?, ?, ?, ?, ?)",
            (request_id, time.time(), rating, comment, rows[0]["question"], rows[0]["answer"]))
    return True


def latency_report(last_n: int = 1000) -> dict:
    rows = execute("SELECT total_ms, cached, prompt_tokens, completion_tokens FROM request_log ORDER BY ts DESC LIMIT ?", (last_n,))
    ratings = {r["rating"]: r["n"] for r in execute("SELECT rating, COUNT(*) AS n FROM feedback GROUP BY rating")}
    report = {"count": len(rows), "feedback_up": ratings.get("up", 0), "feedback_down": ratings.get("down", 0)}
    values = sorted(r["total_ms"] for r in rows)
    if not values:
        return report

    def pct(p: float) -> float:
        return round(values[min(len(values) - 1, int(p / 100 * len(values)))], 1)

    return {
        **report,
        "p5_ms": pct(5), "p50_ms": pct(50), "p95_ms": pct(95),
        "cache_hit_rate": round(sum(r["cached"] for r in rows) / len(rows), 3),
        "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rows),
        "completion_tokens": sum(r["completion_tokens"] or 0 for r in rows),
    }


def cleanup() -> dict:
    """Delete chat sessions and request-log rows older than their retention period.
    Both tables otherwise grow for ever. Feedback is kept."""
    now = time.time()
    sessions = execute("DELETE FROM sessions WHERE updated_at < ? RETURNING id", (now - settings.session_ttl_days * 86400,))
    requests = execute("DELETE FROM request_log WHERE ts < ? RETURNING request_id", (now - settings.request_log_days * 86400,))
    return {"sessions_deleted": len(sessions), "requests_deleted": len(requests)}
