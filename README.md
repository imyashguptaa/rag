# Document Q&A assistant with evaluation

Ask questions about a folder of PDFs. Answers cite their source page, and the assistant says
"I do not know" when the documents do not cover the question.

Why each part is built the way it is, and every measured result: [docs/DECISIONS.md](docs/DECISIONS.md).

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt -r requirements-dev.txt   # Linux/macOS: .venv/bin/python
cp .env.example .env                                      # then put your GROQ_API_KEY in it
.venv/Scripts/python scripts/download_corpus.py           # ten public arXiv papers into data/pdfs
.venv/Scripts/python -m docqa.ingest                      # parse, chunk, embed, store
```

To use your own documents, copy PDFs into `data/pdfs` and run the ingest command again. It only
processes files that are new, changed or removed.

## Use

```bash
# One question from the command line
.venv/Scripts/python -m docqa.answer "How many attention heads does the Transformer use?"

# The API
.venv/Scripts/python -m uvicorn docqa.api:app --port 8000
```

| Request | What it does |
|---|---|
| `POST /ask` `{"question": "...", "session_id": "abc", "stream": true}` | Answer. Streams server-sent events (`token`, then `done` with sources and timings). `session_id` turns on chat memory. `"stream": false` returns one JSON object |
| `GET /documents` | List indexed documents |
| `POST /feedback` `{"request_id": "...", "rating": "up"}` | Rate an answer. The request id comes back with every answer |
| `POST /documents` (multipart `file`) | Add a PDF, or replace the indexed version of the same name. Rejects files with no text. Returns a job id; indexing runs in the background |
| `GET /documents/jobs/{id}` | Progress of one upload: queued, processing, done or failed |
| `DELETE /documents/{name}` | Remove one document |
| `GET /metrics` | Latency p5 / p50 / p95, cache hit rate, token totals and feedback counts |
| `GET /health` | Status and index size |

Open `http://localhost:8000/` for the chat page: ask a question, read the streamed answer, and click a citation to see the page it came from. Interactive API docs are at `http://localhost:8000/docs`.

## Evaluate

```bash
.venv/Scripts/python -m pytest -q                                   # unit tests + 10 prompt-injection tests
.venv/Scripts/python -m eval.run_eval --label my_run                # recall@5 on 100 questions (free, local)
.venv/Scripts/python -m eval.run_eval --label my_run --answers      # also answer correctness (uses Groq)
.venv/Scripts/python -m eval.run_eval --mode dense --chunk-words 220 --chunk-overlap 40   # the baseline
.venv/Scripts/python -m scripts.latency_bench                       # p5 / p50 / p95 before and after caching + routing
.venv/Scripts/python -m ruff check .                                # lint
.venv/Scripts/python -m pip_audit -r requirements.txt               # known vulnerabilities in dependencies
```

## Docker

```bash
docker compose up --build
```

This starts the API on port 8000 with a Qdrant server. Set `GROQ_API_KEY` (and optionally
`API_KEY`) in the environment or in `.env` first.

## Deploy on Render

1. Push this folder to a GitHub repository.
2. In the repository settings, add `GROQ_API_KEY` under Secrets and variables > Actions, so CI can run the injection tests and the answer evaluation.
3. In the Render dashboard choose New > Blueprint and pick the repository. Render reads `render.yaml` and asks for `GROQ_API_KEY`.
4. After the first deploy, copy the generated `API_KEY` from the service's Environment page. Send it as the `X-API-Key` header on every request except `/health`.

`render.yaml` defaults to the free plan without the reranker. Its header comment explains the paid alternative that runs the full pipeline.
