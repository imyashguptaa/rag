# Document Q&A assistant

A RAG service over PDFs: ingestion, hybrid retrieval with a reranker, cited answers, evaluation.
How it works and why each choice was made: `docs/DECISIONS.md`.

## Working rules (Karpathy guidelines)

These bias toward caution over speed. For trivial tasks, use judgment.

### 1. Think before coding
- State assumptions. If uncertain, ask.
- If there are several interpretations or approaches, present them with tradeoffs before picking.
  This includes library and architecture choices, not only requirements.
- If a simpler approach exists, say so. Push back when warranted.

### 2. Simplicity first
- Minimum code that solves the problem. No features beyond what was asked.
- No abstractions for single-use code. No configurability that was not requested.
- No error handling for impossible scenarios.
- No code path that cannot be run and verified here. If it cannot be tested, say so in
  `docs/DECISIONS.md` under "Not verified".

### 3. Surgical changes
- Touch only what the request needs. Match the existing style.
- Do not refactor or reformat neighbouring code. Mention unrelated problems, do not fix them.
- Remove imports, variables and functions that your own change made unused.

### 4. Goal-driven execution
- Turn the task into a check before writing code: a test that fails then passes, or a metric
  from `eval/run_eval.py` with a number to beat.
- Change one thing at a time and record the measured result, including results that got worse.
- Never write a number in `docs/DECISIONS.md` that did not come from a run in `eval/results/`.

## Project rules
- Every decision, default or measurement that changes goes into `docs/DECISIONS.md` in the same
  piece of work: what was chosen, why, what was rejected, and its limits.
- A retrieval change is not done until `python -m eval.run_eval` has been run and the experiment
  log has a new row.
- Prompts live only in `docqa/prompts.py`. A prompt change reruns `tests/test_injection.py`.

## Commands
```bash
.venv/Scripts/python -m pytest -q                                # unit tests + 10 injection tests (injection needs GROQ_API_KEY)
.venv/Scripts/python -m pytest -q --deselect tests/test_injection.py   # offline only
.venv/Scripts/python -m ruff check .                             # lint; CI fails on any finding
.venv/Scripts/python -m eval.run_eval --label <name>             # recall@5, free and local
.venv/Scripts/python -m eval.run_eval --label <name> --answers   # + answer correctness (Groq, resumable)
.venv/Scripts/python -m docqa.ingest                             # sync the index with data/pdfs
.venv/Scripts/python -m uvicorn docqa.api:app --port 8000        # API; chat page at /, API docs at /docs
```

## Things that will trip you up
- Embedded Qdrant (`data/qdrant`) allows one process at a time. Stop the API before running
  ingestion or an evaluation, and the other way round.
- Groq free tier: 8,000 tokens per minute and 200,000 tokens per day per model. One full
  `--answers` run uses most of a day's quota for the large model.
- The refusal threshold (`min_score`) was calibrated for this embedding model and corpus.
  Recalibrate it after changing either.
