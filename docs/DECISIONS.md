# Document Q&A assistant: how it works and why it is built this way

This is the living record of the project. Every choice has its reason and the alternatives
that were rejected. Every number comes from a run whose raw output is in `eval/results/`.
Update this file whenever a decision or a measurement changes.

## 1. What the system does

It answers questions about a folder of PDFs. Each answer cites the page it came from, and the
assistant says "I do not know" when the documents do not cover the question.

```
PDFs ──parse──▶ pages ──chunk──▶ chunks ──embed──▶ Qdrant (dense vector + BM25 vector + source, page, date)

question ─▶ [answer cache] ─▶ [rewrite follow-up] ─▶ retrieve ─▶ [confidence check] ─▶ [pick model] ─▶ LLM ─▶ answer + citations
                 hit: return        only if chat           dense + BM25      low: "I do not know"   easy: small     streamed
                                    history exists         ─▶ RRF ─▶ rerank                         hard: large
```

| File | Job |
|---|---|
| `docqa/config.py` | Every setting, overridable by environment variable |
| `docqa/ingest.py` | Parse, chunk, embed, store. Incremental add / update / delete |
| `docqa/index.py` | Embedding models and the Qdrant collection |
| `docqa/retrieve.py` | Dense, hybrid and reranked retrieval |
| `docqa/prompts.py` | Every prompt, and the context formatter |
| `docqa/answer.py` | The pipeline: cache, refusal, model routing, generation, citations |
| `docqa/memory.py` | Chat history with summarisation |
| `docqa/store.py` | SQLite: document registry, sessions, latency log |
| `docqa/api.py` | FastAPI app |
| `docqa/static/index.html` | The chat web page served at `/` |
| `eval/run_eval.py` | recall@5 and answer correctness |
| `tests/` | Unit tests and the 10 prompt-injection tests |

## 2. How one question is answered

1. **Cache.** If the question has no chat history and was answered before on the same index
   version, the stored answer is returned in a few milliseconds.
2. **Rewrite.** If there is chat history, the small model rewrites a follow-up such as "how was
   it trained?" into a standalone search query.
3. **Retrieve.** The query is embedded twice (dense and BM25). Each index returns 15 candidates,
   reciprocal rank fusion merges the two lists, and a cross-encoder reranks the merged list.
   The top 10 chunks go to the model.
4. **Confidence check.** If the best dense cosine similarity is under 0.66, the assistant
   replies "I do not know" without calling the LLM.
5. **Pick a model.** Short single-fact questions with confident retrieval go to the small model.
   Everything else goes to the large one. If the chosen model is rate-limited, the other answers.
6. **Generate.** The model sees the numbered excerpts and must cite them as `[1]`, `[2]`.
   Tokens stream to the client as they arrive.
7. **Finish.** Citation numbers are mapped back to file and page, the request latency is
   logged, the answer is cached, and the turn is saved to the session.

## 3. Decisions

### Corpus
**Chosen:** ten open-access arXiv papers on retrieval and RAG (157 pages), versions pinned in
`scripts/download_corpus.py`.
**Why:** the evaluation needs a fixed corpus with known page numbers. Pinned versions mean a
gold page stays correct for anyone who reruns it, including CI.
**Limit:** all ten are two-column English research papers. Scanned PDFs, tables-heavy reports
and other languages are untested.

### PDF parsing: PyMuPDF
**Why:** fast, keeps page boundaries (needed for page-level citations), no system dependencies.
**Rejected:** `pypdf` (slower, worse on two-column layouts), `unstructured` (heavy install, and
its layout models are not needed for text PDFs), OCR (the corpus has a text layer).
**Limit:** scanned PDFs produce no text and are silently skipped. Tables come out as flat text.

### Chunking: 120 words, 20 overlap, never across a page
**Why per page:** every chunk has exactly one page number, so citations and the evaluation's
"correct source page" are exact.
**Why overlap:** a fact that straddles a boundary is whole in at least one chunk.
**Why 120:** measured, see section 4. Smaller chunks put the matching sentence in a tighter
vector. 120 beat 220 and 350 on recall@5.
**Why words, not tokens:** no tokenizer dependency, and the size is easy to reason about.
**Rejected:** semantic or recursive chunking. They add a moving part before there was a
measurement showing fixed-size chunks were the bottleneck.
**Cost:** a sentence that continues onto the next page is cut.

### Document title in front of every embedded chunk
**What:** the text that is embedded is "<document title>. <chunk>". The stored and displayed
text is the bare chunk. The title comes from `manifest.json`, else the PDF's metadata, else the
file name.
**Why:** a chunk from the middle of a paper rarely names the paper. A question such as "which
retriever does Self-RAG use?" then matches the few pages that repeat the name, not the page
with the answer. With the title on every chunk, the name stops deciding the ranking.
**Measured:** no change at top 5 (0.89), but recall@10 rose from 0.94 to 0.97.

### Ten chunks go to the answer model, not five
**Why:** the baseline's wrong answers were mostly retrieval misses (27 of 47), and in another 12
the right page was retrieved but not the passage holding the answer. Ten 120-word chunks are
about 1,200 words, close to what the baseline sent (five 220-word chunks), and they put a gold
page in front of the model for 97% of questions instead of 89%.
**Cost:** about 2,400 tokens per request instead of 1,400. On Groq's free tier that is about
three answers a minute.
**Note:** recall@5 is still the reported retrieval metric, because the project brief names it.

### Embeddings: `BAAI/bge-small-en-v1.5` through fastembed (ONNX)
**Why:** 384 dimensions, strong for its size, runs on CPU in about 7 ms per query. fastembed
runs ONNX, so the install and the Docker image avoid a multi-gigabyte PyTorch.
**Rejected:** API embeddings (cost per call, network latency on every query, documents leave
the machine), larger local models (slower; try one only if retrieval stalls).
**Limit:** English only. Hinglish or Hindi queries would need a multilingual model.

### Vector store: Qdrant
**Why:** it supports delete-by-filter and upsert-by-id, which is what makes document update and
delete possible without a rebuild. It stores dense and sparse vectors on the same point. The
same client code runs embedded (a folder, used for development and tests) or against a server
(used in docker-compose).
**Rejected:** FAISS (no metadata filtering or deletes without extra bookkeeping), Chroma (no
built-in sparse vectors), pgvector (needs a Postgres server from day one).
**Limit:** embedded mode allows one process at a time, so it is for development. Production
uses the Qdrant server.

### Keyword search: BM25 as a Qdrant sparse vector
**Why:** dense retrieval misses exact tokens such as `IVFPQ`, `Nq = 32` or `gpt-3.5-turbo-16k`.
BM25 catches them. Storing BM25 as a sparse vector on the same point means one delete removes
both representations, and Qdrant computes IDF over the live collection, so statistics stay
right after updates.
**Rejected:** `rank_bm25` in memory. It must be rebuilt on every document change and it lives
outside the store, so the two indexes can drift apart.

### Fusion: reciprocal rank fusion (RRF)
**Why:** dense cosine scores and BM25 scores are on different scales. RRF uses ranks only, so
no score normalisation or weight tuning is needed.
**Rejected:** weighted score sum (needs a tuned weight that changes with the corpus).

### Reranker: `ms-marco-MiniLM-L-6-v2` cross-encoder on 15 candidates
**Why:** a cross-encoder reads query and chunk together, which is more accurate than comparing
two separate vectors. It lifted recall@1 from 0.57 to 0.67.
**Why 15 candidates:** measured. 30 candidates gave the same recall@5 (0.89) at 1,104 ms per
query; 15 gave 0.89 at 479 ms; 10 dropped to 0.85.
**Rejected:** `bge-reranker-base` (about 1 GB and several times slower on CPU).
**Cost:** about 0.45 s per query on CPU. It is the largest part of retrieval time.

### LLM: Groq, `openai/gpt-oss-120b` (large) and `openai/gpt-oss-20b` (small)
**Why Groq:** a key was already available, there is a free tier, and generation is fast.
**Why two models:** the small one is used for easy queries, query rewriting, summarising and
judging. The large one answers anything that needs reasoning.
**Limit:** the free tier allows 8,000 tokens per minute and 200,000 tokens per day per model.
A full 100-question answer evaluation uses about 150,000 to 200,000 tokens of the large model,
so only one fits in a day. `eval/run_eval.py` saves answers as it goes and resumes when rerun.
Fine for development, not for real traffic.

### Model fallback when a model is rate-limited
**What:** for a live question, if the chosen model returns a rate limit, timeout or server
error, the other model answers at once. If both fail, the API returns a clear "try again in a
minute" message (HTTP 503 or a stream `error` event). The response says which model answered.
**Why:** on 2026-11-13 the large model's daily quota ran out and every reasoning question
failed. A user should get a slightly weaker answer, not an error.
**Why no retries here:** a person is waiting. Skipping to the other model is faster than
waiting out a rate limit. Batch work (`llm.complete`, used by evaluation) still waits and
retries on one model, so evaluation numbers are never mixed across models.

### No LangChain or LlamaIndex
**Why:** the whole application is about 1,100 lines of plain Python. Every step is visible, testable
and measurable. A framework would hide exactly the parts this project is about.

### Prompt and citations
**How:** excerpts are numbered inside `<context>` with source, page and date. The model must
cite `[n]` after each claim. After generation, numbers are mapped back to file and page, and a
number that matches no excerpt is dropped.
**Why not ask the model for file names:** it can misspell or invent them. It can only pick an
index, and the server resolves it.

### "I do not know": two layers
1. **Retrieval score.** Top dense cosine under 0.66 means refuse without calling the LLM.
   Dense cosine is used in every mode because RRF and reranker scores are not on a stable scale.
2. **Prompt rule.** The model must reply with the exact refusal sentence when the excerpts do
   not hold the answer. This catches near-domain questions that score high.

**Why 0.66:** measured on 100 answerable and 10 out-of-scope questions. Answerable questions
score 0.68 or more. At 0.66, 8 of 10 out-of-scope questions are refused and no answerable one
is. The two that pass ("How many parameters does GPT-4 have?", "What learning rate was used to
train Mamba?") look like the corpus, which is what layer 2 is for.
**Limit:** the threshold belongs to this embedding model and corpus. Recalibrate after
changing either. The gap between 0.66 and 0.68 is narrow.

### Chat memory
**How:** recent turns are kept word for word in SQLite. When they pass 1,500 tokens, all but the
last 4 turns are folded into a running summary by the small model.
**Why:** the prompt stays a bounded size however long the chat is, and recent turns keep their
exact wording.
**Why rewrite follow-ups:** "how was it trained?" retrieves nothing useful alone.
**Rejected:** retrieving over past turns with embeddings (more machinery than a Q&A chat needs).
**Limit:** token counts are estimated as characters / 4. The limit is soft.

### API: FastAPI with server-sent events
**Why SSE:** streaming is one-directional here, SSE works through ordinary HTTP proxies, and it
needs no client library.
**Rejected:** WebSockets (two-way, not needed).
**One code path:** `/ask` with `stream=false` consumes the same generator as the stream.

### Web page: one static HTML file served at `/`
**What:** `docqa/static/index.html`. A chat page that streams the answer, turns `[n]` into
buttons that jump to the source card (file, page, year, the passage itself), marks refusals
clearly, and has a panel to list, add and remove documents.
**Why one file with no framework or build step:** the page has one screen and talks to four
endpoints. A React build would add a toolchain and a second deploy artefact for no gain.
**Safety:** answer text is inserted as text nodes, never as HTML, so a document cannot inject
markup into the page. Only `**bold**` and `` `code` `` are styled.
**API key:** the page itself is public. If the server has an `API_KEY`, the page asks for it on
the first 401 and keeps it in that browser's local storage.
**Choosing a file to upload:** a real "Choose a PDF" button at the top of the Documents panel,
and drag-and-drop anywhere on the page. The first version hid the file input with
`display: none` inside a label at the bottom of the panel, and a user could not upload with it:
no upload request ever reached the server. Hidden-by-`display:none` inputs do not open the file
picker in every browser, and the control sat below the document list. The input is now only
visually hidden and is opened by the button.
**Page caching:** `/` is sent with `Cache-Control: no-cache`. Without it the browser kept
showing an old copy of the page after the file changed, which hid the fix above.
**Checked in a browser:** the button calls the file picker, a dropped PDF is uploaded, indexed
and listed, and a dropped non-PDF is refused with a message. **Not checked:** the operating
system's file dialog itself, which test tools cannot drive.

### SQLite for app state
**Why:** the document registry, sessions and latency log are small relational data. SQLite is
one file with no server.
**Limit:** one API process. Several replicas would need Postgres and a shared cache.

### Caching
**What:** an in-process answer cache keyed on (normalised question, retrieval mode, index
version), plus an LRU cache of query embeddings.
**Why the index version is in the key:** any add, update or delete makes old answers
unreachable, so the cache cannot serve an answer from a deleted document.
**Why not for chat turns:** with history, the same words can mean a different question.
**Rejected:** semantic cache (matching similar questions by embedding). A wrong match returns a
confident wrong answer, and that risk is not worth it before exact-match hit rates are known.

### Routing easy queries to the small model
**Rule:** 14 words or fewer, top dense score 0.70 or more, and no reasoning word (why, how,
compare, explain, ...). "How many / much / long" count as easy.
**Why a rule:** it costs nothing at request time and is easy to explain and test.
**Rejected:** an LLM classifier call (adds the latency it is meant to save).

### Update and delete without a rebuild
**How:** the registry stores a SHA-256 per file. `sync()` compares the folder with the registry:
new file → embed and add; changed hash → delete that file's chunks by filter, then add; missing
file → delete. Unchanged files are never touched. Chunk ids are deterministic
(`uuid5(file, page, chunk)`).
**Why delete before re-adding on update:** the new version may have fewer chunks, and leftovers
from the old one would keep being retrieved.
**Tested:** `tests/test_ingest.py` asserts only the changed file is re-embedded.

### Prompt injection
**Defences:** (1) the system prompt says excerpts and history are untrusted data, never
instructions; (2) excerpts are wrapped in tags, and any tag-like text inside a document is
neutralised so a document cannot close the context and fake a system message; (3) the model
cites by index only; (4) the system prompt may not be revealed.
**Tests:** 10 live cases in `tests/test_injection.py`: six through a poisoned document, four
through the user message. All 10 pass on `gpt-oss-120b`.
**Limit:** this lowers the risk, it does not remove it. Ten cases are a regression check, not
proof. The assistant has no tools and takes no actions, which bounds the damage to a wrong or
misleading answer.

### Evaluation design
**Set:** 100 questions, 10 per paper, each with gold page(s) and a reference answer, plus 10
out-of-scope questions. Gold pages were found by searching the parsed text for the answer, so
each one is verified against the PDF. A question lists several pages when the fact is stated
on several.
**recall@5:** a hit when any top-5 chunk comes from a gold page of the right document.
**Answer correctness:** `gpt-oss-20b` grades each answer CORRECT / PARTIAL / INCORRECT against
the reference. Only CORRECT counts.
**Judge prompt, version 2 (2026-11-13):** the first prompt marked an answer PARTIAL when the
reference held a detail the question never asked for. Version 2 grades against what the
question asks and allows extra correct detail. The saved baseline answers were re-graded with
it (`baseline_rejudged.json`): still 0.53, with three verdicts changing in both directions. So
the judge change does not inflate scores, and later runs are comparable with the baseline.
**Limits:** the questions were written by the same author as the system, mostly single-fact,
in clean English. Real user questions are messier, so real-world numbers will be lower. The
judge is an LLM and can be wrong; spot-check its verdicts in the results files. Gold pages can
miss a page that also states the fact, which makes recall slightly pessimistic.

### CI
Every push runs the linter, the dependency audit, the unit tests, the 10 injection tests, the
retrieval evaluation (fails under recall@5 0.85) and answer correctness on a 20-question sample
(fails under 0.75). Retrieval is gated on all 100 questions because it is free. Answers use a
sample because of the rate limit.

### Pinned dependencies, lint and a vulnerability audit
**What:** `requirements.txt` pins every direct dependency to the exact version the evaluation
numbers were measured with. `requirements-dev.txt` holds pytest, ruff and pip-audit, which the
Docker image does not install. CI runs `ruff check .` and `pip-audit -r requirements.txt`.
**Why pins:** with open-ended versions, the same commit can build differently next month, and
a retrieval number can move without any code change.
**Not pinned:** transitive dependencies. A full lock file made on Windows would not install on
the Linux image. The audit is what watches them.

### Python 3.12 in Docker and CI, 3.10 on the development machine
**Why:** the first audit run found known vulnerabilities in two transitive packages, Pillow
12.3.0 (needed by fastembed) and filelock 4.1.0 (needed by huggingface_hub). Both ship their
fixes only for Python 3.11 and newer, and Python 3.10 passed end-of-life in October 2026.
**Exposure today:** low. The app never opens an image, so Pillow's decoders are not reached.
**Decision:** the image and CI use 3.12, where pip installs the fixed versions. The code still
targets 3.10 syntax so the existing machine keeps working.
**Not verified:** nothing here has run on Python 3.12 yet. The first CI run is the check.
**To do on the development machine:** install Python 3.12 and rebuild `.venv`, so development
and production match.

### Rate limit on `/ask`
**What:** 20 questions per minute per caller, sliding window. The caller is the API key, or
the client address when no key is sent. Over the limit returns HTTP 429 with `Retry-After`.
**Why:** one script in a loop could spend the whole day's LLM quota in minutes.
**Why in-process:** the app runs as one process. A shared counter (Redis) is only needed with
several instances.
**Limit:** behind a proxy without a key, all callers share the proxy's address. The Render
deployment always has a key.

### Request ids and structured request logs
**What:** every `/ask` gets a request id, returned in the `X-Request-ID` header and in the
answer. One JSON line per request goes to stdout with the id, question, search query, each
retrieved chunk's source, page and score, which chunks were cited, the model that answered,
cache and refusal flags, timings and token counts. The same row is stored in SQLite.
**Why:** when someone reports a bad answer, the id leads to what was retrieved and which model
answered. Without it there is nothing to look at.
**Privacy:** questions and answers are stored. They are deleted after 30 days (see housekeeping).

### Upload checks and background indexing
**What:** an upload is parsed before it replaces anything. A file that is not a readable PDF,
or has no text layer, is rejected with HTTP 422 and a reason. A good file is accepted with HTTP
202 and a job id, indexed in the background, and `GET /documents/jobs/{id}` reports progress.
**Why the check:** a scanned PDF used to be accepted and indexed as zero chunks, and the user
only found out when every question about it was refused.
**Why background:** embedding a long PDF takes longer than an HTTP request should stay open.
**Also fixed:** replacing a document with an unreadable new version used to delete the old
chunks first. The new file is now parsed before anything is deleted.
**Limits:** job status is kept in memory and is lost on restart. Jobs run one at a time.
**Rejected:** OCR for scans (a heavy dependency for a case nobody has asked for yet).

### Housekeeping
**What:** at start-up and every hour, chat sessions untouched for 30 days and request-log rows
older than 30 days are deleted. Feedback is kept, with its own copy of the question and answer.
**Why:** both tables grew without limit.

### Feedback and the faithfulness metric
**Feedback:** `POST /feedback` stores a thumbs up or down against a request id, and the page
has the two buttons under each answer. Counts appear in `/metrics`. Rated answers are the raw
material for new evaluation questions written by real users.
**Faithfulness:** `eval/run_eval.py --answers` now also asks the judge model whether every
claim in an answer is supported by the excerpts it cites. An answer with no citation counts as
unsupported. Correctness asks "is it right?"; faithfulness asks "can the reader verify it from
the cited pages?". An answer can be right from the model's own memory and still unfaithful.
**Not measured yet:** the metric was added after the last full run. It will be filled in by
the next one. A 4-question smoke run gave 2 of 4 supported with all 4 correct. That is far too
few to read as a rate, and nobody has yet checked whether those two are real unsupported claims
or a strict judge. Read the verdicts by hand after the first full run before trusting the number.
**Bug found while adding it:** the evaluation did not convert the model's full-width citation
brackets to `[n]`, so it saw no citations at all. The conversion now lives in one function,
`plain_brackets`, used by both the live pipeline and the evaluation.

### Docker
Model weights are downloaded at build time, so a container starts without a network download.
The image runs as a non-root user. `docker-compose.yml` runs the API with a Qdrant server.

### Deployment: Render, from the Dockerfile
**How:** `render.yaml` is a Render Blueprint. Render builds the Dockerfile on every push that
passes CI and runs the container on the port it passes in `PORT`.
**Memory decides the plan.** Measured on the development machine: about 420 MB in hybrid mode
and about 920 MB with the reranker loaded.

| Render plan | RAM | What fits | recall@5 |
|---|---|---|---|
| `free` (blueprint default) | 512 MB, 0.1 CPU | hybrid, no reranker. Little headroom | 0.84 |
| `1c-2g` (paid) | 2 GB, 1 CPU | full pipeline with reranker | 0.89 |

**Why the index is baked into the image (`BAKE_CORPUS=1`):** the free plan has no persistent
disk, and the filesystem is wiped on every restart and every wake from sleep. Building the index
at start would add minutes to each wake. Baking it at image build makes start-up a model load.
**Cost of that:** PDFs uploaded through the API on the free plan disappear on restart. Keeping
them needs a paid plan with a disk mounted at `/app/data`.
**Why an API key is generated:** the URL is public, and without a key anyone could spend the
Groq quota or delete documents. `/health` stays open for Render's health check.
**Rejected:** a separate Qdrant service on Render (a second paid service for 955 chunks).
**Not verified:** the Docker image has never been built (no Docker on the development
machine), and the Qdrant-server code path (`QDRANT_URL`) has never been run.

## 4. Experiment log

One change at a time. 100 questions. Raw output: `eval/results/<label>.json`.

| # | Label | Change from the row above | recall@5 | recall@1 | MRR | Retrieval p50 |
|---|---|---|---|---|---|---|
| 0 | `baseline_retrieval` | Baseline: dense, 220-word chunks, 40 overlap | 0.69 | 0.38 | 0.501 | 7 ms |
| 1a | `dense_chunk120` | Chunk 120 / 20 | 0.76 | 0.45 | 0.573 | 7 ms |
| 1b | `dense_chunk350` | Chunk 350 / 60 (from row 0) | 0.71 | 0.39 | 0.502 | 7 ms |
| 2 | `hybrid_chunk120` | Row 1a + BM25 with RRF | 0.84 | 0.57 | 0.666 | 40 ms |
| 3 | `hybrid_rerank_chunk120` | Row 2 + reranker, 30 candidates | 0.89 | 0.68 | 0.766 | 1,104 ms |
| 4 | `hybrid_rerank_chunk120_cand15` | Row 3 with 15 candidates | 0.89 | 0.67 | 0.763 | 479 ms |
| 4b | `hybrid_rerank_chunk120_cand10` | Row 3 with 10 candidates | 0.85 | 0.70 | 0.763 | 317 ms |
| 5 | `title_prefix` | Row 4 + document title in front of each embedded chunk (**current default**) | 0.89 | 0.68 | 0.773 | 480 ms |
| 6 | `query_instruction` | Row 5 + BGE query instruction. **Rejected** | 0.90 | 0.69 | 0.778 | 480 ms |
| 7a | `rerank_jina-reranker-v1-turbo-en` | Row 5 with the Jina turbo reranker. **Rejected** | 0.91 | 0.68 | 0.777 | 654 ms |
| 7b | `rerank_ms-marco-MiniLM-L-12-v2` | Row 5 with the 12-layer MiniLM reranker. **Rejected** | 0.92 | 0.68 | 0.778 | 961 ms |

Recall beyond the top 5 (how often a gold page is among the chunks the model is given):

| Config | recall@5 | recall@8 | recall@10 |
|---|---|---|---|
| Row 4, no title | 0.89 | 0.92 | 0.94 |
| Row 5, with title | 0.89 | 0.93 | 0.97 |

Why rows 6 and 7 were rejected: one to three more questions out of 100 is inside the noise of a
100-question set, and each costs something real. The query instruction pushed five answerable
questions under the refusal threshold. The other rerankers add 0.2 to 0.5 s to every query.
`bge-reranker-base` was also tried and stopped: over 8 minutes for 100 questions on CPU.

Also measured at 220-word chunks: hybrid 0.82, hybrid + reranker 0.86.

What each step taught:
- **Chunk size.** Smaller chunks helped (+7 points). Larger ones did not.
- **Hybrid.** The biggest single gain (+8 points). The questions it fixed ask for exact tokens.
- **Reranker.** +5 points on recall@5 and +10 on recall@1, for about 0.45 s per query.
- **Still missed (11 questions).** In nearly all, the right paper is retrieved but the wrong
  page: the question names the paper's topic and the answer sits in a details section.

### Answer correctness

| Label | Retrieval | Answer model | Correct | Partial | Incorrect |
|---|---|---|---|---|---|
| `baseline` | dense, 220, 5 chunks | gpt-oss-120b | 0.53 | 0.02 | 0.45 |
| `baseline_rejudged` | same answers, judge prompt v2 | gpt-oss-120b | 0.53 | 0.00 | 0.47 |
| `final_qwen` (77 of 100 answered) | hybrid + rerank, 120, title, 10 chunks | qwen3.8-27b | 66 of 77 = 0.86 | 1 of 77 | 10 of 77 |
| `final` | same as `final_qwen` | gpt-oss-120b | NOT MEASURED | | |

**Like for like on the same 77 questions:** baseline 48 correct (0.62), improved pipeline 66
correct (0.86). Of the improved pipeline's 77 answers, one was a refusal.

**What is still open.** `final_qwen` stopped at question 77 when that model's 200,000-token
daily quota ran out. The 23 unanswered questions are the last of HyDE and all of Ragas and
Self-RAG, and Self-RAG holds five of the eleven retrieval misses, so the full-set number will
probably be lower than 0.86. The answer model also differs from the baseline's (the large
model's quota was already spent), so part of the gain may come from the model and not from
retrieval. Two reruns close both gaps, each on a fresh daily quota:

```bash
LLM_MODEL=qwen/qwen3.8-27b python -m eval.run_eval --label final_qwen --answers   # resumes at question 78
python -m eval.run_eval --label final --answers                                   # same answer model as the baseline
```

The baseline's 45 incorrect answers are mostly retrieval failures: recall@5 was 0.69, and when the
gold page is not retrieved the model correctly says "I do not know", which the judge marks
INCORRECT. Answer quality here is capped by retrieval.

### Refusal

| Threshold | Out-of-scope refused (of 10) | Answerable wrongly refused (of 100) |
|---|---|---|
| 0.55 | 3 | 0 |
| 0.60 | 6 | 0 |
| 0.66 (**current**) | 8 | 0 |

### Latency

NOT MEASURED. The benchmark stopped on the same daily quota. Run
`python -m scripts.latency_bench` after it resets (about 55,000 large-model tokens).

What is already known: retrieval takes about 480 ms with the reranker (was 1,104 ms with 30
candidates), a cache hit returns in a few milliseconds, and model load (about 10 s) is paid at
start-up instead of on the first request.

Expectation to check, not a result: the cache will cut the median sharply, but p95 is set by the
slowest uncached requests, so the p95 cut depends on how much faster the small model is on the
easy queries routed to it. If p95 falls short of the 30% target, record that here.

## 5. Known limits and what to do next

- **Not deployed yet.** `render.yaml` is ready, but deploying needs a GitHub repository and a
  Render account. The Docker image has never been built: Docker is not installed on the
  development machine, so the first build will be in CI or on Render.
- **CI has never run.** The linter and tests pass locally, but the audit, the Python 3.12
  build and the Docker build are all first verified by CI.
- **Free-tier rate limit.** 8,000 tokens per minute and 200,000 per day per model. A paid tier
  or a second provider is needed before real traffic.
- **Single process.** The answer cache, rate limiter, upload jobs and SQLite assume one API
  process. Several instances would need Redis and Postgres.
- **Left out on purpose until needed:** per-user accounts, OCR, Prometheus/Grafana dashboards,
  a prompt-injection screening model on every request, and load testing.
- **English only.** A multilingual embedding model is the first step for Hinglish queries.
- **Retrieval misses.** Recall@5 has stalled near 0.9 across four further experiments. The
  next things to try, one at a time: a larger embedding model, and splitting on section
  headings instead of fixed word counts.
- **No authentication beyond one shared API key.**
