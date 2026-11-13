"""Evaluation run.

Retrieval (free, local):
    recall@5      share of questions where a top-5 chunk comes from a gold page
    recall@1, MRR the same idea, stricter
    refusal       share of out-of-scope questions below the "I do not know" threshold,
                  and share of real questions wrongly below it

Answers (uses the LLM, pass --answers):
    correctness   share of answers an LLM judge marks CORRECT against the reference answer
    faithfulness  share of given answers whose every claim is supported by the excerpts they cite

Examples:
    python -m eval.run_eval --label baseline --answers
    python -m eval.run_eval --label hybrid --mode hybrid
    python -m eval.run_eval --label chunk120 --chunk-words 120 --chunk-overlap 20
    python -m eval.run_eval --mode hybrid_rerank --min-recall 0.85      # CI gate
"""
import argparse
import json
import sys
import time
from pathlib import Path

from docqa import index, ingest, llm, prompts
from docqa.answer import cited_sources, plain_brackets
from docqa.config import settings
from docqa.retrieve import retrieve

HERE = Path(__file__).resolve().parent


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (HERE / name).read_text(encoding="utf-8").splitlines() if line.strip()]


def first_gold_rank(q: dict, hits) -> int | None:
    for rank, h in enumerate(hits, start=1):
        if h.source == q["source"] and h.page in q["pages"]:
            return rank
    return None


def judge(question: str, reference: str, answer: str) -> str:
    verdict = llm.complete(
        [{"role": "system", "content": prompts.JUDGE},
         {"role": "user", "content": f"Question: {question}\nReference answer: {reference}\nAnswer to grade: {answer}"}],
        model=settings.llm_model_small, max_tokens=200,
    ).upper()
    for label in ("INCORRECT", "PARTIAL", "CORRECT"):  # INCORRECT first: it contains "CORRECT"
        if label in verdict:
            return label
    return "INCORRECT"


def faithful(answer: str, hits) -> str:
    """Is the answer supported by the excerpts it cites? An answer that cites nothing counts
    as UNSUPPORTED: the reader has no way to check it."""
    cited = cited_sources(answer, hits)
    if not cited:
        return "UNSUPPORTED"
    excerpts = "\n\n".join(f"[{c['id']}] {c['text']}" for c in cited)
    verdict = llm.complete(
        [{"role": "system", "content": prompts.FAITHFULNESS},
         {"role": "user", "content": f"Excerpts:\n{excerpts}\n\nAnswer to check: {answer}"}],
        model=settings.llm_model_small, max_tokens=200,
    ).upper()
    return "UNSUPPORTED" if "UNSUPPORTED" in verdict else "SUPPORTED"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="")
    ap.add_argument("--mode", default=settings.retrieval_mode)
    ap.add_argument("--chunk-words", type=int, default=settings.chunk_words)
    ap.add_argument("--chunk-overlap", type=int, default=settings.chunk_overlap)
    ap.add_argument("--title-prefix", action=argparse.BooleanOptionalAction, default=settings.chunk_title_prefix)
    ap.add_argument("--top-k", type=int, default=settings.top_k)
    ap.add_argument("--answers", action="store_true", help="also generate answers and judge them")
    ap.add_argument("--limit", type=int, default=0, help="use only the first N questions per document set")
    ap.add_argument("--min-recall", type=float, default=0.0, help="exit 1 if recall@5 is below this")
    ap.add_argument("--min-correct", type=float, default=0.0, help="exit 1 if correctness is below this")
    args = ap.parse_args()

    settings.chunk_words, settings.chunk_overlap = args.chunk_words, args.chunk_overlap
    settings.chunk_title_prefix, settings.top_k = args.title_prefix, args.top_k
    ingest.sync()  # no-op if this chunking config is already indexed

    questions = load("questions.jsonl")
    if args.limit:
        questions = questions[:: max(1, len(questions) // args.limit)][: args.limit]  # spread across documents

    rows, all_hits, retrieve_ms = [], [], []
    for q in questions:
        t0 = time.perf_counter()
        hits, confidence = retrieve(q["question"], args.mode, top_k=max(10, args.top_k))
        retrieve_ms.append((time.perf_counter() - t0) * 1000)
        all_hits.append(hits[:args.top_k])  # what the answer model is given
        rows.append({"id": q["id"], "rank": first_gold_rank(q, hits), "confidence": round(confidence, 3),
                     "retrieved": [[h.source, h.page] for h in hits]})
    oos = [retrieve(q["question"], args.mode)[1] for q in load("unanswerable.jsonl")]
    # Embedded Qdrant allows one process at a time. Retrieval is finished, so release it
    # before the slow, rate-limited LLM phase.
    index.client().close()

    if args.answers:
        # Answers are saved as they are produced. If the run dies (Groq's free tier has a daily
        # token limit), running the same command again continues where it stopped.
        partial_path = HERE / "results" / f"{args.label or args.mode}.partial.json"
        partial_path.parent.mkdir(exist_ok=True)
        done = json.loads(partial_path.read_text(encoding="utf-8")) if partial_path.exists() else {}
        for q, row, hits in zip(questions, rows, all_hits):
            if q["id"] not in done:
                if row["confidence"] < settings.min_score:
                    answer = prompts.REFUSAL
                else:
                    answer = plain_brackets(llm.complete(prompts.build_messages(q["question"], hits)))
                done[q["id"]] = {"answer": answer, "verdict": judge(q["question"], q["answer"], answer)}
                if not answer.startswith("I do not know"):
                    done[q["id"]]["faithful"] = faithful(answer, hits)
                partial_path.write_text(json.dumps(done), encoding="utf-8")
            row.update(done[q["id"]])
            print(f"{q['id']:12} rank={row['rank']} {row['verdict']} {row.get('faithful', '')}", flush=True)
        partial_path.unlink()

    n = len(rows)
    ranks = [r["rank"] for r in rows]
    retrieve_ms.sort()
    summary = {
        "label": args.label or args.mode,
        "mode": args.mode, "chunk_words": args.chunk_words, "chunk_overlap": args.chunk_overlap,
        "questions": n,
        "top_k": args.top_k, "title_prefix": args.title_prefix,
        "recall@5": round(sum(r is not None and r <= 5 for r in ranks) / n, 3),
        "recall@8": round(sum(r is not None and r <= 8 for r in ranks) / n, 3),
        "recall@10": round(sum(r is not None for r in ranks) / n, 3),
        "recall@1": round(sum(r == 1 for r in ranks) / n, 3),
        "mrr": round(sum(1 / r for r in ranks if r) / n, 3),
        "retrieve_ms_p50": round(retrieve_ms[n // 2], 1),
        "min_score": settings.min_score,
        "out_of_scope_refused": round(sum(c < settings.min_score for c in oos) / len(oos), 3),
        "answerable_wrongly_refused": round(sum(r["confidence"] < settings.min_score for r in rows) / n, 3),
    }
    if args.answers:
        verdicts = [r["verdict"] for r in rows]
        summary["correct"] = round(verdicts.count("CORRECT") / n, 3)
        summary["partial"] = round(verdicts.count("PARTIAL") / n, 3)
        summary["incorrect"] = round(verdicts.count("INCORRECT") / n, 3)
        # Faithfulness: of the answers that were given (not refusals), how many are fully
        # supported by the excerpts they cite. Rows from runs older than this metric have none.
        checked = [r["faithful"] for r in rows if "faithful" in r]
        summary["faithfulness"] = round(checked.count("SUPPORTED") / len(checked), 3) if checked else None
        summary["faithfulness_checked"] = len(checked)
        summary["answer_model"], summary["judge_model"] = settings.llm_model, settings.llm_model_small

    out = HERE / "results" / f"{summary['label']}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "out_of_scope_confidence": oos, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    misses = [r["id"] for r in rows if r["rank"] is None or r["rank"] > 5]
    if misses:
        print("missed:", ", ".join(misses))

    if summary["recall@5"] < args.min_recall or summary.get("correct", 1.0) < args.min_correct:
        print("FAILED quality gate")
        sys.exit(1)


if __name__ == "__main__":
    main()
