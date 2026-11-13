"""Latency benchmark: the same workload before and after the latency features.

before  cache off, every query on the large model
after   answer cache on, easy queries routed to the small model

The workload replays evaluation questions with repeats (default 40% of requests repeat an
earlier question), because a cache can only help on traffic that repeats. Change --repeat-share
to see how much the result depends on that assumption.

Requests are paced with a pause so Groq's free-tier rate limit does not add retry waits to
the measurements. The pause is not counted.

Run: python -m scripts.latency_bench
"""
import argparse
import json
import random
import time
from pathlib import Path

from docqa import cache, index
from docqa.answer import answer
from docqa.config import settings

ROOT = Path(__file__).resolve().parent.parent


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    return round(values[min(len(values) - 1, int(p / 100 * len(values)))], 1)


def run(workload: list[str], pause: float) -> list[dict]:
    cache._cache.clear()
    rows = []
    for question in workload:
        result = answer(question)
        rows.append({"question": question, "ms": result["timings"]["total_ms"], "model": result["model"],
                     "cached": result["cached"]})
        print(f"  {result['timings']['total_ms']:8.1f} ms  {'cache' if result['cached'] else result['model']}", flush=True)
        if not result["cached"]:
            time.sleep(pause)
    return rows


def report(rows: list[dict]) -> dict:
    ms = [r["ms"] for r in rows]
    return {"requests": len(ms), "p5_ms": percentile(ms, 5), "p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95),
            "cache_hits": sum(r["cached"] for r in rows),
            "small_model": sum(r["model"] == settings.llm_model_small and not r["cached"] for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=30)
    ap.add_argument("--repeat-share", type=float, default=0.4)
    ap.add_argument("--pause", type=float, default=9.0)
    args = ap.parse_args()

    rng = random.Random(7)
    lines = (ROOT / "eval" / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    questions = [json.loads(line)["question"] for line in lines]
    distinct = rng.sample(questions, round(args.requests * (1 - args.repeat_share)))
    workload = distinct + [rng.choice(distinct) for _ in range(args.requests - len(distinct))]
    rng.shuffle(workload)

    # Warm the models so neither run pays the one-off load time.
    index.embed_query_dense("warm up")
    index.embed_query_sparse("warm up")
    list(index.rerank_model().rerank("warm up", ["warm up"]))

    results = {}
    for name, cache_on, routing_on in (("before", False, False), ("after", True, True)):
        settings.cache_enabled, settings.route_easy_queries = cache_on, routing_on
        print(f"{name}: cache={cache_on} routing={routing_on}")
        rows = run(workload, args.pause)
        results[name] = {**report(rows), "rows": rows}

    before, after = results["before"]["p95_ms"], results["after"]["p95_ms"]
    results["p95_reduction"] = round(1 - after / before, 3)
    results["workload"] = {"requests": args.requests, "repeat_share": args.repeat_share}
    out = ROOT / "eval" / "results" / "latency.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    for name in ("before", "after"):
        print(name, {k: v for k, v in results[name].items() if k != "rows"})
    print(f"p95 reduction: {results['p95_reduction']:.0%}")


if __name__ == "__main__":
    main()
