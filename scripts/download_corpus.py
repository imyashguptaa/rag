"""Download the public evaluation corpus (open-access arXiv papers) into data/pdfs.

The version is pinned in each id so page numbers in eval/questions.jsonl stay valid.
Run: python scripts/download_corpus.py
"""
import json
import time
from pathlib import Path

import httpx

PAPERS = [
    # (arxiv id with version, file stem, title, published date)
    ("1706.03762v7", "attention_is_all_you_need", "Attention Is All You Need", "2017-06-12"),
    ("1810.04805v2", "bert", "BERT: Pre-training of Deep Bidirectional Transformers", "2018-10-11"),
    ("1908.10084v1", "sentence_bert", "Sentence-BERT", "2019-08-27"),
    ("2004.04906v3", "dense_passage_retrieval", "Dense Passage Retrieval for Open-Domain QA", "2020-04-10"),
    ("2004.12832v2", "colbert", "ColBERT", "2020-04-27"),
    ("2005.11401v4", "rag", "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks", "2020-05-22"),
    ("2104.08663v4", "beir", "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models", "2021-04-17"),
    ("2212.10496v1", "hyde", "Precise Zero-Shot Dense Retrieval without Relevance Labels (HyDE)", "2022-12-20"),
    ("2309.15217v1", "ragas", "Ragas: Automated Evaluation of Retrieval Augmented Generation", "2023-09-26"),
    ("2310.11511v1", "self_rag", "Self-RAG", "2023-10-17"),
]

OUT = Path(__file__).resolve().parent.parent / "data" / "pdfs"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": "docqa-corpus-fetch"}) as client:
        for arxiv_id, stem, title, date in PAPERS:
            path = OUT / f"{stem}.pdf"
            manifest[path.name] = {"title": title, "date": date, "arxiv_id": arxiv_id}
            if path.exists():
                print(f"have  {path.name}")
                continue
            r = client.get(f"https://arxiv.org/pdf/{arxiv_id}")
            r.raise_for_status()
            path.write_bytes(r.content)
            print(f"saved {path.name} ({len(r.content) / 1e6:.1f} MB)")
            time.sleep(3)  # arXiv asks automated clients to go slowly
    # Sidecar metadata: ingestion reads the document date from here when present.
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
