"""Retrieval in three modes, each adding one thing to the previous:

dense          cosine similarity on embeddings (the baseline)
hybrid         dense + BM25, merged with reciprocal rank fusion
hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""
from dataclasses import asdict, dataclass

from . import index
from .config import settings


@dataclass
class Hit:
    id: str
    source: str
    page: int
    text: str
    doc_date: str
    score: float  # the score this mode ranked by

    def public(self) -> dict:
        d = asdict(self)
        d.pop("id")
        return d


def _search(query_vector, using: str, limit: int) -> list[Hit]:
    with index.lock:
        points = index.client().query_points(
            settings.collection, query=query_vector, using=using, limit=limit, with_payload=True,
        ).points
    return [
        Hit(str(p.id), p.payload["source"], p.payload["page"], p.payload["text"], p.payload["doc_date"], p.score)
        for p in points
    ]


def rrf(rankings: list[list[Hit]], k: int = 60) -> list[Hit]:
    """Reciprocal rank fusion: score = sum over lists of 1 / (k + rank). It uses ranks only,
    so dense cosine scores and BM25 scores never need to be put on the same scale."""
    scores: dict[str, float] = {}
    by_id: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            by_id[hit.id] = hit
    fused = sorted(by_id.values(), key=lambda h: scores[h.id], reverse=True)
    for h in fused:
        h.score = scores[h.id]
    return fused


def retrieve(query: str, mode: str | None = None, top_k: int | None = None) -> tuple[list[Hit], float]:
    """Return (top hits, confidence). Confidence is the best dense cosine similarity in every
    mode: it is the one score with a stable scale, so one refusal threshold works for all modes."""
    mode = mode or settings.retrieval_mode
    top_k = top_k or settings.top_k
    dense_vec = list(index.embed_query_dense(query))

    if mode == "dense":
        hits = _search(dense_vec, "dense", top_k)
        return hits, (hits[0].score if hits else 0.0)

    dense = _search(dense_vec, "dense", settings.candidate_k)
    confidence = dense[0].score if dense else 0.0
    sparse = _search(index.embed_query_sparse(query), "bm25", settings.candidate_k)
    fused = rrf([dense, sparse])

    if mode == "hybrid":
        return fused[:top_k], confidence
    if mode == "hybrid_rerank":
        candidates = fused[:settings.candidate_k]
        scores = list(index.rerank_model().rerank(query, [h.text for h in candidates]))
        for h, s in zip(candidates, scores):
            h.score = float(s)
        return sorted(candidates, key=lambda h: h.score, reverse=True)[:top_k], confidence
    raise ValueError(f"unknown retrieval mode: {mode}")
