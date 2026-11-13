"""Vector index: embedding models and the Qdrant collection.

Each chunk is stored with two vectors: a dense one (meaning) and a sparse BM25 one
(exact words). Keeping both in the same point means an update or delete of a document
removes both at once, so the keyword index can never drift from the vector index."""
import atexit
import threading
from functools import lru_cache

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient, models

from .config import settings

# Embedded Qdrant is not safe for concurrent calls from several threads. Every Qdrant call
# takes this lock. With a Qdrant server (QDRANT_URL set) it costs nothing noticeable.
lock = threading.RLock()


@lru_cache
def client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_path.mkdir(parents=True, exist_ok=True)
    c = QdrantClient(path=str(settings.qdrant_path))
    atexit.register(c.close)  # embedded mode holds a file lock; release it cleanly
    return c


@lru_cache
def dense_model() -> TextEmbedding:
    return TextEmbedding(settings.embed_model, cache_dir=str(settings.model_cache))


@lru_cache
def sparse_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(settings.sparse_model, cache_dir=str(settings.model_cache))


@lru_cache
def rerank_model() -> TextCrossEncoder:
    return TextCrossEncoder(settings.rerank_model, cache_dir=str(settings.model_cache))


def ensure_collection() -> None:
    if client().collection_exists(settings.collection):
        return
    dim = len(next(iter(dense_model().embed(["dimension probe"]))))
    client().create_collection(
        settings.collection,
        vectors_config={"dense": models.VectorParams(size=dim, distance=models.Distance.COSINE)},
        # IDF is computed by Qdrant across the live collection, so BM25 statistics stay
        # correct after updates and deletes without any rebuild.
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    if settings.qdrant_url:  # payload indexes only exist in server mode
        client().create_payload_index(settings.collection, "source", models.PayloadSchemaType.KEYWORD)


def embed_passages(texts: list[str]) -> list[dict]:
    dense = dense_model().embed(texts)
    sparse = sparse_model().embed(texts)
    return [
        {"dense": d.tolist(), "bm25": models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())}
        for d, s in zip(dense, sparse)
    ]


@lru_cache(maxsize=2048)
def embed_query_dense(query: str) -> tuple[float, ...]:
    """Cached: repeated and follow-up queries skip the embedding model entirely."""
    return tuple(next(iter(dense_model().query_embed(query))).tolist())


def embed_query_sparse(query: str) -> models.SparseVector:
    s = next(iter(sparse_model().query_embed(query)))
    return models.SparseVector(indices=s.indices.tolist(), values=s.values.tolist())


def source_filter(name: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="source", match=models.MatchValue(value=name))])
