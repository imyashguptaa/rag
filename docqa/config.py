"""All tunable settings in one place. Override any of them with an environment variable
or a line in .env (for example RETRIEVAL_MODE=hybrid_rerank)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    # Paths
    pdf_dir: Path = ROOT / "data" / "pdfs"
    qdrant_path: Path = ROOT / "data" / "qdrant"  # embedded Qdrant; ignored when qdrant_url is set
    qdrant_url: str = ""  # e.g. http://qdrant:6333 in docker-compose
    db_path: Path = ROOT / "data" / "app.db"
    model_cache: Path = ROOT / "data" / "models"  # downloaded embedding / reranker weights

    # Chunking (sizes in words; a chunk never crosses a page so its page number is exact)
    chunk_words: int = 120
    chunk_overlap: int = 20
    chunk_title_prefix: bool = True  # embed "<document title>. <chunk>" instead of the bare chunk

    # Models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    groq_api_key: str = ""
    api_key: str = ""  # if set, every API request must send it in the X-API-Key header
    llm_model: str = "openai/gpt-oss-120b"
    llm_model_small: str = "openai/gpt-oss-20b"

    # Retrieval
    retrieval_mode: str = "hybrid_rerank"  # dense | hybrid | hybrid_rerank
    top_k: int = 10  # chunks given to the answer model
    candidate_k: int = 15  # candidates fetched per retriever, and the number reranked
    # Below this top dense cosine similarity the assistant answers "I do not know".
    min_score: float = 0.66

    # Chat memory: when the stored turns exceed the limit, the oldest are folded into a summary
    memory_token_limit: int = 1500
    memory_keep_turns: int = 4

    # Protection and housekeeping
    ask_rate_limit_per_minute: int = 20  # per API key, or per client address when no key is sent
    session_ttl_days: int = 30  # chat sessions untouched for this long are deleted
    request_log_days: int = 30  # request-log rows (question, answer, timings) older than this are deleted

    # Latency features
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    route_easy_queries: bool = True

    @property
    def collection(self) -> str:
        # One collection per chunking config, so chunk-size experiments never overwrite each other.
        return f"chunks_w{self.chunk_words}_o{self.chunk_overlap}" + ("_title" if self.chunk_title_prefix else "")


settings = Settings()
