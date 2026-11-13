# Python 3.12, not the 3.10 used on the first development machine: 3.10 is past end-of-life,
# and Pillow and filelock only ship their security fixes for 3.11 and newer.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY docqa ./docqa
COPY scripts ./scripts
COPY eval ./eval

# Download the embedding and reranker weights at build time, into a path outside /app/data
# so a mounted data volume does not hide them. Containers then start without a download.
ENV MODEL_CACHE=/opt/models
RUN python -c "from docqa import index; index.dense_model(); index.sparse_model(); index.rerank_model()"

# BAKE_CORPUS=1 downloads the public corpus and builds the index into the image. Use it on
# hosts with no persistent disk (Render's free plan), where an index built at runtime is lost
# on every restart. Leave it at 0 when the index lives in a Qdrant server or on a volume.
ARG BAKE_CORPUS=0
RUN if [ "$BAKE_CORPUS" = "1" ]; then python scripts/download_corpus.py && python -m docqa.ingest; fi

# Run as a non-root user that owns the data directory.
RUN useradd --create-home app && mkdir -p /app/data && chown -R app /app/data /opt/models
USER app

# Hosts such as Render tell the container which port to use through PORT.
ENV PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ['PORT'] + '/health')"

# One worker: the answer cache and (in embedded mode) the Qdrant lock live in the process.
CMD ["sh", "-c", "exec uvicorn docqa.api:app --host 0.0.0.0 --port $PORT"]
