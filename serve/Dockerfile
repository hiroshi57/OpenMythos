# OpenMythos inference server — Production image (Gunicorn + Uvicorn workers)
#
# Dev/single-worker:
#   docker build -t openmythos-serve -f serve/Dockerfile .
#   docker run -p 8000:8000 openmythos-serve
#
# Production (docker-compose):
#   docker compose up -d
#
# GPU:
#   docker run --gpus all -e DEVICE=cuda -p 8000:8000 openmythos-serve
#
# Auth (set API key):
#   docker run -e API_KEY=your-secret-key -p 8000:8000 openmythos-serve

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache optimization)
COPY pyproject.toml requirements.txt ./
COPY open_mythos/ ./open_mythos/
COPY serve/ ./serve/

# Install CPU torch + runtime + production server deps
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system \
        transformers \
        datasets \
        fastapi \
        "uvicorn[standard]" \
        gunicorn \
        pydantic \
        starlette

# Package is already copied to /app — set PYTHONPATH so it's importable without pip install

# Non-root user for security
RUN useradd -m -u 1000 mythos && chown -R mythos:mythos /app
USER mythos

# Runtime environment defaults (override via -e or docker-compose.yml)
ENV PYTHONPATH=/app
ENV DEVICE=cpu
ENV DEFAULT_LOOPS=4
ENV MAX_LOOPS=16
ENV MODEL_DIM=256
ENV MODEL_ATTN=gqa
ENV TOKENIZER_NAME=gpt2
ENV RATE_LIMIT_RPM=60
ENV WORKERS=2
# Authentication (leave empty to disable):
# ENV API_KEY=your-secret-key
# Pre-trained checkpoint (leave empty for random weights):
# ENV MODEL_CHECKPOINT=/checkpoints/model.pt

EXPOSE 8000

# Production: Gunicorn manages multiple Uvicorn workers for concurrency.
# WORKERS env allows runtime tuning (e.g. -e WORKERS=4 on larger VM).
CMD gunicorn serve.api:app \
    --workers ${WORKERS} \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --keep-alive 5 \
    --access-logfile - \
    --error-logfile -
