# OpenMythos inference server
# CPU image for dev/staging; swap base to nvidia/cuda for GPU production.
#
# Build:   docker build -t openmythos-serve -f serve/Dockerfile .
# Run:     docker run -p 8000:8000 openmythos-serve
# GPU run: docker run --gpus all -e DEVICE=cuda -p 8000:8000 openmythos-serve

FROM python:3.12-slim

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer cache)
COPY pyproject.toml requirements.txt ./
COPY open_mythos/ ./open_mythos/
COPY serve/ ./serve/

# Install CPU torch + runtime deps
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --system transformers datasets fastapi uvicorn[standard] pydantic

# Install the package itself
RUN uv pip install --system -e .

# Runtime environment
ENV DEVICE=cpu
ENV DEFAULT_LOOPS=4
ENV MAX_LOOPS=16
ENV MODEL_DIM=256
ENV MODEL_ATTN=gqa
ENV TOKENIZER_NAME=gpt2
# Set MODEL_CHECKPOINT to load a trained .pt file:
# ENV MODEL_CHECKPOINT=/checkpoints/model.pt

EXPOSE 8000

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
