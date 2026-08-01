FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# LightGBM's compiled extension links against libgomp (GNU OpenMP) at
# runtime, which python:3.12-slim doesn't ship -- without this,
# `import lightgbm` fails at container startup with
# "OSError: libgomp.so.1: cannot open shared object file", caught by
# actually running the built image, not just building it.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first, so this layer is cached across code-only changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY src/ ./src/

# The model bundle and DuckDB database are gitignored, regenerable build
# artifacts (see .gitignore) -- produced by `src/pipeline/preprocessing.py`
# and `src/pipeline/registry.py` before this image is built, matching
# README's "Getting started" order. Baking a specific trained snapshot into
# the image keeps `docker run` self-contained for this project's scope;
# swapping to a volume mount or loading from the MLflow registry at
# container startup is the natural next step once Phase 15 (Deployment)
# needs the model to be updatable without a rebuild.
COPY models/ ./models/
COPY data/energy.duckdb ./data/energy.duckdb

RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
