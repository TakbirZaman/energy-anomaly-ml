# ─────────────────────────────────────────────────────────────────────────────
#  LSTM Autoencoder — Anomaly Detection API
#  Base image: python:3.9-slim (Debian Bookworm, ~130 MB)
#
#  Build:
#    docker build -t anomaly-api .
#
#  Run (model files must be present in the current directory):
#    docker run --rm -p 8000:8000 \
#      -v "$(pwd)/best_model.pt:/app/best_model.pt:ro" \
#      -v "$(pwd)/model_config.json:/app/model_config.json:ro" \
#      anomaly-api
#
#  Override env vars at runtime:
#    docker run ... -e PORT=9090 -e MODEL_CHECKPOINT=/app/best_model.pt anomaly-api
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────────────────────────
# Compile wheels in an isolated stage so build tools never reach the final image.
FROM python:3.9-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Install only the system tools needed to compile any C-extension wheels,
# then purge the apt cache in the same layer to keep the layer small.
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy the pinned requirements file first so Docker can cache this
# expensive layer — it only re-runs when requirements.txt changes.
COPY requirements.txt .

# Install all packages into a self-contained prefix so they can be
# copied cleanly into the runtime stage.
# --no-cache-dir  : don't write the pip HTTP cache (saves ~50 MB)
# --index-url     : pull CPU-only PyTorch wheels (~1 GB lighter than CUDA)
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.3.1 \
 && pip install --no-cache-dir \
        numpy==1.26.4 \
        fastapi==0.111.1 \
        "uvicorn[standard]==0.30.1" \
        pydantic==2.7.4


# ── Stage 2: lean runtime image ───────────────────────────────────────────────
FROM python:3.9-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Application defaults — override with -e at `docker run` time
    HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_CHECKPOINT=/app/best_model.pt \
    MODEL_CONFIG=/app/model_config.json

# Create a non-root user for the process (principle of least privilege)
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

WORKDIR /app

# Pull only the installed site-packages from the builder — no compiler,
# no build headers, no apt caches reach this stage.
COPY --from=builder /usr/local/lib/python3.9/site-packages \
                    /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin \
                    /usr/local/bin

# Copy application source
COPY app.py .

# Model artefacts are injected via volume mounts at runtime (see header above).
# Providing placeholder paths keeps the image portable and avoids baking
# a trained model's weights into the image layer.
# If you prefer to bake the weights in (e.g. for air-gapped deployment),
# uncomment the two lines below:
# COPY best_model.pt .
# COPY model_config.json .

# Switch to the non-root user before the process starts
USER appuser

# Document which port the application listens on
EXPOSE 8000

# Health-check: poll /health every 30 s; mark unhealthy after 3 failures.
# Docker (and ECS/K8s adapters) use this to restart unhealthy containers.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python - <<'PYEOF'
import urllib.request, sys
try:
    r = urllib.request.urlopen("http://localhost:8000/health", timeout=8)
    import json
    body = json.loads(r.read())
    sys.exit(0 if body.get("healthy") else 1)
except Exception:
    sys.exit(1)
PYEOF

# Production entrypoint:
#   --workers 1   : one process per container (scale out via replicas, not workers)
#   --no-access-log : reduces noise; structured access logging can be added via
#                     a reverse proxy (nginx / AWS ALB)
CMD ["sh", "-c", \
     "uvicorn app:app \
      --host $HOST \
      --port $PORT \
      --workers 1 \
      --no-access-log \
      --log-level info"]
