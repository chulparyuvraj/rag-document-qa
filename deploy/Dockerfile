# ── Stage 1: Builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY .env.example ./.env

# Create directories for runtime data
RUN mkdir -p data/papers indexes/faiss models

# Expose FastAPI port
EXPOSE 8000

# Health check — Azure uses this to detect readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start uvicorn with multiple workers for concurrency
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--timeout-keep-alive", "30"]
