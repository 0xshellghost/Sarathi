# ── Build Stage ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime Stage ────────────────────────────────────────────────
FROM python:3.12-slim

# Security: non-root user
RUN groupadd -r sarathi && useradd -r -g sarathi -s /sbin/nologin sarathi

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create directories for persistent data
RUN mkdir -p /app/chroma_data /app/tmp && \
    chown -R sarathi:sarathi /app

USER sarathi

# ChromaDB persistence
VOLUME ["/app/chroma_data"]

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
