# ── Sales Forecasting API — Dockerfile ────────────────────────────────────────
# Phase 4: Containerise the FastAPI prediction endpoint
#
# Build:  docker build -t sales-forecasting-api .
# Run:    docker run -p 8000:8000 sales-forecasting-api
# Test:   curl http://localhost:8000/health

# Use slim Python image — smaller, faster to pull
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ── System dependencies ────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
# Copy requirements first (Docker layer caching — only reinstalls if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic \
        pandas \
        numpy \
        scikit-learn \
        lightgbm \
        xgboost \
        joblib \
        pyyaml

# ── Copy application code ──────────────────────────────────────────────────────
COPY app/        ./app/
COPY src/        ./src/
COPY models/     ./models/
COPY configs/    ./configs/
COPY data/processed/feature_meta.json        ./data/processed/feature_meta.json
COPY data/processed/reports/monitoring_summary.json ./data/processed/reports/monitoring_summary.json

# ── Environment variables ──────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Health check ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Expose port ────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start server ───────────────────────────────────────────────────────────────
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
