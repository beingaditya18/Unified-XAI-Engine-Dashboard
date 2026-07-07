# ==============================================================================
# Base Image: Python 3.10 slim
# ==============================================================================
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

# Install system utilities & build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY config/ config/
COPY explainability/ explainability/
COPY models/ models/
COPY data/ data/
COPY dashboard/ dashboard/
COPY src/ src/

# Run training to ensure models exist prior to serving
RUN python models/train_models.py

# Create non-privileged user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# ==============================================================================
# Development Stage
# ==============================================================================
FROM base AS dev
ENV ENV=development
EXPOSE 8000 8501

# ==============================================================================
# FastAPI API Server
# ==============================================================================
FROM base AS api
ENV ENV=production
EXPOSE 8000
CMD ["uvicorn", "src.api.api:app", "--host", "0.0.0.0", "--port", "8000"]

# ==============================================================================
# Streamlit Frontend UI
# ==============================================================================
FROM base AS ui
ENV ENV=production
EXPOSE 8501
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
