# V1 Code Review AI Service Dockerfile
# Multi-stage build for optimized production image

# =============================================================================
# Stage 1: Base with CUDA support
# =============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    curl \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser


# =============================================================================
# Stage 2: Dependencies
# =============================================================================
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies in correct order
# 1. First install torch (required by flash-attn build)
RUN pip install --upgrade pip setuptools wheel \
    && pip install torch>=2.1.0 --index-url https://download.pytorch.org/whl/cu121

# 2. Install flash-attn separately (requires torch to be pre-installed)
RUN pip install flash-attn>=2.3.0 --no-build-isolation || echo "flash-attn installation skipped"

# 3. Install remaining dependencies
RUN pip install -r requirements.txt || pip install --ignore-installed -r requirements.txt


# =============================================================================
# Stage 3: Production
# =============================================================================
FROM dependencies AS production

WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data /app/models /app/cache /app/logs \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]


# =============================================================================
# Stage 4: Development
# =============================================================================
FROM dependencies AS development

WORKDIR /app

# Install development dependencies
RUN pip install pytest pytest-asyncio pytest-cov black ruff mypy ipython

# Copy application code
COPY --chown=appuser:appgroup . .

# Switch to non-root user
USER appuser

# Expose ports (app + debugger)
EXPOSE 8000 5678

# Development command with hot reload
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
