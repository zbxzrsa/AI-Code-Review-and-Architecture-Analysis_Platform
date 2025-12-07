# =============================================================================
# VCAI (Version Control AI) Service Dockerfile
# Root-level Dockerfile for three-version pipeline builds
# =============================================================================

FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user
RUN groupadd --gid 1000 vcai && \
    useradd --uid 1000 --gid vcai --shell /bin/bash --create-home vcai

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM base AS dependencies

# Copy requirements files
COPY backend/requirements.txt ./requirements.txt
COPY backend/requirements-test.txt ./requirements-test.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# =============================================================================
# Application Stage
# =============================================================================
FROM dependencies AS app

# Copy application code
COPY backend/app ./app
COPY backend/shared ./shared
COPY backend/dev-api-server.py ./dev-api-server.py

# Copy AI core modules
COPY ai_core ./ai_core

# Set ownership
RUN chown -R vcai:vcai /app

# Switch to non-root user
USER vcai

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "dev-api-server.py"]

# =============================================================================
# Development Stage
# =============================================================================
FROM app AS development

USER root
RUN pip install -r requirements-test.txt
USER vcai

# =============================================================================
# Production Stage
# =============================================================================
FROM app AS production

# Add production optimizations
ENV ENVIRONMENT=production

# Labels
LABEL org.opencontainers.image.title="VCAI Service" \
      org.opencontainers.image.description="Version Control AI Service for Three-Version Pipeline" \
      org.opencontainers.image.vendor="CodeRev Platform" \
      org.opencontainers.image.version="1.0.0"
