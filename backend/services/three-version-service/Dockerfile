# Three-Version Evolution Service Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy AI core module
COPY ../../../ai_core /app/ai_core

# Copy service code
COPY . /app/services/three-version-service

WORKDIR /app/services/three-version-service

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8010

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8010/api/v1/evolution/health || exit 1

# Run the service
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
