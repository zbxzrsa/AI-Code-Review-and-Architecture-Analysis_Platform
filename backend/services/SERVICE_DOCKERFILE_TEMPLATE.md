# Service Dockerfile Template

Use this template as a starting point for creating Dockerfiles for backend services.

## Standard Dockerfile

```dockerfile
# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY src/ ./src/
COPY requirements.txt .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Required Health Endpoints

Every service must implement these health endpoints in `src/main.py`:

```python
@app.get("/health/live", tags=["Health"])
async def liveness():
    """Liveness probe - is the service running?"""
    return {"status": "alive"}

@app.get("/health/ready", tags=["Health"])
async def readiness():
    """Readiness probe - is the service ready to accept traffic?"""
    # Check database, cache, and other dependencies
    return {"status": "ready"}

@app.get("/health", tags=["Health"])
async def health():
    """General health check - for load balancers"""
    return {"status": "healthy", "service": "service-name", "version": "1.0.0"}
```

## Standard requirements.txt

```txt
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
pydantic>=2.5.0
pydantic-settings>=2.1.0
sqlalchemy>=2.0.0
asyncpg>=0.29.0
redis>=5.0.0
prometheus-client>=0.19.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
httpx>=0.26.0
structlog>=24.1.0
```

## Directory Structure

```
service-name/
├── Dockerfile
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── main.py          # FastAPI app
│   ├── config.py        # Settings
│   ├── database.py      # Database connection
│   ├── models.py        # SQLAlchemy models
│   ├── schemas.py       # Pydantic schemas
│   ├── routers/         # API routers
│   │   ├── __init__.py
│   │   └── *.py
│   ├── services/        # Business logic
│   │   ├── __init__.py
│   │   └── *.py
│   └── middleware/      # Custom middleware
│       ├── __init__.py
│       └── *.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    └── test_*.py
```
