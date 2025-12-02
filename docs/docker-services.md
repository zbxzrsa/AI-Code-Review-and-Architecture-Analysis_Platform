# Backend Services Dockerization Guide

## Overview

This document describes the Docker containerization strategy for all backend microservices in the AI Code Review Platform.

## Services

| Service                 | Port | Description                                    |
| ----------------------- | ---- | ---------------------------------------------- |
| auth-service            | 8001 | Authentication, authorization, user management |
| project-service         | 8002 | Project and file management                    |
| analysis-service        | 8003 | Code analysis orchestration                    |
| ai-orchestrator         | 8004 | AI model routing and coordination              |
| version-control-service | 8005 | Experiment and version management              |
| comparison-service      | 8006 | A/B testing and comparison                     |
| provider-service        | 8007 | AI provider management and quotas              |

## Dockerfile Standards

All services follow these standards:

### Multi-Stage Build

```dockerfile
# Stage 1: Builder
FROM python:3.11-slim as builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Production
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /home/appuser/.local
COPY --chown=appuser:appuser src/ ./src/
```

### Security

- **Non-root user**: All containers run as `appuser` (UID 1000)
- **Read-only filesystem**: Where possible
- **No privilege escalation**: `allowPrivilegeEscalation: false`
- **Dropped capabilities**: All capabilities dropped

### Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1
```

### Environment Variables

| Variable       | Description           | Default     |
| -------------- | --------------------- | ----------- |
| `SERVICE_NAME` | Service identifier    | Required    |
| `PORT`         | HTTP port             | 8000        |
| `DATABASE_URL` | PostgreSQL connection | Required    |
| `REDIS_URL`    | Redis connection      | Required    |
| `LOG_LEVEL`    | Logging level         | INFO        |
| `ENVIRONMENT`  | Runtime environment   | development |

## Quick Start

### Build All Services

```bash
# From project root
docker-compose -f docker/docker-compose.services.yml build
```

### Run All Services

```bash
# Start services
docker-compose -f docker/docker-compose.services.yml up -d

# View logs
docker-compose -f docker/docker-compose.services.yml logs -f

# Stop services
docker-compose -f docker/docker-compose.services.yml down
```

### Build Individual Service

```bash
cd backend/services/auth-service
docker build -t auth-service:latest .
```

## Health Check Endpoints

Each service exposes:

| Endpoint          | Purpose         | Used By             |
| ----------------- | --------------- | ------------------- |
| `/health`         | Liveness check  | K8s liveness probe  |
| `/ready`          | Readiness check | K8s readiness probe |
| `/health/startup` | Startup check   | K8s startup probe   |

### Health Check Response

```json
{
  "status": "healthy",
  "service": "auth-service",
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime_seconds": 3600,
  "components": [
    {
      "name": "database",
      "status": "healthy",
      "latency_ms": 5.2
    },
    {
      "name": "redis",
      "status": "healthy",
      "latency_ms": 1.1
    }
  ]
}
```

## Graceful Shutdown

Services handle `SIGTERM` for graceful shutdown:

1. Health checks return 503
2. Stop accepting new requests
3. Wait for in-flight requests (15s default)
4. Close database connections
5. Exit cleanly

```dockerfile
STOPSIGNAL SIGTERM
```

## Resource Limits

### Development

| Service                 | CPU | Memory |
| ----------------------- | --- | ------ |
| auth-service            | 0.5 | 512M   |
| project-service         | 0.5 | 512M   |
| analysis-service        | 1.0 | 1G     |
| ai-orchestrator         | 2.0 | 4G     |
| version-control-service | 0.5 | 512M   |
| comparison-service      | 0.5 | 512M   |
| provider-service        | 0.5 | 512M   |

### Production (Kubernetes)

| Service                 | Request CPU | Limit CPU | Request Mem | Limit Mem |
| ----------------------- | ----------- | --------- | ----------- | --------- |
| auth-service            | 100m        | 500m      | 256Mi       | 512Mi     |
| project-service         | 100m        | 500m      | 256Mi       | 512Mi     |
| analysis-service        | 250m        | 1000m     | 512Mi       | 1Gi       |
| ai-orchestrator         | 500m        | 2000m     | 1Gi         | 4Gi       |
| version-control-service | 100m        | 500m      | 256Mi       | 512Mi     |
| comparison-service      | 100m        | 500m      | 256Mi       | 512Mi     |
| provider-service        | 100m        | 500m      | 256Mi       | 512Mi     |

## Kubernetes Deployment

### Apply Configurations

```bash
# Create namespace and configs
kubectl apply -f kubernetes/config/secrets-configmaps.yaml

# Deploy services
kubectl apply -f kubernetes/services/auth-service.yaml
kubectl apply -f kubernetes/services/all-services.yaml
```

### Check Status

```bash
# View deployments
kubectl get deployments -n ai-codereview

# View pods
kubectl get pods -n ai-codereview

# Check logs
kubectl logs -f deployment/auth-service -n ai-codereview
```

## Environment Configuration

### Development (.env)

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres

# JWT
JWT_SECRET_KEY=dev-secret-key-change-in-production

# AI Providers (optional for dev)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# Environment
ENVIRONMENT=development
LOG_LEVEL=DEBUG
```

### Production

Use Kubernetes secrets for sensitive values:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: database-credentials
type: Opaque
stringData:
  auth-db-url: "postgresql://..."
```

## Network Architecture

```
                                    ┌─────────────────┐
                                    │   API Gateway   │
                                    │    (Nginx)      │
                                    └────────┬────────┘
                                             │
                ┌────────────────────────────┼────────────────────────────┐
                │                            │                            │
        ┌───────▼───────┐           ┌───────▼───────┐           ┌───────▼───────┐
        │ Auth Service  │           │Project Service│           │Analysis Service│
        │   :8001       │           │   :8002       │           │   :8003        │
        └───────┬───────┘           └───────┬───────┘           └───────┬───────┘
                │                            │                            │
                └────────────────────────────┼────────────────────────────┘
                                             │
                                    ┌────────▼────────┐
                                    │ AI Orchestrator │
                                    │   :8004         │
                                    └────────┬────────┘
                                             │
        ┌────────────────────────────────────┼────────────────────────────────────┐
        │                                    │                                    │
┌───────▼───────┐               ┌───────────▼───────────┐               ┌───────▼───────┐
│Version Control│               │ Comparison Service    │               │Provider Service│
│   :8005       │               │   :8006               │               │   :8007        │
└───────────────┘               └───────────────────────┘               └───────────────┘
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker logs auth-service

# Check health
docker exec auth-service curl -f http://localhost:8000/health
```

### Database Connection Issues

```bash
# Test connectivity
docker exec auth-service python -c "import asyncpg; print('OK')"

# Check environment
docker exec auth-service env | grep DATABASE
```

### Memory Issues

```bash
# Check resource usage
docker stats auth-service

# Increase limits in docker-compose.yml
```

## Image Optimization

### Current Image Sizes

| Service          | Size   |
| ---------------- | ------ |
| auth-service     | ~180MB |
| project-service  | ~160MB |
| analysis-service | ~200MB |
| ai-orchestrator  | ~250MB |
| Other services   | ~150MB |

### Optimization Tips

1. Use multi-stage builds (already implemented)
2. Pin exact package versions
3. Use .dockerignore to exclude unnecessary files
4. Combine RUN commands to reduce layers

---

## Testing

### Unit Tests

```bash
# Run unit tests locally
cd backend
pytest tests/unit/ -v --cov=src

# Run in Docker
docker-compose -f docker/docker-compose.test.yml run --rm unit-tests
```

### Integration Tests

```bash
# Start test infrastructure
docker-compose -f docker/docker-compose.test.yml up -d test-postgres test-redis

# Run integration tests
docker-compose -f docker/docker-compose.test.yml run --rm integration-tests
```

### Load Testing

```bash
# Start services and load testing
docker-compose -f docker/docker-compose.test.yml --profile load-test up -d

# Access Locust UI
open http://localhost:8089

# Run headless load test
docker-compose -f docker/docker-compose.test.yml run --rm locust-master \
  --headless --users 100 --spawn-rate 10 --run-time 5m
```

### Test Configuration

```python
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
asyncio_mode = auto
addopts = -v --tb=short
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow running tests
```

---

## Security Scanning

### Trivy (Container Scanning)

```bash
# Scan a single image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy:latest image auth-service:latest

# Scan with severity filter
trivy image --severity HIGH,CRITICAL auth-service:latest

# Generate JSON report
trivy image --format json --output trivy-report.json auth-service:latest
```

### Snyk (Dependency Scanning)

```bash
# Install Snyk CLI
npm install -g snyk

# Authenticate
snyk auth

# Scan Python dependencies
snyk test --file=backend/requirements.txt

# Scan Docker image
snyk container test auth-service:latest
```

### Security Best Practices

| Practice                | Implementation                        |
| ----------------------- | ------------------------------------- |
| Non-root user           | `USER appuser` in Dockerfile          |
| Read-only filesystem    | K8s `readOnlyRootFilesystem: true`    |
| No privilege escalation | K8s `allowPrivilegeEscalation: false` |
| Minimal base image      | `python:3.11-slim`                    |
| No secrets in images    | Environment variables / K8s secrets   |
| Image signing           | Cosign in CI/CD                       |
| SBOM generation         | Anchore SBOM Action                   |

---

## CI/CD Integration

### Pipeline Overview

```bash
Code Push
    ↓
┌─────────────────────────────────────────┐
│  1. Lint & Unit Tests                   │
│     - ruff, black, isort                │
│     - pytest with coverage              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  2. Security Scanning                   │
│     - Gitleaks (secrets)                │
│     - Trivy (vulnerabilities)           │
│     - Snyk (dependencies)               │
│     - OWASP Dependency Check            │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  3. Build Docker Images                 │
│     - Multi-stage builds                │
│     - Push to GCR                       │
│     - Trivy container scan              │
│     - Generate SBOM                     │
│     - Sign with Cosign                  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  4. Integration Tests                   │
│     - Service-to-service tests          │
│     - Database integration              │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  5. Deploy                              │
│     - Staging (develop branch)          │
│     - Production (main branch)          │
│     - Health checks                     │
│     - Slack notification                │
└─────────────────────────────────────────┘
```

### Required Secrets

| Secret             | Description                     |
| ------------------ | ------------------------------- |
| `GCP_PROJECT_ID`   | Google Cloud project ID         |
| `GCR_JSON_KEY`     | Service account key for GCR     |
| `GKE_SA_KEY`       | Service account key for GKE     |
| `SNYK_TOKEN`       | Snyk API token                  |
| `SLACK_BOT_TOKEN`  | Slack bot token                 |
| `SLACK_CHANNEL_ID` | Slack channel for notifications |

### Manual Deployment

```bash
# Build images
docker-compose -f docker/docker-compose.services.yml build

# Push to registry
docker tag auth-service:latest gcr.io/PROJECT_ID/auth-service:latest
docker push gcr.io/PROJECT_ID/auth-service:latest

# Deploy to Kubernetes
kubectl apply -f kubernetes/services/
kubectl rollout status deployment/auth-service -n ai-codereview
```

---

## Kubernetes Deployment Guide

### Prerequisites

1. GKE cluster running
2. kubectl configured
3. Secrets created

### Step-by-Step Deployment

```bash
# 1. Create namespace
kubectl create namespace ai-codereview

# 2. Create secrets and configmaps
kubectl apply -f kubernetes/config/secrets-configmaps.yaml

# 3. Deploy infrastructure
kubectl apply -f kubernetes/infrastructure/

# 4. Deploy services
kubectl apply -f kubernetes/services/

# 5. Verify deployment
kubectl get pods -n ai-codereview
kubectl get services -n ai-codereview
```

### Scaling

```bash
# Manual scaling
kubectl scale deployment auth-service --replicas=5 -n ai-codereview

# HPA is configured automatically
kubectl get hpa -n ai-codereview
```

### Rolling Updates

```bash
# Update image
kubectl set image deployment/auth-service \
  auth-service=gcr.io/PROJECT_ID/auth-service:v2.0.0 \
  -n ai-codereview

# Check rollout status
kubectl rollout status deployment/auth-service -n ai-codereview

# Rollback if needed
kubectl rollout undo deployment/auth-service -n ai-codereview
```

---

## Health Check Troubleshooting

### Common Issues

| Issue                        | Cause                 | Solution                          |
| ---------------------------- | --------------------- | --------------------------------- |
| Health check timeout         | Service slow to start | Increase `initialDelaySeconds`    |
| Readiness fails              | Database not ready    | Check DB connection, add retry    |
| Liveness fails after running | Memory leak           | Check resource limits, add memory |
| Intermittent failures        | Network issues        | Check service discovery, DNS      |

### Debug Commands

```bash
# Check pod logs
kubectl logs -f deployment/auth-service -n ai-codereview

# Exec into pod
kubectl exec -it deployment/auth-service -n ai-codereview -- /bin/bash

# Test health endpoint manually
kubectl exec deployment/auth-service -n ai-codereview -- \
  curl -v http://localhost:8000/health

# Check events
kubectl get events -n ai-codereview --sort-by='.lastTimestamp'

# Describe pod for details
kubectl describe pod -l app=auth-service -n ai-codereview
```

### Health Check Response Codes

| Endpoint          | Success | Failure | Meaning                            |
| ----------------- | ------- | ------- | ---------------------------------- |
| `/health`         | 200     | 503     | Service is/isn't running           |
| `/ready`          | 200     | 503     | Service is/isn't ready for traffic |
| `/health/startup` | 200     | 503     | Service has/hasn't started         |

### Probe Configuration

```yaml
# Recommended probe settings
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10 # Wait before first check
  periodSeconds: 10 # Check every 10s
  timeoutSeconds: 5 # Timeout after 5s
  failureThreshold: 3 # Fail after 3 failures

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5 # Start checking sooner
  periodSeconds: 5 # Check more frequently
  timeoutSeconds: 3
  failureThreshold: 2 # Remove from LB faster

startupProbe:
  httpGet:
    path: /health/startup
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 30 # Allow 150s for startup
```

---

## Dockerfile Reference

### Multi-Stage Build Explained

```dockerfile
# =============================================================================
# Stage 1: Builder
# =============================================================================
# Purpose: Install dependencies in a full environment
# This stage is discarded after build

FROM python:3.11-slim as builder
WORKDIR /app

# Install build tools (needed for some Python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages to user directory
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 2: Production
# =============================================================================
# Purpose: Minimal runtime image
# Only includes what's needed to run

FROM python:3.11-slim
WORKDIR /app

# Install runtime dependencies only (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \        # PostgreSQL client library
    curl \          # For health checks
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# Copy only the installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local

# Copy application code
COPY --chown=appuser:appuser src/ ./src/

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \            # Don't buffer stdout/stderr
    PYTHONDONTWRITEBYTECODE=1 \     # Don't create .pyc files
    PYTHONPATH=/app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Graceful shutdown signal
STOPSIGNAL SIGTERM

# Run application
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```
