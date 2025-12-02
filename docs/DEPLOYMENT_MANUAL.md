# Operations & Deployment Manual

**Version**: 2.0  
**Date**: December 2, 2025

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Development Setup](#2-development-setup)
3. [Production Deployment](#3-production-deployment)
4. [Ollama Setup](#4-ollama-setup)
5. [Configuration](#5-configuration)
6. [Monitoring](#6-monitoring)
7. [Troubleshooting](#7-troubleshooting)
8. [Rollback Procedures](#8-rollback-procedures)

---

## 1. Quick Start

### Prerequisites

- Docker 24.0+
- Docker Compose 2.20+
- 16GB RAM minimum (32GB recommended for local AI)
- NVIDIA GPU (optional, for faster AI inference)

### 5-Minute Setup

```bash
# Clone repository
git clone https://github.com/your-org/ai-code-review-platform.git
cd ai-code-review-platform

# Start infrastructure services
docker-compose up -d postgres redis neo4j minio kafka

# Wait for services to be healthy
docker-compose ps

# Start Ollama with CodeLlama
docker-compose up -d ollama
docker exec -it platform-ollama ollama pull codellama:13b

# Start application services
docker-compose up -d

# Verify all services
docker-compose ps
```

### Access URLs

| Service       | URL                   | Credentials        |
| ------------- | --------------------- | ------------------ |
| Frontend      | http://localhost:3000 | -                  |
| API           | http://localhost:8000 | -                  |
| Grafana       | http://localhost:3002 | admin/admin        |
| Prometheus    | http://localhost:9090 | -                  |
| Neo4j Browser | http://localhost:7474 | neo4j/password     |
| MinIO Console | http://localhost:9001 | admin/dev_password |

---

## 2. Development Setup

### 2.1 Environment Setup

```bash
# Create .env file
cp .env.example .env

# Edit configuration
nano .env
```

### 2.2 Required Environment Variables

```bash
# Database
DATABASE_URL=postgresql://coderev:dev_password@localhost:5432/code_review_platform

# Redis
REDIS_URL=redis://localhost:6379/0

# AI Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=codellama:13b

# Security (change in production!)
JWT_SECRET_KEY=dev-secret-key-change-in-production
CSRF_SECRET_KEY=csrf-secret-key-change-in-production

# Environment
ENVIRONMENT=development
DEBUG=true
```

### 2.3 Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Run tests
npm run test

# Build for production
npm run build
```

### 2.4 Backend Development

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn src.main:app --reload --port 8000
```

---

## 3. Production Deployment

### 3.1 Kubernetes Deployment

```bash
# Create namespaces
kubectl apply -f kubernetes/namespaces.yaml

# Deploy secrets
kubectl create secret generic db-credentials \
  --from-literal=username=coderev \
  --from-literal=password=secure-password \
  -n platform-infrastructure

kubectl create secret generic jwt-secret \
  --from-literal=secret-key=$(openssl rand -base64 32) \
  -n platform-v2-stable

# Deploy infrastructure
kubectl apply -f kubernetes/infrastructure/

# Deploy services
kubectl apply -f kubernetes/deployments/

# Deploy ingress
kubectl apply -f kubernetes/ingress/

# Verify deployment
kubectl get pods -n platform-v2-stable
```

### 3.2 Helm Deployment

```bash
# Add Helm repo
helm repo add ai-code-review https://charts.example.com
helm repo update

# Install with custom values
helm install ai-code-review ai-code-review/platform \
  --namespace platform \
  --create-namespace \
  --values values-production.yaml

# Upgrade
helm upgrade ai-code-review ai-code-review/platform \
  --namespace platform \
  --values values-production.yaml
```

### 3.3 Production Checklist

- [ ] Change all default passwords
- [ ] Set `DEBUG=false`
- [ ] Configure HTTPS/TLS
- [ ] Set up backup strategy
- [ ] Configure monitoring alerts
- [ ] Set up log aggregation
- [ ] Configure rate limiting
- [ ] Enable CSRF protection
- [ ] Set secure cookie flags
- [ ] Configure network policies

---

## 4. Ollama Setup

### 4.1 Docker Setup (Recommended)

```bash
# CPU-only
docker run -d \
  --name ollama \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama

# With NVIDIA GPU
docker run -d \
  --name ollama \
  --gpus all \
  -p 11434:11434 \
  -v ollama_data:/root/.ollama \
  ollama/ollama
```

### 4.2 Pull Models

```bash
# Best for code review (requires 20GB+ VRAM)
docker exec ollama ollama pull codellama:34b

# Good balance (requires 8GB+ VRAM)
docker exec ollama ollama pull codellama:13b

# Fast/lightweight (requires 4GB+ VRAM)
docker exec ollama ollama pull codellama:7b

# Alternative models
docker exec ollama ollama pull deepseek-coder:33b
docker exec ollama ollama pull mistral:7b
```

### 4.3 Model Selection Guide

| Model              | VRAM | Quality | Speed     | Use Case     |
| ------------------ | ---- | ------- | --------- | ------------ |
| codellama:7b       | 4GB  | 70%     | Fast      | Quick checks |
| codellama:13b      | 8GB  | 80%     | Medium    | Development  |
| codellama:34b      | 20GB | 85%     | Slow      | Production   |
| deepseek-coder:33b | 20GB | 88%     | Slow      | High quality |
| llama3:70b         | 40GB | 90%     | Very slow | Best quality |

### 4.4 Verify Ollama

```bash
# Health check
curl http://localhost:11434/api/tags

# Test generation
curl http://localhost:11434/api/generate -d '{
  "model": "codellama:13b",
  "prompt": "Write a Python function to sort a list",
  "stream": false
}'
```

---

## 5. Configuration

### 5.1 AI Provider Configuration

```yaml
# config/ai-providers.yaml
providers:
  primary:
    type: ollama
    endpoint: http://ollama:11434
    model: codellama:34b
    timeout: 120

  secondary:
    type: ollama
    endpoint: http://ollama:11434
    model: codellama:13b
    timeout: 60

  fallback:
    type: openai # Only if user provides API key
    model: gpt-4
    enabled: false
```

### 5.2 Rate Limiting Configuration

```yaml
# config/rate-limits.yaml
rate_limits:
  auth:
    login:
      per_minute: 5
      per_hour: 20
    register:
      per_minute: 3
      per_hour: 10

  api:
    analyze:
      per_minute: 10
      per_hour: 100
    default:
      per_minute: 60
      per_hour: 1000
```

### 5.3 Security Configuration

```yaml
# config/security.yaml
security:
  jwt:
    algorithm: HS256
    access_token_expire_minutes: 15
    refresh_token_expire_days: 7

  cookies:
    secure: true
    httponly: true
    samesite: lax

  csrf:
    enabled: true
    header_name: X-CSRF-Token
```

---

## 6. Monitoring

### 6.1 Health Checks

```bash
# Service health
curl http://localhost:8001/health/live  # Auth
curl http://localhost:8003/health/live  # Analysis
curl http://localhost:8004/health/live  # AI Orchestrator

# Ollama health
curl http://localhost:11434/api/tags

# Database health
docker exec platform-postgres pg_isready
```

### 6.2 Prometheus Metrics

Key metrics to monitor:

```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])

# Latency P95
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# AI provider health
ai_provider_healthy{provider="ollama"}

# Rate limit hits
rate(rate_limit_exceeded_total[5m])
```

### 6.3 Grafana Dashboards

Access Grafana at http://localhost:3002

Pre-configured dashboards:

- **Overview**: System health, request rates, errors
- **AI Providers**: Model performance, latency, costs
- **Security**: Auth failures, rate limits, CSRF blocks
- **Infrastructure**: Database, Redis, Kafka metrics

### 6.4 Log Aggregation

```bash
# View logs
docker-compose logs -f auth-service
docker-compose logs -f analysis-service
docker-compose logs -f ai-orchestrator

# Search in Loki (via Grafana)
{job="auth-service"} |= "error"
{job="analysis-service"} | json | level="error"
```

---

## 7. Troubleshooting

### 7.1 Common Issues

#### Ollama Not Responding

```bash
# Check status
docker logs platform-ollama

# Restart
docker restart platform-ollama

# Check memory
docker stats platform-ollama
```

#### Database Connection Failed

```bash
# Check PostgreSQL
docker exec platform-postgres pg_isready

# Check logs
docker logs platform-postgres

# Reset database
docker-compose down -v postgres
docker-compose up -d postgres
```

#### Rate Limit Exceeded

```bash
# Check Redis rate limit keys
docker exec platform-redis redis-cli keys "ratelimit:*"

# Clear rate limits (development only)
docker exec platform-redis redis-cli flushdb
```

#### CSRF Token Invalid

```bash
# Verify CSRF cookie is set
# In browser DevTools > Application > Cookies

# Check CSRF header is sent
# In browser DevTools > Network > Request Headers
# X-CSRF-Token should be present
```

### 7.2 Performance Issues

```bash
# Check container resources
docker stats

# Check slow queries
docker exec platform-postgres psql -U coderev -c "
  SELECT query, calls, mean_time
  FROM pg_stat_statements
  ORDER BY mean_time DESC
  LIMIT 10;
"

# Check Redis memory
docker exec platform-redis redis-cli info memory
```

### 7.3 Debug Mode

```bash
# Enable debug logging
export DEBUG=true
export LOG_LEVEL=DEBUG

# Restart services
docker-compose restart auth-service analysis-service
```

---

## 8. Rollback Procedures

### 8.1 Application Rollback

```bash
# Kubernetes rollback
kubectl rollout undo deployment/auth-service -n platform-v2-stable
kubectl rollout undo deployment/analysis-service -n platform-v2-stable

# Docker Compose rollback
docker-compose pull --ignore-pull-failures
git checkout v1.2.3  # Previous version
docker-compose up -d

# Verify rollback
docker-compose ps
curl http://localhost:8000/health
```

### 8.2 Database Rollback

```bash
# Restore from backup
docker exec platform-postgres psql -U coderev -c "
  DROP DATABASE code_review_platform;
  CREATE DATABASE code_review_platform;
"
docker exec -i platform-postgres psql -U coderev < backup.sql

# Or use point-in-time recovery
# (Requires WAL archiving configured)
```

### 8.3 AI Model Rollback

```bash
# List available models
docker exec platform-ollama ollama list

# Switch to previous model
# Update config/ai-providers.yaml
# model: codellama:13b  # Previous working model

# Restart AI orchestrator
docker-compose restart ai-orchestrator
```

### 8.4 Emergency Procedures

```bash
# Complete system stop
docker-compose down

# Data backup
docker exec platform-postgres pg_dump -U coderev > emergency_backup.sql
docker exec platform-redis redis-cli BGSAVE

# Minimal recovery (database + auth only)
docker-compose up -d postgres redis auth-service

# Verify critical path
curl http://localhost:8001/health/live
```

---

## Appendix: Docker Compose Commands

```bash
# Start all services
docker-compose up -d

# Start specific services
docker-compose up -d postgres redis ollama

# View logs
docker-compose logs -f [service]

# Restart service
docker-compose restart [service]

# Stop all
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild images
docker-compose build --no-cache

# Scale service
docker-compose up -d --scale analysis-service=3
```

---

_Manual maintained by DevOps Team_  
_Last updated: December 2, 2025_
