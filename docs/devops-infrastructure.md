# DevOps & Infrastructure Guide

## Overview

Comprehensive Docker Compose setup with all microservices, databases, message queues, and observability stack for local development and production deployment.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend (React 18)                       │
│                      Port 3000                               │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                   API Gateway (Nginx)                        │
│                  Ports 80, 443                               │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────┐  ┌────────▼────┐  ┌───────▼────┐
│Auth Service│  │Project Svc  │  │Analysis Svc│
│Port 8001   │  │Port 8002    │  │Port 8003   │
└────────────┘  └─────────────┘  └────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────┐  ┌────────▼────┐  ┌───────▼────┐
│AI Orch      │  │Version Ctrl │  │Comparison  │
│Port 8004   │  │Port 8005    │  │Port 8006   │
└────────────┘  └─────────────┘  └────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────┐  ┌────────▼────┐  ┌───────▼────┐
│PostgreSQL  │  │Redis        │  │Neo4j       │
│Port 5432   │  │Port 6379    │  │Port 7687   │
└────────────┘  └─────────────┘  └────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────┐  ┌────────▼────┐  ┌───────▼────┐
│MinIO (S3)  │  │Kafka        │  │OPA         │
│Port 9000   │  │Port 9092    │  │Port 8181   │
└────────────┘  └─────────────┘  └────────────┘
```

---

## Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 16GB RAM minimum
- 50GB disk space

### Start Development Environment

```bash
# Clone repository
git clone <repo-url>
cd AI-Code-Review-and-Architecture-Analysis_Platform

# Set environment variables
cp .env.example .env

# Start all services
docker-compose up -d

# Wait for services to be healthy
docker-compose ps

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Clean up volumes
docker-compose down -v
```

---

## Services Overview

### Databases

#### PostgreSQL (Port 5432)

- **Image**: postgres:16-alpine
- **User**: coderev
- **Password**: dev_password
- **Database**: code_review_platform
- **Schemas**: auth, projects, experiments_v1, production, quarantine, providers, audits
- **Initialization**: Auto-loads all schema files

#### Redis (Port 6379)

- **Image**: redis:7-alpine
- **Features**: Persistence enabled (AOF)
- **Use Cases**: Caching, sessions, rate limiting, queues

#### Neo4j (Port 7687)

- **Image**: neo4j:5-community
- **User**: neo4j
- **Password**: dev_password
- **Plugins**: APOC, Graph Data Science
- **HTTP**: Port 7474
- **Bolt**: Port 7687

#### MinIO (Port 9000)

- **Image**: minio/minio:latest
- **User**: admin
- **Password**: dev_password
- **Console**: Port 9001
- **Use Cases**: S3-compatible object storage

### Message Queue

#### Kafka (Port 9092)

- **Image**: bitnami/kafka:latest
- **Configuration**: Single broker with controller
- **Use Cases**: Event streaming, async processing

### Backend Services

#### Auth Service (Port 8001)

- JWT token management
- User authentication
- Session management
- Role-based access control

#### Project Service (Port 8002)

- Project management
- Version control
- Baseline management
- Policy enforcement

#### Analysis Service (Port 8003)

- Code analysis execution
- Artifact storage
- Neo4j integration
- S3 report storage

#### AI Orchestrator (Port 8004)

- AI model routing
- Provider management
- Experiment orchestration
- GPU support

#### Version Control Service (Port 8005)

- Experiment lifecycle
- Promotion workflows
- OPA policy integration
- Kafka event publishing

#### Comparison Service (Port 8006)

- A/B testing
- Statistical analysis
- Metrics comparison

#### Provider Service (Port 8007)

- Provider management
- API key encryption
- Quota enforcement
- Health monitoring

### Message Workers

#### Celery Worker

- Async task processing
- 2 replicas by default
- Concurrency: 4

#### Celery Beat

- Scheduled tasks
- Periodic jobs

### Policy Engine

#### OPA (Port 8181)

- Policy-based access control
- Quota enforcement
- Alert rules
- Audit compliance

### Observability

#### Prometheus (Port 9090)

- Metrics collection
- Time-series database
- Scrapes all services

#### Grafana (Port 3001)

- Visualization
- Dashboards
- Alerting

#### Loki (Port 3100)

- Log aggregation
- Structured logging

#### Tempo (Port 3200)

- Distributed tracing
- OTLP support

#### OpenTelemetry Collector (Port 4317/4318)

- Metrics collection
- Log forwarding
- Trace processing

### API Gateway

#### Nginx (Ports 80, 443)

- Request routing
- Load balancing
- SSL/TLS termination
- Rate limiting

### Frontend

#### React App (Port 3000)

- React 18 + TypeScript
- Ant Design 5
- Monaco Editor
- Real-time collaboration

---

## Environment Variables

### Database

```bash
DATABASE_URL=postgresql://coderev:dev_password@postgres:5432/code_review_platform
REDIS_URL=redis://redis:6379/0
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=dev_password
```

### Storage

```bash
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=admin
S3_SECRET_KEY=dev_password
S3_BUCKET=code-review-artifacts
```

### AI Providers

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
HUGGINGFACE_API_KEY=hf-...
```

### Security

```bash
JWT_SECRET_KEY=dev_secret_key_change_in_production
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
```

### Message Queue

```bash
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
```

### Policy Engine

```bash
OPA_URL=http://opa:8181
```

---

## Port Mapping

| Service            | Port      | Purpose        |
| ------------------ | --------- | -------------- |
| PostgreSQL         | 5432      | Database       |
| Redis              | 6379      | Cache/Queue    |
| Neo4j Bolt         | 7687      | Graph DB       |
| Neo4j HTTP         | 7474      | Graph DB UI    |
| Kafka              | 9092      | Message Queue  |
| MinIO API          | 9000      | Object Storage |
| MinIO Console      | 9001      | Storage UI     |
| Auth Service       | 8001      | API            |
| Project Service    | 8002      | API            |
| Analysis Service   | 8003      | API            |
| AI Orchestrator    | 8004      | API            |
| Version Control    | 8005      | API            |
| Comparison Service | 8006      | API            |
| Provider Service   | 8007      | API            |
| OPA                | 8181      | Policy Engine  |
| Prometheus         | 9090      | Metrics        |
| Grafana            | 3001      | Dashboards     |
| Loki               | 3100      | Logs           |
| Tempo              | 3200      | Traces         |
| OTEL Collector     | 4317/4318 | Telemetry      |
| Nginx              | 80/443    | Gateway        |
| Frontend           | 3000      | Web UI         |

---

## Health Checks

All services include health checks:

```bash
# Check service health
docker-compose ps

# Check specific service
docker-compose exec postgres pg_isready -U coderev
docker-compose exec redis redis-cli ping
docker-compose exec neo4j cypher-shell -u neo4j -p dev_password 'RETURN 1'
```

---

## Logs

### View Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs auth-service

# Follow logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail 100

# Since specific time
docker-compose logs --since 2024-12-02T10:00:00
```

### Log Locations

- **Container logs**: `docker-compose logs`
- **Loki logs**: http://localhost:3100
- **Grafana logs**: http://localhost:3001

---

## Monitoring

### Prometheus

Access: http://localhost:9090

**Targets**:

- All microservices (port 9090)
- Prometheus itself
- OTEL Collector

### Grafana

Access: http://localhost:3001

- **Username**: admin
- **Password**: admin

**Dashboards**:

- Service Health
- Request Metrics
- Database Performance
- Cache Statistics

### Loki

Access: http://localhost:3100

**Log Queries**:

```
{job="auth-service"}
{job="analysis-service"} | json
{level="error"}
```

### Tempo

Access: http://localhost:3200

**Trace Queries**:

- Service traces
- Latency analysis
- Error traces

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs <service>

# Check dependencies
docker-compose ps

# Restart service
docker-compose restart <service>

# Rebuild image
docker-compose build --no-cache <service>
```

### Database Connection Issues

```bash
# Check PostgreSQL
docker-compose exec postgres psql -U coderev -d code_review_platform

# Check Redis
docker-compose exec redis redis-cli ping

# Check Neo4j
docker-compose exec neo4j cypher-shell -u neo4j -p dev_password 'RETURN 1'
```

### Memory Issues

```bash
# Check resource usage
docker stats

# Increase Docker memory
# Edit Docker Desktop settings or docker-compose resource limits
```

### Port Conflicts

```bash
# Find process using port
lsof -i :8001

# Change port in docker-compose.yml
# Rebuild and restart
docker-compose up -d --build
```

---

## Performance Tuning

### Database

```sql
-- Analyze query performance
EXPLAIN ANALYZE SELECT ...;

-- Create indexes
CREATE INDEX idx_name ON table(column);

-- Vacuum
VACUUM ANALYZE;
```

### Redis

```bash
# Monitor commands
MONITOR

# Get stats
INFO stats

# Optimize memory
CONFIG SET maxmemory-policy allkeys-lru
```

### Kafka

```bash
# Check topics
kafka-topics --list --bootstrap-server localhost:9092

# Monitor lag
kafka-consumer-groups --bootstrap-server localhost:9092 --list
```

---

## Backup & Recovery

### Database Backup

```bash
# Backup PostgreSQL
docker-compose exec postgres pg_dump -U coderev code_review_platform > backup.sql

# Restore PostgreSQL
docker-compose exec -T postgres psql -U coderev code_review_platform < backup.sql
```

### Volume Backup

```bash
# Backup volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/postgres_backup.tar.gz -C /data .

# Restore volumes
docker run --rm -v postgres_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/postgres_backup.tar.gz -C /data
```

---

## Production Deployment

### Kubernetes

See `kubernetes/` directory for K8s manifests.

### Environment Setup

```bash
# Production environment
cp .env.production .env

# Update secrets
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export JWT_SECRET_KEY=<strong-random-key>
```

### Security

- Use secrets management (Vault, AWS Secrets Manager)
- Enable TLS/HTTPS
- Configure firewall rules
- Set resource limits
- Enable audit logging

---

## Maintenance

### Regular Tasks

- Monitor disk space
- Review logs
- Update dependencies
- Backup databases
- Rotate secrets
- Performance tuning

### Scaling

```bash
# Scale service
docker-compose up -d --scale celery-worker=4

# Load balancing
# Configure Nginx upstream
```

---

## Future Enhancements

- [ ] Kubernetes deployment
- [ ] Helm charts
- [ ] CI/CD pipeline
- [ ] Automated backups
- [ ] Disaster recovery
- [ ] Multi-region deployment
