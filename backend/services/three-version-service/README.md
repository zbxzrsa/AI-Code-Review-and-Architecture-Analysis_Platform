# Three-Version Evolution Service

REST API service for managing the three-version self-evolution cycle.

## Overview

This service orchestrates the V1/V2/V3 spiral evolution:

| Version         | Role            | Access |
| --------------- | --------------- | ------ |
| **V1** (New)    | Experimentation | Admin  |
| **V2** (Stable) | Production      | Users  |
| **V3** (Old)    | Quarantine      | Admin  |

## Quick Start

### Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run service
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```

### Docker

```bash
docker build -t three-version-service .
docker run -p 8010:8010 three-version-service
```

### Docker Compose

```bash
# From project root
docker-compose up -d three-version-service
```

## API Endpoints

| Method | Endpoint                       | Description         |
| ------ | ------------------------------ | ------------------- |
| GET    | `/api/v1/evolution/status`     | Cycle status        |
| POST   | `/api/v1/evolution/start`      | Start cycle         |
| POST   | `/api/v1/evolution/stop`       | Stop cycle          |
| POST   | `/api/v1/evolution/v1/errors`  | Report V1 error     |
| POST   | `/api/v1/evolution/promote`    | V1→V2 promotion     |
| POST   | `/api/v1/evolution/degrade`    | V2→V3 degradation   |
| POST   | `/api/v1/evolution/reeval`     | V3→V1 re-evaluation |
| GET    | `/api/v1/evolution/health`     | Health check        |
| GET    | `/api/v1/evolution/prometheus` | Prometheus metrics  |

## API Documentation

Interactive API docs available at:

- **Swagger UI**: http://localhost:8010/docs
- **ReDoc**: http://localhost:8010/redoc

## Example Usage

### Start Cycle

```bash
curl -X POST http://localhost:8010/api/v1/evolution/start
```

### Report V1 Error

```bash
curl -X POST http://localhost:8010/api/v1/evolution/v1/errors \
  -H "Content-Type: application/json" \
  -d '{
    "tech_id": "mqa_123",
    "tech_name": "Multi-Query Attention",
    "error_type": "compatibility",
    "description": "Incompatible with transformer layers"
  }'
```

### Trigger Promotion

```bash
curl -X POST http://localhost:8010/api/v1/evolution/promote \
  -H "Content-Type: application/json" \
  -d '{"tech_id": "mqa_123"}'
```

### Get Status

```bash
curl http://localhost:8010/api/v1/evolution/status
```

## Environment Variables

| Variable                  | Required | Default     | Description           |
| ------------------------- | -------- | ----------- | --------------------- |
| `DATABASE_URL`            | Yes      | -           | PostgreSQL connection |
| `REDIS_URL`               | Yes      | -           | Redis connection      |
| `KAFKA_BOOTSTRAP_SERVERS` | No       | -           | Kafka servers         |
| `ENVIRONMENT`             | No       | development | Environment name      |
| `LOG_LEVEL`               | No       | INFO        | Logging level         |

## Metrics

Prometheus metrics available at `/api/v1/evolution/prometheus`:

- `evolution_cycle_running` - Cycle status
- `evolution_cycles_total` - Completed cycles
- `v1_experiments_total` - V1 experiments
- `v1_errors_total` - V1 errors by type
- `v2_fixes_total` - V2 fixes by status
- `v2_fix_success_rate` - Fix success rate
- `ai_instance_status` - AI status per version
- `pending_promotions` - Pending promotions
- `quarantine_technologies_total` - Quarantined count

## Health Check

```bash
curl http://localhost:8010/api/v1/evolution/health
```

Response:

```json
{
  "status": "healthy",
  "service": "three-version-evolution",
  "cycle_running": true,
  "timestamp": "2024-12-04T10:00:00Z"
}
```

## Development

### Run Tests

```bash
pytest tests/ -v
```

### Code Style

```bash
black .
isort .
ruff check .
```

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI Application                │
├─────────────────────────────────────────────────────┤
│  API Routes (/api/v1/evolution/*)                   │
├─────────────────────────────────────────────────────┤
│  EnhancedSelfEvolutionCycle                         │
│  ├── SpiralEvolutionManager                         │
│  │   ├── DualAICoordinator                          │
│  │   ├── CrossVersionFeedbackSystem                 │
│  │   └── V3ComparisonEngine                         │
│  └── VersionManager                                 │
├─────────────────────────────────────────────────────┤
│  PostgreSQL │ Redis │ Kafka                         │
└─────────────────────────────────────────────────────┘
```

## Related

- [Three-Version Evolution Docs](../../../docs/three-version-evolution.md)
- [AI Core Package](../../../ai_core/three_version_cycle/)
- [Grafana Dashboard](../../../monitoring/grafana/provisioning/dashboards/three-version-evolution.json)
