# File Index

Complete guide to all files in the AI Code Review Platform.

## Root Level Files

| File                 | Purpose                                          |
| -------------------- | ------------------------------------------------ |
| `README.md`          | Project overview and main documentation          |
| `QUICKSTART.md`      | 5-minute quick start guide                       |
| `PROJECT_SUMMARY.md` | Comprehensive project summary                    |
| `CONTRIBUTING.md`    | Contributing guidelines and development workflow |
| `CHANGELOG.md`       | Version history and release notes                |
| `LICENSE`            | MIT License                                      |
| `FILE_INDEX.md`      | This file - index of all files                   |
| `.gitignore`         | Git ignore rules                                 |
| `docker-compose.yml` | Local development with Docker Compose            |

## Backend Structure

### Shared Code (`backend/shared/`)

**Configuration** (`backend/shared/config/`)

- `settings.py` - Global settings and configuration
- `__init__.py` - Package initialization

**Data Models** (`backend/shared/models/`)

- `experiment.py` - Experiment, metrics, and quarantine models
- `__init__.py` - Package initialization

**Utilities** (`backend/shared/utils/`)

- `ai_client.py` - Unified AI client for multiple providers
- `__init__.py` - Package initialization

### V2 Production (`backend/v2-production/`)

**Main Application**

- `src/main.py` - FastAPI application entry point
- `src/config/settings.py` - V2-specific settings
- `src/config/__init__.py` - Config package init

**API Routers** (`src/routers/`)

- `code_review.py` - Code review endpoints
- `health.py` - Health check endpoints
- `metrics.py` - Metrics endpoints
- `__init__.py` - Routers package init

**Database** (`src/database/`)

- `connection.py` - Database connection and session management
- `__init__.py` - Database package init

**Middleware** (`src/middleware/`)

- `monitoring.py` - Request monitoring and metrics
- `slo.py` - SLO enforcement middleware
- `__init__.py` - Middleware package init

**Models** (`src/models/`)

- `__init__.py` - Models package init

**Utils** (`src/utils/`)

- `__init__.py` - Utils package init

**Configuration**

- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition

### V1 Experimentation (`backend/v1-experimentation/`)

**Main Application**

- `src/main.py` - FastAPI application entry point
- `src/config/settings.py` - V1-specific settings
- `src/config/__init__.py` - Config package init

**API Routers** (`src/routers/`)

- `experiments.py` - Experiment management endpoints
- `evaluation.py` - Evaluation and promotion endpoints
- `health.py` - Health check endpoints
- `__init__.py` - Routers package init

**Database** (`src/database/`)

- `connection.py` - Database connection and session management
- `__init__.py` - Database package init

**Middleware** (`src/middleware/`)

- `monitoring.py` - Request monitoring and metrics
- `__init__.py` - Middleware package init

**Models** (`src/models/`)

- `__init__.py` - Models package init

**Utils** (`src/utils/`)

- `__init__.py` - Utils package init

**Configuration**

- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition

### V3 Quarantine (`backend/v3-quarantine/`)

**Main Application**

- `src/main.py` - FastAPI application entry point
- `src/config/settings.py` - V3-specific settings
- `src/config/__init__.py` - Config package init

**API Routers** (`src/routers/`)

- `quarantine.py` - Quarantine record endpoints
- `health.py` - Health check endpoints
- `__init__.py` - Routers package init

**Database** (`src/database/`)

- `connection.py` - Database connection (read-only)
- `__init__.py` - Database package init

**Models** (`src/models/`)

- `__init__.py` - Models package init

**Configuration**

- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker image definition

## Kubernetes Configuration

### Namespaces (`kubernetes/namespaces/`)

- `namespaces.yaml` - Namespace definitions for all three versions

### Network Policies (`kubernetes/network-policies/`)

- `isolation.yaml` - Network policies enforcing version isolation

### Deployments (`kubernetes/deployments/`)

- `v2-deployment.yaml` - V2 production deployment with HPA
- `v1-deployment.yaml` - V1 experimentation deployment
- `v3-deployment.yaml` - V3 quarantine deployment
- `three-version-service.yaml` - Three-version evolution service

### Configuration (`kubernetes/config/`)

- `configmap.yaml` - ConfigMaps for all versions
- `secrets.yaml` - Secrets (API keys, credentials)

## Database

### Schemas (`database/schemas/`)

- `init.sql` - PostgreSQL schema initialization script
  - Creates `production` schema (V2)
  - Creates `experiments_v1` schema (V1)
  - Creates `quarantine` schema (V3)
  - Creates `audit` schema (shared)
  - Sets up permissions and views

## Monitoring

### Prometheus (`monitoring/prometheus/`)

- `prometheus.yml` - Prometheus configuration
- `alerts.yml` - Alert rules for all versions

### Grafana (`monitoring/grafana/`)

- `provisioning/datasources/prometheus.yaml` - Prometheus data source
- `provisioning/dashboards/dashboard-provider.yaml` - Dashboard provisioning

## Documentation

### Main Docs (`docs/`)

- `architecture.md` - Detailed architecture documentation

  - Three-version system explanation
  - Evolution cycle
  - Metrics and SLOs
  - Network isolation
  - Database schema isolation
  - Deployment strategies
  - Monitoring and observability

- `deployment.md` - Complete deployment guide

  - Local development with Docker Compose
  - Kubernetes deployment
  - Database setup
  - Monitoring setup
  - API testing
  - Troubleshooting
  - Production checklist

- `api-reference.md` - Complete API documentation

  - Base URLs and authentication
  - V2 Production endpoints
  - V1 Experimentation endpoints
  - V3 Quarantine endpoints
  - Error responses
  - Rate limiting
  - Pagination

- `operations.md` - Operations runbook

  - Daily operations
  - Health checks
  - Experiment management
  - Scaling operations
  - Database operations
  - Deployment operations
  - Troubleshooting
  - Incident response
  - Maintenance windows
  - Cost optimization
  - Security operations

- `three-version-evolution.md` - Three-version spiral evolution system
  - V1/V2/V3 architecture
  - Dual-AI per version (VC-AI + CR-AI)
  - 8-phase spiral evolution cycle
  - API reference
  - Python SDK usage
  - Monitoring and alerting
  - Deployment guide

## Three-Version Service (`backend/services/three-version-service/`)

- `api.py` - REST API endpoints
- `main.py` - FastAPI application
- `metrics.py` - Prometheus metrics
- `requirements.txt` - Dependencies
- `Dockerfile` - Container configuration
- `README.md` - Service documentation

## AI Core Three-Version (`ai_core/three_version_cycle/`)

- `cross_version_feedback.py` - V2 fixes V1 errors
- `v3_comparison_engine.py` - Comparison and exclusion
- `dual_ai_coordinator.py` - VCAI + CRAI management
- `spiral_evolution_manager.py` - 8-phase orchestration
- `version_manager.py` - V1/V2/V3 state management
- `version_ai_engine.py` - Version-specific AI engines
- `experiment_framework.py` - V1 experiments
- `self_evolution_cycle.py` - Self-evolution cycle

## File Organization Summary

```
AI-Code-Review-and-Architecture-Analysis_Platform/
├── Root Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── PROJECT_SUMMARY.md
│   ├── CONTRIBUTING.md
│   ├── CHANGELOG.md
│   ├── LICENSE
│   └── FILE_INDEX.md (this file)
│
├── Backend Services
│   ├── backend/shared/          (Shared code)
│   ├── backend/v2-production/   (Production API)
│   ├── backend/v1-experimentation/ (Experimentation API)
│   └── backend/v3-quarantine/   (Quarantine API)
│
├── Infrastructure
│   ├── kubernetes/              (K8s manifests)
│   ├── database/                (Database setup)
│   ├── monitoring/              (Prometheus & Grafana)
│   └── docker-compose.yml       (Local dev)
│
└── Documentation
    └── docs/
        ├── architecture.md
        ├── deployment.md
        ├── api-reference.md
        └── operations.md
```

## Quick Reference

### To Start Development

1. Read: `QUICKSTART.md`
2. Setup: `docker-compose up -d`
3. Test: `curl http://localhost:8001/api/v1/health/status`

### To Understand Architecture

1. Read: `README.md`
2. Deep dive: `docs/architecture.md`
3. Reference: `PROJECT_SUMMARY.md`

### To Deploy to Kubernetes

1. Read: `docs/deployment.md`
2. Build images: See deployment guide
3. Apply manifests: `kubernetes/`

### To Operate the System

1. Read: `docs/operations.md`
2. Monitor: Grafana at `http://localhost:3000`
3. Troubleshoot: See operations guide

### To Contribute Code

1. Read: `CONTRIBUTING.md`
2. Setup dev environment: See CONTRIBUTING.md
3. Follow code style: See CONTRIBUTING.md

### To Understand APIs

1. Reference: `docs/api-reference.md`
2. Test: Use curl examples
3. Explore: Swagger UI (if enabled)

## File Statistics

| Category            | Count   | Purpose             |
| ------------------- | ------- | ------------------- |
| Python files        | 25+     | Backend services    |
| YAML files          | 10+     | Kubernetes & Docker |
| SQL files           | 1       | Database schema     |
| Markdown files      | 8       | Documentation       |
| Configuration files | 5+      | Settings & secrets  |
| **Total**           | **50+** | Complete platform   |

## Important Files to Know

### Critical for Operations

- `kubernetes/deployments/v2-deployment.yaml` - V2 production deployment
- `kubernetes/network-policies/isolation.yaml` - Network isolation
- `database/schemas/init.sql` - Database initialization
- `docker-compose.yml` - Local development setup

### Critical for Development

- `backend/shared/config/settings.py` - Global configuration
- `backend/shared/models/experiment.py` - Data models
- `backend/shared/utils/ai_client.py` - AI integration
- `backend/v2-production/src/main.py` - V2 entry point
- `backend/v1-experimentation/src/main.py` - V1 entry point

### Critical for Understanding

- `README.md` - Project overview
- `docs/architecture.md` - System design
- `PROJECT_SUMMARY.md` - Complete summary
- `QUICKSTART.md` - Quick start guide

## File Naming Conventions

- **Python files**: `snake_case.py`
- **YAML files**: `kebab-case.yaml`
- **Markdown files**: `PascalCase.md`
- **Directories**: `kebab-case/`
- **Classes**: `PascalCase`
- **Functions**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`

## Version Control

- `.gitignore` - Files to exclude from git
- `CHANGELOG.md` - Track changes
- Commit messages follow conventional commits

## Next Steps

1. **Start here**: Read `QUICKSTART.md`
2. **Understand**: Read `README.md` and `docs/architecture.md`
3. **Deploy**: Follow `docs/deployment.md`
4. **Operate**: Use `docs/operations.md`
5. **Contribute**: Follow `CONTRIBUTING.md`

---

**Last Updated**: December 2, 2025
**Total Files**: 50+
**Total Lines of Code**: 5000+
**Documentation Pages**: 8
