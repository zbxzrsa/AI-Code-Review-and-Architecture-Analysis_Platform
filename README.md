# AI-Powered Code Review and Architecture Analysis Platform

A revolutionary intelligent code review platform with a **three-version self-evolving cycle** mechanism powered by dual AI models. Ensures zero-error user experience while enabling safe experimentation with cutting-edge AI technologies.

## 🚀 Quick Start

### Prerequisites

| Requirement    | Version | Check Command            |
| -------------- | ------- | ------------------------ |
| Docker         | 20.10+  | `docker --version`       |
| Docker Compose | 2.0+    | `docker compose version` |
| Node.js        | 20+     | `node --version`         |
| Python         | 3.10+   | `python --version`       |

### Option 1: Quick Demo (No API Keys Required)

```bash
# 1. Setup environment
cp .env.example .env
# MOCK_MODE=true is enabled by default

# 2. Start infrastructure
docker compose up -d

# 3. Start backend API (Terminal 1)
cd backend && python dev-api-server.py

# 4. Start frontend (Terminal 2)
cd frontend && npm install && npm run dev
```

### Option 2: Full Mode (With AI Providers)

```bash
# 1. Configure API keys in .env
cp .env.example .env
# Edit .env:
#   MOCK_MODE=false
#   OPENAI_API_KEY=sk-your-key
#   ANTHROPIC_API_KEY=sk-ant-your-key

# 2. Follow steps 2-4 from Option 1
```

### Access Points

| Service        | URL                        | Description            |
| -------------- | -------------------------- | ---------------------- |
| **Frontend**   | http://localhost:5173      | Vite dev server        |
| **API Server** | http://localhost:8000      | FastAPI backend        |
| **API Docs**   | http://localhost:8000/docs | Swagger/OpenAPI        |
| **Grafana**    | http://localhost:3002      | Monitoring dashboards  |
| **Prometheus** | http://localhost:9090      | Metrics collection     |
| **MinIO**      | http://localhost:9001      | Object storage console |

### Validate Environment

```bash
# Run environment validation
python scripts/validate_env.py

# Run API health check
python scripts/health_check.py
```

### Demo Credentials

- **Email**: demo@example.com
- **Password**: demo123

## 🏗️ Architecture Overview

### Three-Version Isolation System

#### **V1 - Experimentation Zone** 🧪

- **Purpose**: Testing new AI models, prompts, routing strategies, and analysis techniques
- **Kubernetes Namespace**: `platform-v1-exp` (relaxed resource quotas)
- **Database Schema**: `experiments_v1` (PostgreSQL)
- **Tracking**: Metrics for accuracy, latency, cost, error_rate
- **Promotion**: Automatic graduation to V2 upon passing evaluation thresholds
- **Failure Handling**: Archived to V3 with detailed failure analysis

#### **V2 - Stable Production Zone** ✅

- **Purpose**: Only version accessible to end users
- **Kubernetes Namespace**: `platform-v2-stable` (guaranteed resources, HPA enabled)
- **Database Schema**: `production` (PostgreSQL with comprehensive backups)
- **SLO Enforcement**:
  - 95th percentile response time < 3s
  - Error rate < 2%
- **Deployment Policy**: Immutable - changes only through V1 graduation
- **Network Policy**: Isolated from V1/V3, no cross-contamination

#### **V3 - Quarantine Zone** 🔒

- **Purpose**: Archive for underperforming techniques and blacklisted configurations
- **Database Schema**: `quarantine` (read-only PostgreSQL)
- **Resource Allocation**: Minimal
- **Review Process**: Administrators can review and request re-evaluation to V1
- **Evidence Chain**: Maintains failure reasons, timestamps, and impact analysis

## 📋 Project Structure

```
.
├── backend/
│   ├── v1-experimentation/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── v2-production/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── v3-quarantine/
│   │   ├── src/
│   │   ├── tests/
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   └── shared/
│       ├── config/
│       ├── models/
│       ├── utils/
│       └── services/
│           ├── version_control_ai.py      ← Admin-only evaluation
│           ├── code_review_ai.py          ← User-facing analysis
│           ├── event_bus.py               ← Event-driven architecture
│           └── feature_flags.py           ← Gradual rollouts
├── kubernetes/
│   ├── namespaces/
│   ├── deployments/
│   │   ├── v1-deployment.yaml
│   │   ├── v2-deployment.yaml
│   │   ├── v3-deployment.yaml
│   │   ├── version-control-ai.yaml       ← GPU-accelerated
│   │   └── code-review-ai.yaml           ← HPA: 3-50 pods
│   ├── network-policies/
│   └── config/
├── database/
│   ├── schemas/
│   └── init-scripts/
├── monitoring/
│   ├── prometheus/
│   ├── grafana/
│   └── alerting/
├── docs/
│   ├── architecture.md
│   ├── deployment.md
│   ├── api-reference.md
│   ├── operations.md
│   ├── dual-ai-architecture.md           ← NEW: Dual AI guide
│   └── dual-ai-integration.md            ← NEW: Integration guide
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (for production deployment)
- Python 3.9+
- Node.js 16+
- PostgreSQL 13+

### Local Development

```bash
# Clone the repository
git clone <repo-url>
cd AI-Code-Review-and-Architecture-Analysis_Platform

# Copy environment configuration
cp .env.example .env

# Start all services with Docker Compose
docker-compose up -d

# Initialize databases
python scripts/init-databases.py

# Run migrations
python scripts/run-migrations.py

# Start frontend development server
cd frontend && npm install && npm run dev
```

### Access Points

| Service     | URL                                  | Description           |
| ----------- | ------------------------------------ | --------------------- |
| Frontend    | http://localhost:5173                | Main application      |
| AI Hub      | http://localhost:5173/ai-hub         | Three-version AI chat |
| Code Review | http://localhost:5173/ai-code-review | AI code analysis      |
| Admin VCAI  | http://localhost:5173/admin/vcai     | Version control AI    |
| API Docs    | http://localhost:8010/docs           | Evolution API docs    |
| Dev API     | http://localhost:8000/docs           | Dev server API docs   |
| Grafana     | http://localhost:3001                | Monitoring dashboards |

## 🔄 Evolution Cycle

### Experiment Promotion Flow

```
V1 (Experiment)
    ↓
    [Evaluation Gate]
    ↓
    ├─→ PASS → V2 (Production)
    └─→ FAIL → V3 (Quarantine)
```

### Metrics Tracked

- **Accuracy**: Code review correctness rate
- **Latency**: Response time (p50, p95, p99)
- **Cost**: API calls, compute resources
- **Error Rate**: Failed analyses
- **User Satisfaction**: Feedback scores

### Three-Version Evolution Service

The platform includes a dedicated **spiral evolution management service** with:

```bash
# Start evolution service
make start-three-version

# View admin UI
# Navigate to: /admin/three-version

# API documentation
# http://localhost:8010/docs
```

**8-Phase Spiral Cycle:**

1. **Experimentation** - V1 tests new technologies
2. **Error Remediation** - V2 fixes V1 errors
3. **Evaluation** - Check promotion criteria
4. **Promotion** - V1 → V2
5. **Stabilization** - V2 optimizes
6. **Degradation** - V2 → V3 poor performers
7. **Comparison** - V3 baseline analysis
8. **Re-evaluation** - V3 → V1 retry

**Key Commands:**

```bash
make verify-three-version      # Verify implementation
make test-three-version        # Run tests
make logs-three-version        # View logs
```

## 📊 Key Features

### Three-Version Architecture

- ✅ **V1 Experimentation**: Safe testing ground for new models
- ✅ **V2 Production**: Stable user-facing API with strict SLOs
- ✅ **V3 Quarantine**: Archive for failed experiments

### Dual AI Model Architecture

- ✅ **Version Control AI**: Admin-only evaluation with statistical testing
- ✅ **Code Review AI**: User-facing analysis with comprehensive scanning
- ✅ **Event-Driven**: Async processing with event bus
- ✅ **Feature Flags**: Gradual rollouts and A/B testing

### Version Control AI (Admin)

- ✅ Statistical significance testing (t-test, chi-square)
- ✅ Regression detection (accuracy, latency, cost, error rate, security)
- ✅ Cost-benefit analysis with ROI calculation
- ✅ A/B testing analysis
- ✅ Cryptographic report signatures
- ✅ S3 storage with integrity verification
- ✅ OPA policy engine integration

### Code Review AI (User-Facing)

- ✅ Security vulnerability scanning (SAST)
- ✅ Code quality and style analysis
- ✅ Performance bottleneck detection
- ✅ Architecture dependency analysis
- ✅ Test generation and coverage recommendations
- ✅ Documentation and comment generation
- ✅ Intelligent patch generation
- ✅ Multi-model routing with fallback chains
- ✅ User-provided API key support
- ✅ HPA scaling (3-50 pods)

### AI Interaction Pages (Frontend)

- ✅ **AI Hub** (`/ai-hub`) - Three-version AI chat interface
  - Switch between V1/V2/V3 models
  - Compare mode for side-by-side responses
  - Real-time streaming responses
  - Quick prompts for common tasks
- ✅ **Code Review AI** (`/ai-code-review`) - User-facing code analysis
  - Paste code for instant AI review
  - Multi-type analysis (security, performance, quality, bugs)
  - One-click auto-fix
  - User feedback collection
- ✅ **Version Control AI** (`/admin/vcai`) - Admin AI management
  - Technology management (promote/degrade)
  - Evolution cycle control (start/stop)
  - Experiment tracking
  - Re-evaluation requests

### Enterprise Features

- ✅ **Dual AI Model Support**: OpenAI GPT-4, Anthropic Claude-3, HuggingFace
- ✅ **Automatic Promotion**: ML-driven evaluation and promotion
- ✅ **Zero-Error UX**: Only stable V2 exposed to users
- ✅ **Comprehensive Audit Trail**: Full traceability of experiments
- ✅ **Resource Isolation**: Kubernetes-based namespace segregation
- ✅ **Real-time Monitoring**: Prometheus + Grafana integration
- ✅ **Scalable Architecture**: Horizontal Pod Autoscaling
- ✅ **Network Security**: Strict network policies between versions
- ✅ **GPU Acceleration**: Version Control AI on GPU nodes
- ✅ **Feature Flags**: Request-level control for gradual rollouts

## 🛠️ Development

### Running Tests

```bash
# V1 Experimentation tests
cd backend/v1-experimentation && pytest tests/

# V2 Production tests
cd backend/v2-production && pytest tests/

# V3 Quarantine tests
cd backend/v3-quarantine && pytest tests/
```

### Building Docker Images

```bash
# Build all services
docker-compose build

# Build specific service
docker build -t platform-v1:latest backend/v1-experimentation/
docker build -t platform-v2:latest backend/v2-production/
docker build -t platform-v3:latest backend/v3-quarantine/
```

## 📈 Monitoring & Observability

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization dashboards
- **ELK Stack**: Centralized logging
- **Jaeger**: Distributed tracing
- **Custom Alerts**: SLO-based alerting

## 🔐 Security

- Network policies enforce version isolation
- RBAC for Kubernetes access control
- Encrypted database connections
- API authentication and rate limiting
- Audit logging for all operations

## 📚 Documentation

### Architecture & Design

- [Architecture Guide](docs/architecture.md) - Three-version system design
- [Dual AI Architecture](docs/dual-ai-architecture.md) - Version Control AI & Code Review AI
- [Microservices Layer](docs/microservices.md) - Auth, Project, Repo, Analysis services
- [Frontend Stack](docs/frontend-stack.md) - React 18, TypeScript, Ant Design
- [API Gateway](docs/api-gateway.md) - Traefik/Nginx configuration

### Deployment & Operations

- [Deployment Guide](docs/deployment.md) - Docker Compose & Kubernetes
- [Operations Runbook](docs/operations.md) - Production operations
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Integration Guide](docs/dual-ai-integration.md) - Service integration

### Implementation Summaries

- [Dual AI Summary](DUAL_AI_SUMMARY.md) - Dual AI implementation
- [Frontend & Gateway Summary](FRONTEND_GATEWAY_SUMMARY.md) - Frontend and API gateway
- [Microservices Summary](MICROSERVICES_SUMMARY.md) - Microservices implementation
- [AI Orchestration Summary](AI_ORCHESTRATION_SUMMARY.md) - AI orchestration layer

### Quick References

- [Quick Start](QUICKSTART.md) - 5-minute setup
- [Project Summary](PROJECT_SUMMARY.md) - Complete overview

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See LICENSE file for details
