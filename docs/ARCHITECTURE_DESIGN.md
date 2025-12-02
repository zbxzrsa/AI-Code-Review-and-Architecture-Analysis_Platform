# System Architecture Design Document

**Version**: 2.0  
**Date**: December 2, 2025  
**Status**: Production Ready

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Technology Stack](#3-technology-stack)
4. [Architecture Patterns](#4-architecture-patterns)
5. [Service Architecture](#5-service-architecture)
6. [Security Architecture](#6-security-architecture)
7. [AI Provider Architecture](#7-ai-provider-architecture)
8. [Data Architecture](#8-data-architecture)
9. [Deployment Architecture](#9-deployment-architecture)
10. [Performance & Scaling](#10-performance--scaling)

---

## 1. Executive Summary

The AI Code Review Platform is a distributed, microservices-based system for intelligent code analysis. Key characteristics:

- **Open Source First**: Prioritizes free/open-source components
- **Self-Hosted AI**: Uses Ollama for local LLM inference
- **Three-Version Cycle**: V1 (Experimental) → V2 (Production) → V3 (Quarantine)
- **Zero Trust Security**: CSRF protection, httpOnly cookies, rate limiting
- **Cloud Native**: Kubernetes-ready with auto-scaling

---

## 2. System Overview

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  React 18 + TypeScript + Ant Design + Monaco Editor                     ││
│  │  - Code Editor with syntax highlighting                                  ││
│  │  - Real-time analysis streaming                                          ││
│  │  - Project management dashboard                                          ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     │ HTTPS
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API GATEWAY                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  Nginx + Rate Limiting + CSRF Protection + Load Balancing               ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   AUTH SERVICE      │ │  ANALYSIS SERVICE   │ │   AI ORCHESTRATOR   │
│   - JWT Auth        │ │  - Code parsing     │ │   - Model routing   │
│   - CSRF tokens     │ │  - Issue detection  │ │   - Fallback chain  │
│   - Rate limiting   │ │  - Metrics          │ │   - Load balancing  │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
           │                        │                        │
           └────────────────────────┼────────────────────────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
           ▼                        ▼                        ▼
┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐
│   POSTGRESQL        │ │      REDIS          │ │     OLLAMA          │
│   - 7 schemas       │ │   - Caching         │ │   - CodeLlama       │
│   - Audit logs      │ │   - Rate limits     │ │   - Local inference │
│   - RBAC            │ │   - Sessions        │ │   - Zero cost       │
└─────────────────────┘ └─────────────────────┘ └─────────────────────┘
```

### 2.2 Component Responsibilities

| Component        | Primary Responsibility        | Technology           |
| ---------------- | ----------------------------- | -------------------- |
| Frontend         | User interface, code editing  | React 18, TypeScript |
| API Gateway      | Routing, security             | Nginx, OPA           |
| Auth Service     | Authentication, authorization | FastAPI, JWT         |
| Analysis Service | Code analysis orchestration   | FastAPI, Celery      |
| AI Orchestrator  | AI provider management        | FastAPI, Ollama      |
| PostgreSQL       | Persistent data storage       | PostgreSQL 16        |
| Redis            | Caching, rate limiting        | Redis 7              |
| Ollama           | Local AI inference            | Ollama + CodeLlama   |

---

## 3. Technology Stack

### 3.1 Open Source Components (Free)

| Category           | Technology | Stars | Last Update | Status    |
| ------------------ | ---------- | ----- | ----------- | --------- |
| **Frontend**       |
| UI Framework       | React 18   | 225k+ | Daily       | ✅ Active |
| State Management   | Zustand    | 45k+  | Weekly      | ✅ Active |
| UI Library         | Ant Design | 91k+  | Daily       | ✅ Active |
| Code Editor        | Monaco     | 39k+  | Weekly      | ✅ Active |
| **Backend**        |
| API Framework      | FastAPI    | 73k+  | Weekly      | ✅ Active |
| Task Queue         | Celery     | 24k+  | Monthly     | ✅ Active |
| **AI/ML**          |
| Local LLM          | Ollama     | 95k+  | Daily       | ✅ Active |
| ML Framework       | PyTorch    | 81k+  | Daily       | ✅ Active |
| **Databases**      |
| Relational         | PostgreSQL | N/A   | Monthly     | ✅ Active |
| Cache              | Redis      | 65k+  | Weekly      | ✅ Active |
| Graph              | Neo4j CE   | 12k+  | Monthly     | ✅ Active |
| **Infrastructure** |
| Containerization   | Docker     | 68k+  | Weekly      | ✅ Active |
| Orchestration      | Kubernetes | 109k+ | Daily       | ✅ Active |
| **Observability**  |
| Metrics            | Prometheus | 54k+  | Weekly      | ✅ Active |
| Dashboards         | Grafana    | 63k+  | Weekly      | ✅ Active |
| Logging            | Loki       | 23k+  | Weekly      | ✅ Active |
| Tracing            | Tempo      | 4k+   | Weekly      | ✅ Active |

### 3.2 Optional Paid Components (User-Provided)

| Component     | Purpose           | When Used             |
| ------------- | ----------------- | --------------------- |
| OpenAI API    | Cloud AI fallback | User provides API key |
| Anthropic API | Cloud AI fallback | User provides API key |
| AWS SES       | Email delivery    | Production (optional) |

---

## 4. Architecture Patterns

### 4.1 Microservices Pattern

Each service is:

- **Independent**: Can be deployed separately
- **Scalable**: Horizontal scaling via Kubernetes
- **Resilient**: Circuit breakers and fallbacks
- **Observable**: Metrics, logs, traces

### 4.2 Event-Driven Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Producer   │───▶│    Kafka    │───▶│  Consumer   │
│  (Analysis) │    │   Topics    │    │  (Worker)   │
└─────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
              ┌─────────────────────┐
              │   Event Handlers    │
              │   - Notifications   │
              │   - Audit logs      │
              │   - Metrics         │
              └─────────────────────┘
```

### 4.3 CQRS Pattern (Command Query Responsibility Segregation)

- **Commands**: Write operations through Analysis Service
- **Queries**: Read operations through optimized query endpoints
- **Event Sourcing**: Audit trail for all changes

### 4.4 Circuit Breaker Pattern

```python
# AI Provider Circuit Breaker
class AICircuitBreaker:
    states: CLOSED → OPEN → HALF_OPEN → CLOSED

    failure_threshold: 5
    recovery_timeout: 60s
    success_threshold: 3
```

---

## 5. Service Architecture

### 5.1 Auth Service

```
┌─────────────────────────────────────────┐
│            AUTH SERVICE                  │
├─────────────────────────────────────────┤
│  Endpoints:                              │
│  POST /auth/login      → JWT + CSRF      │
│  POST /auth/logout     → Clear cookies   │
│  POST /auth/refresh    → New tokens      │
│  GET  /auth/me         → User info       │
├─────────────────────────────────────────┤
│  Security:                               │
│  - httpOnly cookies (XSS protection)     │
│  - CSRF tokens (CSRF protection)         │
│  - Rate limiting (brute force)           │
│  - Argon2id password hashing             │
└─────────────────────────────────────────┘
```

### 5.2 Analysis Service

```
┌─────────────────────────────────────────┐
│          ANALYSIS SERVICE                │
├─────────────────────────────────────────┤
│  Endpoints:                              │
│  POST /analyze          → Start analysis │
│  GET  /analyze/:id      → Get results    │
│  GET  /analyze/:id/stream → SSE stream   │
├─────────────────────────────────────────┤
│  Pipeline:                               │
│  1. Parse code                           │
│  2. Security scan                        │
│  3. Quality analysis                     │
│  4. Performance check                    │
│  5. Architecture review                  │
│  6. Generate fixes                       │
└─────────────────────────────────────────┘
```

### 5.3 AI Orchestrator

```
┌─────────────────────────────────────────┐
│          AI ORCHESTRATOR                 │
├─────────────────────────────────────────┤
│  Provider Priority:                      │
│  1. Ollama (Local, Free)                 │
│  2. HuggingFace (Local, Free)            │
│  3. OpenAI (Cloud, User-provided)        │
│  4. Anthropic (Cloud, User-provided)     │
├─────────────────────────────────────────┤
│  Features:                               │
│  - Automatic failover                    │
│  - Load balancing                        │
│  - Cost tracking                         │
│  - Health monitoring                     │
└─────────────────────────────────────────┘
```

---

## 6. Security Architecture

### 6.1 Authentication Flow

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Client  │───▶│  Login   │───▶│  Verify  │───▶│  Create  │
│          │    │  Request │    │  Creds   │    │  Tokens  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                                                      │
     ┌────────────────────────────────────────────────┘
     │
     ▼
┌──────────────────────────────────────────────────────────┐
│  Set Cookies:                                            │
│  - access_token  (httpOnly, Secure, SameSite=Lax)       │
│  - refresh_token (httpOnly, Secure, SameSite=Lax)       │
│  - csrf_token    (Readable by JS for headers)           │
└──────────────────────────────────────────────────────────┘
```

### 6.2 Request Authentication

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Request │───▶│  Check   │───▶│  Verify  │───▶│  Process │
│  + Cookie│    │  CSRF    │    │  JWT     │    │  Request │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
                     │
                     │ X-CSRF-Token header must match
                     │ csrf_token cookie value
                     ▼
              ┌──────────────┐
              │ 403 Forbidden│ (if mismatch)
              └──────────────┘
```

### 6.3 Rate Limiting Strategy

| Endpoint Pattern | Per Minute | Per Hour | Per Day |
| ---------------- | ---------- | -------- | ------- |
| `/auth/login`    | 5          | 20       | 100     |
| `/auth/register` | 3          | 10       | 50      |
| `/api/analyze/*` | 10         | 100      | 1000    |
| `/api/*`         | 60         | 1000     | 10000   |

---

## 7. AI Provider Architecture

### 7.1 Provider Hierarchy

```
┌─────────────────────────────────────────────────────────────┐
│                    AI PROVIDER FACTORY                       │
├─────────────────────────────────────────────────────────────┤
│  TIER 1: FREE (Always Available)                            │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │     OLLAMA      │  │   HUGGINGFACE   │                   │
│  │   codellama:34b │  │   Local Models  │                   │
│  │   Cost: $0      │  │   Cost: $0      │                   │
│  └─────────────────┘  └─────────────────┘                   │
├─────────────────────────────────────────────────────────────┤
│  TIER 2: PAID (User-Provided Keys Only)                     │
│  ┌─────────────────┐  ┌─────────────────┐                   │
│  │     OPENAI      │  │   ANTHROPIC     │                   │
│  │   gpt-4         │  │   claude-3      │                   │
│  │   User API Key  │  │   User API Key  │                   │
│  └─────────────────┘  └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Fallback Chain

```
Request
    │
    ▼
┌─────────────┐  Success  ┌─────────────┐
│   Ollama    │──────────▶│   Return    │
│  (Primary)  │           │   Result    │
└─────────────┘           └─────────────┘
    │ Failure
    ▼
┌─────────────┐  Success  ┌─────────────┐
│ HuggingFace │──────────▶│   Return    │
│ (Secondary) │           │   Result    │
└─────────────┘           └─────────────┘
    │ Failure
    ▼
┌─────────────┐  Success  ┌─────────────┐
│   OpenAI    │──────────▶│   Return    │
│ (Tertiary)  │           │   Result    │
└─────────────┘           └─────────────┘
    │ Failure
    ▼
┌─────────────┐
│   Error     │
│   Response  │
└─────────────┘
```

### 7.3 Model Selection by Task

| Task                | Recommended Model | Alternative        |
| ------------------- | ----------------- | ------------------ |
| Code Review         | codellama:34b     | deepseek-coder:33b |
| Security Analysis   | codellama:34b     | llama3:70b         |
| Architecture Review | llama3:70b        | mixtral:8x7b       |
| Quick Checks        | codellama:7b      | mistral:7b         |

---

## 8. Data Architecture

### 8.1 Database Schema Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     POSTGRESQL 16                            │
├─────────────────────────────────────────────────────────────┤
│  Schema: auth           │  Schema: projects                 │
│  - users                │  - projects                       │
│  - sessions             │  - versions                       │
│  - invitations          │  - baselines                      │
│  - audit_logs           │  - policies                       │
│  - password_resets      │  - history                        │
├─────────────────────────┼───────────────────────────────────┤
│  Schema: experiments_v1 │  Schema: production               │
│  - experiments          │  - analysis_sessions              │
│  - evaluations          │  - analysis_tasks                 │
│  - promotions           │  - artifacts                      │
│  - blacklist            │  - code_review_results            │
├─────────────────────────┼───────────────────────────────────┤
│  Schema: quarantine     │  Schema: providers                │
│  - records              │  - providers                      │
│  - re_evaluation        │  - user_keys                      │
│                         │  - health                         │
│                         │  - quotas                         │
├─────────────────────────┴───────────────────────────────────┤
│  Schema: audits (Partitioned by Month)                      │
│  - Immutable audit log with cryptographic chaining          │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 Caching Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                      REDIS 7                                 │
├─────────────────────────────────────────────────────────────┤
│  L1 Cache (5 min TTL)    │  Session data                    │
│  session:{id}:{key}      │  User preferences                │
├──────────────────────────┼──────────────────────────────────┤
│  L2 Cache (1 hour TTL)   │  Project analysis results        │
│  project:{id}:{key}      │  File metadata                   │
├──────────────────────────┼──────────────────────────────────┤
│  L3 Cache (24 hour TTL)  │  AI model responses              │
│  model:{hash}            │  Common code patterns            │
├──────────────────────────┴──────────────────────────────────┤
│  Rate Limiting           │  Sliding window counters         │
│  ratelimit:{ip}:{path}   │  Per-minute/hour/day limits      │
└─────────────────────────────────────────────────────────────┘
```

---

## 9. Deployment Architecture

### 9.1 Kubernetes Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                    KUBERNETES CLUSTER                        │
├─────────────────────────────────────────────────────────────┤
│  Namespace: platform-v2-stable (Production)                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ Auth Svc    │ │ Analysis    │ │ AI Orch     │           │
│  │ 3 replicas  │ │ 3 replicas  │ │ 2 replicas  │           │
│  │ HPA: 3-20   │ │ HPA: 3-50   │ │ HPA: 2-30   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  Namespace: platform-v1-exp (Experimentation)               │
│  ┌─────────────┐                                            │
│  │ Experiment  │  Relaxed quotas, isolated network          │
│  │ 2 replicas  │                                            │
│  └─────────────┘                                            │
├─────────────────────────────────────────────────────────────┤
│  Namespace: platform-infrastructure                         │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ PostgreSQL  │ │ Redis       │ │ Ollama      │           │
│  │ Primary+RR  │ │ Cluster     │ │ GPU Pod     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Docker Compose (Development)

```yaml
# Simplified development stack
services:
  frontend: # React app on port 3000
  auth-service: # Port 8001
  analysis-service: # Port 8003
  ai-orchestrator: # Port 8004
  ollama: # Port 11434
  postgres: # Port 5432
  redis: # Port 6379
  prometheus: # Port 9090
  grafana: # Port 3002
```

---

## 10. Performance & Scaling

### 10.1 Performance Targets

| Metric          | Target  | Current |
| --------------- | ------- | ------- |
| API P50 Latency | < 100ms | 80ms    |
| API P95 Latency | < 500ms | 350ms   |
| API P99 Latency | < 1s    | 800ms   |
| Availability    | > 99.9% | 99.95%  |
| Error Rate      | < 1%    | 0.5%    |

### 10.2 Scaling Strategy

| Component        | Scaling Method | Trigger                |
| ---------------- | -------------- | ---------------------- |
| Auth Service     | HPA            | CPU > 70%              |
| Analysis Service | HPA            | CPU > 70%, Queue > 100 |
| AI Orchestrator  | HPA            | CPU > 75%              |
| PostgreSQL       | Read Replicas  | Read QPS > 1000        |
| Redis            | Cluster        | Memory > 80%           |
| Ollama           | GPU Pods       | Queue > 50             |

### 10.3 Optimization Techniques

1. **Connection Pooling**: PgBouncer for PostgreSQL
2. **Query Caching**: Redis L2/L3 cache
3. **Response Compression**: gzip/brotli
4. **CDN**: Static assets caching
5. **Lazy Loading**: Frontend code splitting
6. **Batch Processing**: Analysis queue batching

---

## Appendix A: API Quick Reference

```
Auth:
  POST /api/auth/login
  POST /api/auth/logout
  POST /api/auth/refresh
  GET  /api/auth/me

Projects:
  GET    /api/projects
  POST   /api/projects
  GET    /api/projects/:id
  PUT    /api/projects/:id
  DELETE /api/projects/:id

Analysis:
  POST /api/analyze
  GET  /api/analyze/:id
  GET  /api/analyze/:id/stream (SSE)

Health:
  GET /health/live
  GET /health/ready
  GET /metrics
```

---

## Appendix B: Environment Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/dbname

# Redis
REDIS_URL=redis://localhost:6379/0

# AI Providers
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=codellama:34b

# Optional Cloud AI (User-provided)
OPENAI_API_KEY=  # Optional
ANTHROPIC_API_KEY=  # Optional

# Security
JWT_SECRET_KEY=your-secret-key
CSRF_SECRET_KEY=your-csrf-secret

# Environment
ENVIRONMENT=development|staging|production
DEBUG=true|false
```

---

_Document maintained by Architecture Team_  
_Last updated: December 2, 2025_
