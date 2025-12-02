# Architecture Documentation

## Three-Version Self-Evolving Cycle

The platform implements a revolutionary three-version isolation system that ensures zero-error user experience while enabling safe experimentation with cutting-edge AI technologies.

### Version Overview

#### V1 - Experimentation Zone 🧪

**Purpose**: Testing ground for new AI models, prompts, routing strategies, and analysis techniques

**Key Characteristics**:

- Isolated Kubernetes namespace: `platform-v1-exp`
- Independent PostgreSQL schema: `experiments_v1`
- Relaxed resource quotas for flexibility
- Comprehensive metrics tracking: accuracy, latency, cost, error_rate
- Automatic promotion to V2 upon passing evaluation thresholds
- Failed experiments archived to V3 with detailed failure analysis

**Deployment**:

- 2 replicas (lower availability acceptable)
- Resource requests: 250m CPU, 256Mi memory
- Resource limits: 1 CPU, 1Gi memory
- No HPA (manual scaling for experimentation)

#### V2 - Stable Production Zone ✅

**Purpose**: Only version accessible to end users

**Key Characteristics**:

- Kubernetes namespace: `platform-v2-stable`
- PostgreSQL production schema with comprehensive backup strategy
- Strict SLO enforcement:
  - 95th percentile response time < 3s
  - Error rate < 2%
- Immutable deployment policy: changes only through V1 graduation
- Network policy: isolated from V1/V3, no cross-contamination possible
- High availability with guaranteed resources

**Deployment**:

- 3 replicas (minimum for HA)
- Resource requests: 500m CPU, 512Mi memory
- Resource limits: 2 CPU, 2Gi memory
- HPA: scales from 3 to 10 replicas based on CPU/memory utilization
- Zero-downtime rolling updates

#### V3 - Quarantine Zone 🔒

**Purpose**: Archive for underperforming techniques and blacklisted configurations

**Key Characteristics**:

- Kubernetes namespace: `platform-v3-quarantine`
- Read-only PostgreSQL schema: `quarantine`
- Minimal resource allocation
- Administrators can review and request re-evaluation to V1
- Maintains evidence chain: failure reasons, timestamps, impact analysis
- Permanent blacklist for dangerous configurations

**Deployment**:

- 1 replica (read-only archive)
- Resource requests: 100m CPU, 128Mi memory
- Resource limits: 500m CPU, 512Mi memory
- No scaling (static archive)

### Evolution Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                    Experiment Lifecycle                     │
└─────────────────────────────────────────────────────────────┘

V1 (Experiment)
    │
    ├─→ [Evaluation Gate]
    │       ├─→ Check Accuracy >= 0.95
    │       ├─→ Check Latency <= 3000ms
    │       └─→ Check Error Rate <= 0.02
    │
    ├─→ PASS ──→ V2 (Production)
    │           ├─→ Deploy to stable namespace
    │           ├─→ Run smoke tests
    │           ├─→ Enable for end users
    │           └─→ Monitor SLO compliance
    │
    └─→ FAIL ──→ V3 (Quarantine)
                ├─→ Archive configuration
                ├─→ Log failure analysis
                ├─→ Analyze impact
                └─→ Allow re-evaluation requests
```

### Metrics Tracked

#### V1 Experimentation Metrics

- **Accuracy**: Code review correctness rate (0-1)
- **Latency**: Response time in milliseconds
- **Cost**: API calls and compute resources
- **Error Rate**: Failed analyses (0-1)
- **Throughput**: Requests per second
- **User Satisfaction**: Feedback scores (0-5)
- **False Positives**: Incorrectly flagged issues
- **False Negatives**: Missed issues

#### V2 Production SLOs

- **Response Time P95**: < 3000ms
- **Error Rate**: < 2%
- **Uptime**: > 99.9%
- **Throughput**: Minimum 100 RPS

#### V3 Quarantine Records

- Failure reasons and timestamps
- Metrics at time of failure
- Impact analysis on related experiments
- Re-evaluation eligibility status

### Network Isolation

#### V1 Network Policy

- Ingress: From same namespace and monitoring
- Egress: To same namespace, DNS, and external APIs (AI providers)
- Blocks cross-namespace communication

#### V2 Network Policy

- Ingress: From same namespace, ingress controller, and monitoring
- Egress: To same namespace, DNS, and external APIs
- Strict isolation from V1 and V3
- No inbound traffic from other versions

#### V3 Network Policy

- Ingress: From same namespace, monitoring, and V1 (for re-evaluation)
- Egress: To same namespace (read-only) and DNS
- Minimal external connectivity
- Read-only database access

### Database Schema Isolation

```
PostgreSQL Instance
├── production (V2)
│   ├── code_reviews
│   ├── slo_metrics
│   └── indexes for performance
├── experiments_v1 (V1)
│   ├── experiments
│   ├── experiment_metrics
│   ├── code_analyses
│   └── promotion_readiness (view)
├── quarantine (V3)
│   ├── quarantine_records
│   └── quarantine_summary (view)
└── audit
    └── event_log (shared audit trail)
```

### Deployment Strategy

#### V2 Production Deployment

```yaml
Strategy: RollingUpdate
- maxSurge: 1 (one extra pod during update)
- maxUnavailable: 0 (zero downtime)
- Affinity: Pod anti-affinity for distribution
- Health checks: Liveness and readiness probes
- Security: Non-root user, read-only filesystem
```

#### V1 Experimentation Deployment

```yaml
Strategy: RollingUpdate
- maxSurge: 1
- maxUnavailable: 1 (acceptable for experimentation)
- No pod anti-affinity (flexible placement)
- Health checks: Liveness and readiness probes
- Security: Non-root user, read-only filesystem
```

#### V3 Quarantine Deployment

```yaml
Strategy: RollingUpdate
- maxSurge: 0 (no extra pods)
- maxUnavailable: 1 (acceptable for read-only archive)
- Static placement (single replica)
- Health checks: Liveness and readiness probes
- Security: Non-root user, read-only filesystem
```

### AI Model Routing

#### Primary Model (GPT-4)

- Used in V2 production (stable)
- Proven accuracy and reliability
- Higher cost but better quality

#### Secondary Model (Claude-3-Opus)

- Used in V1 for experimentation
- Testing alternative approaches
- Cost-effective for testing

#### Routing Strategies

- **Primary**: Use GPT-4 only
- **Secondary**: Use Claude-3 only
- **Ensemble**: Get responses from both models
- **Adaptive**: Try primary, fallback to secondary on failure

### Monitoring and Observability

#### Prometheus Metrics

- `v2_requests_total`: Total requests to V2
- `v2_request_duration_seconds`: Request latency
- `v2_active_requests`: Current active requests
- `v2_errors_total`: Total errors
- `v2_slo_violations_total`: SLO violations
- `v1_experiments_total`: Total experiments created
- `v1_promotions_total`: Promotions to V2
- `v1_quarantines_total`: Quarantines to V3

#### Grafana Dashboards

- V2 Production SLO Compliance
- V1 Experiment Progress
- V3 Quarantine Statistics
- Cross-version Comparison

### Security Considerations

1. **Network Isolation**: Kubernetes network policies enforce version separation
2. **RBAC**: Role-based access control for each namespace
3. **Secrets Management**: API keys stored in Kubernetes secrets
4. **Read-Only Archive**: V3 enforces read-only database access
5. **Audit Trail**: All operations logged in audit schema
6. **Pod Security**: Non-root users, read-only filesystems, dropped capabilities

### Scalability

#### V2 Production Scaling

- Horizontal Pod Autoscaler (HPA) enabled
- Scales based on CPU (70%) and memory (80%) utilization
- Min 3 replicas, max 10 replicas
- Gradual scale-up (100% increase per 30s)
- Conservative scale-down (50% decrease per 60s)

#### V1 Experimentation Scaling

- Manual scaling for controlled experimentation
- Typically 2 replicas for development
- Can scale up for load testing

#### V3 Quarantine Scaling

- Static single replica (read-only archive)
- No scaling needed

### Disaster Recovery

1. **Database Backups**: Regular PostgreSQL backups
2. **Schema Separation**: Failure in one schema doesn't affect others
3. **Version Isolation**: V1 failures don't impact V2 users
4. **Audit Trail**: Complete history for forensics
5. **Re-evaluation**: Failed experiments can be re-evaluated from V3

### Cost Optimization

1. **V1 Lower Resources**: Experimentation uses fewer resources
2. **V3 Minimal Resources**: Archive requires minimal compute
3. **V2 Efficient Scaling**: HPA prevents over-provisioning
4. **Model Selection**: Secondary model cheaper for experimentation
5. **Resource Quotas**: Namespace-level quotas prevent runaway costs
