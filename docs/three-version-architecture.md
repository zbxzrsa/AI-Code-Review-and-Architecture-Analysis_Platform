# Three-Version Self-Evolving Architecture

## Overview

The platform implements a three-version parallel architecture that enables continuous self-evolution while maintaining production stability:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CONTROL PLANE                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  Lifecycle   │  │   Policy     │  │    Model     │  │  Evaluation  │    │
│  │  Controller  │  │   Engine     │  │   Registry   │  │   Pipeline   │    │
│  │              │  │   (OPA)      │  │              │  │  (Argo/ML)   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │   Prompt     │  │  Experiment  │  │    Metric    │  │    Audit     │    │
│  │  Registry    │  │   Tracking   │  │  Aggregator  │  │    Logger    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   V1 EXPERIMENT │      │    V2 STABLE    │      │   V3 LEGACY     │
│   ───────────── │      │   ───────────── │      │   ───────────── │
│                 │      │                 │      │                 │
│  • Shadow Mode  │      │  • User Facing  │      │  • Quarantine   │
│  • New Models   │      │  • Strong SLOs  │      │  • Recovery     │
│  • A/B Testing  │      │  • Baseline     │      │  • Re-eval      │
│  • No Response  │◄────►│  • Auto Scale   │◄────►│  • Low Cost     │
│                 │      │                 │      │                 │
│  ┌───────────┐  │      │  ┌───────────┐  │      │  ┌───────────┐  │
│  │   VCAI    │  │      │  │   VCAI    │  │      │  │   VCAI    │  │
│  │   CRAI    │  │      │  │   CRAI    │  │      │  │   CRAI    │  │
│  │   Cache   │  │      │  │   Cache   │  │      │  │   Cache   │  │
│  │   Queue   │  │      │  │   Queue   │  │      │  │   Queue   │  │
│  └───────────┘  │      │  └───────────┘  │      │  └───────────┘  │
│                 │      │                 │      │                 │
│  Schema:        │      │  Schema:        │      │  Schema:        │
│  experiments_v1 │      │  production     │      │  quarantine     │
└─────────────────┘      └─────────────────┘      └─────────────────┘
         │                          ▲                          │
         │                          │                          │
         └──────────────────────────┴──────────────────────────┘
                           Traffic Flow
                    (Only V2 exposed to users)
```

## Version Descriptions

### V1 Experiment Zone

- **Purpose**: Rapid trial and error for new models, prompts, and routing strategies
- **Traffic**: Shadow traffic (mirrored from V2, no response to users)
- **SLOs**: Relaxed - focus on learning, not stability
- **Resources**: GPU-enabled nodes, scale-to-zero when idle
- **Isolation**: Strict network policies, separate secrets

### V2 Stable Zone

- **Purpose**: Production user-facing service with strong SLOs
- **Traffic**: 100% user traffic
- **SLOs**:
  - P95 latency < 3s
  - Error rate < 2%
  - Availability > 99.9%
  - Security pass rate ≥ 99%
- **Resources**: Highest priority, reserved capacity
- **Deployment**: Argo Rollouts with blue-green strategy

### V3 Legacy/Quarantine Zone

- **Purpose**: Recovery, re-evaluation, and parameter repair
- **Traffic**: Optional comparison traffic (admin only)
- **Resources**: Minimal allocation, low priority
- **Use Cases**:
  - Failed experiments awaiting repair
  - Legacy versions for rollback
  - Re-evaluation candidates

## Traffic Flow

```
                    ┌─────────────────┐
                    │   API Gateway   │
                    │   (Nginx/Kong)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
         ┌────────┐    ┌────────┐    ┌────────┐
         │ Mirror │    │  Main  │    │ Mirror │
         │  (V1)  │    │  (V2)  │    │ (V3)*  │
         └────────┘    └────────┘    └────────┘
              │              │              │
              │              ▼              │
              │      ┌────────────┐         │
              │      │  Response  │         │
              │      │  to User   │         │
              │      └────────────┘         │
              │                             │
              └──────────────┬──────────────┘
                             │
                    ┌────────▼────────┐
                    │   Evaluation    │
                    │    Pipeline     │
                    └─────────────────┘

* V3 mirror is optional, for comparison/debugging
```

## Promotion Lifecycle

```
┌──────────────────────────────────────────────────────────────────────┐
│                        PROMOTION LIFECYCLE                            │
└──────────────────────────────────────────────────────────────────────┘

  New Config
      │
      ▼
┌─────────────┐
│  V1 Shadow  │ ─────────────────────────────────────────┐
│  Evaluation │                                          │
└──────┬──────┘                                          │
       │                                                 │
       │ Pass                                    Fail    │
       ▼                                                 │
┌─────────────┐                                          │
│  OPA Gate   │                                          │
│  Decision   │                                          │
└──────┬──────┘                                          │
       │                                                 │
       │ Approved                                        │
       ▼                                                 ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Gray 1%     │ ──► │ Gray 5%     │ ──► │ Gray 25%    │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       │                   │                   │
       ▼                   ▼                   ▼
   SLO Check           SLO Check           SLO Check
       │                   │                   │
       │ Pass              │ Pass              │ Pass
       ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│ Gray 50%    │ ──► │ Full 100%   │ ──► │ New         │
│             │     │ Rollout     │     │ Baseline    │
└──────┬──────┘     └─────────────┘     └─────────────┘
       │
       │ Fail at any stage
       ▼
┌─────────────┐
│ V3 Quarant. │ ──► Re-evaluation ──► Back to V1
└─────────────┘
```

## OPA Policy Thresholds

| Metric                   | Threshold | Description                   |
| ------------------------ | --------- | ----------------------------- |
| P95 Latency              | < 3000ms  | 95th percentile response time |
| Error Rate               | < 2%      | HTTP 5xx errors               |
| Accuracy Delta           | ≥ +2%     | Improvement over baseline     |
| Security Pass Rate       | ≥ 99%     | Red-team test pass rate       |
| Cost Increase            | ≤ +10%    | Maximum cost increase allowed |
| Statistical Significance | p < 0.05  | T-test for accuracy           |

## Rollback Triggers

Automatic rollback is triggered when:

1. **SLO Violation**: Any metric exceeds threshold for 3 consecutive windows
2. **Security Failure**: Security pass rate drops below 95%
3. **Cost Budget Exceeded**: Daily cost limit reached
4. **Statistical Regression**: Significant accuracy decline (p < 0.05)
5. **Manual Trigger**: Admin-initiated rollback

## Database Schema Separation

```sql
-- V1 Experimentation
CREATE SCHEMA experiments_v1;
  - shadow_analyses
  - experiment_configs
  - evaluation_results

-- V2 Production
CREATE SCHEMA production;
  - analysis_sessions
  - slo_metrics
  - baseline_versions

-- V3 Quarantine
CREATE SCHEMA quarantine;
  - quarantined_versions
  - reevaluation_requests
  - shadow_analyses

-- Cross-version Lifecycle
CREATE SCHEMA lifecycle;
  - version_events (audit)
  - promotions
  - gold_set_results
```

## Kubernetes Resource Allocation

| Zone          | Priority  | CPU Requests | Memory    | GPU | Scale |
| ------------- | --------- | ------------ | --------- | --- | ----- |
| V2 Stable     | 1,000,000 | 500m-4000m   | 1-8Gi     | -   | 5-100 |
| V1 Experiment | 100       | 100m-2000m   | 256Mi-4Gi | 1-4 | 0-20  |
| V3 Legacy     | 10        | 50m-500m     | 64Mi-1Gi  | -   | 1-5   |

## CI/CD Pipeline Stages

```
1. BUILD
   └─ SBOM Generation (Syft)
   └─ Image Signing (Cosign)
   └─ Vulnerability Scan (Trivy)

2. TEST
   └─ Unit Tests
   └─ Integration Tests
   └─ Security Tests (Injection, Escalation)
   └─ Property Tests

3. DEPLOY V1
   └─ Apply V1 Overlay
   └─ Enable Shadow Traffic
   └─ Register Experiment

4. SHADOW EVALUATION
   └─ Gold-Set Evaluation
   └─ Collect Metrics
   └─ Statistical Tests

5. OPA GATE
   └─ Query OPA for Decision
   └─ Record Audit Trail

6. GRAY-SCALE V2
   └─ 1% → 5% → 25% → 50% → 100%
   └─ SLO Checks at Each Phase
   └─ Auto-Rollback on Failure

7. MONITOR
   └─ 30-min SLO Window
   └─ Notify Success/Failure
```

## Evidence Chain

Every recommendation includes an evidence chain:

```json
{
  "recommendation": {
    "type": "security_vulnerability",
    "severity": "high",
    "message": "SQL injection vulnerability detected",
    "evidence_chain": {
      "triggering_rules": ["CWE-89", "OWASP-A03"],
      "code_locations": [{ "file": "api/users.py", "line": 42, "column": 15 }],
      "static_analysis": {
        "tool": "semgrep",
        "rule_id": "python.lang.security.audit.dangerous-exec-use"
      },
      "model_confidence": 0.94,
      "model_version": "gpt-4o-2024-05-13",
      "prompt_version": "security-audit-v2"
    }
  }
}
```

## Compliance Modes

### Financial Mode

- Local models only (no external API calls)
- US data residency
- Full audit logging
- Additional encryption

### Healthcare (HIPAA) Mode

- On-premise deployment only
- No data leaves environment
- Enhanced audit trail
- Specific model blocklist

## Quick Commands

```bash
# Check system status
kubectl get pods -n platform-v1-exp
kubectl get pods -n platform-v2-stable
kubectl get pods -n platform-v3-legacy

# View rollout status
kubectl argo rollouts get rollout vcai-rollout -n platform-v2-stable

# Trigger manual evaluation
curl -X POST http://lifecycle-controller:8080/versions/VERSION_ID/evaluate

# Force rollback
kubectl argo rollouts abort vcai-rollout -n platform-v2-stable

# Check OPA decision
curl -X POST http://opa:8181/v1/data/lifecycle/promotion -d @input.json
```

## Monitoring Dashboards

1. **Three-Version Comparison**: Side-by-side V1/V2/V3 metrics
2. **Shadow Coverage**: V1 shadow traffic statistics
3. **Gray-Scale Health**: Current phase and progression
4. **Top K Rollback Reasons**: Historical failure analysis
5. **Cost Tracking**: Per-version cost breakdown
