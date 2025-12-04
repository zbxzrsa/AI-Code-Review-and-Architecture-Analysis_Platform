# Promotion and Rollback Procedures

## Overview

This document outlines the operational procedures for managing version promotions and rollbacks in the three-version architecture.

## Table of Contents

1. [Promotion Workflow](#promotion-workflow)
2. [Rollback Procedures](#rollback-procedures)
3. [Emergency Procedures](#emergency-procedures)
4. [Troubleshooting Guide](#troubleshooting-guide)

---

## Promotion Workflow

### Standard Promotion Path

```
V1 Shadow → OPA Gate → V2 Gray (1% → 5% → 25% → 50% → 100%)
```

### Pre-Promotion Checklist

Before initiating a promotion:

- [ ] Shadow evaluation completed (min 1000 requests, 24 hours)
- [ ] All gold-set tests passed (≥95% security, ≥90% overall)
- [ ] Statistical tests show significant improvement (p < 0.05)
- [ ] Cost increase within budget (≤10%)
- [ ] No active incidents in V2

### Step-by-Step Promotion

#### 1. Verify Shadow Evaluation Results

```bash
# Check shadow metrics
kubectl exec -n platform-control-plane deployment/lifecycle-controller -- \
  curl localhost:8080/versions/VERSION_ID/metrics

# Verify gold-set results
kubectl exec -n platform-control-plane deployment/evaluation-pipeline -- \
  curl localhost:8080/results/VERSION_ID
```

#### 2. Query OPA for Approval

```bash
# Manual OPA query
curl -X POST http://opa.platform-control-plane.svc:8181/v1/data/lifecycle/promotion \
  -H "Content-Type: application/json" \
  -d @promotion-input.json
```

Example `promotion-input.json`:

```json
{
  "input": {
    "version": {
      "id": "v1-abc123",
      "model": "gpt-4o-2024-05-13",
      "prompt": "code-review-v4-exp",
      "state": "shadow"
    },
    "metrics": {
      "p95_latency_ms": 2500,
      "error_rate": 0.01,
      "accuracy_delta": 0.03,
      "security_pass_rate": 0.995,
      "cost_delta": 0.05
    },
    "statistical_tests": {
      "accuracy_p_value": 0.02,
      "latency_p_value": 0.15,
      "cost_p_value": 0.08
    },
    "thresholds": {
      "p95_latency_ms": 3000,
      "error_rate": 0.02,
      "accuracy_delta": 0.02,
      "security_pass_rate": 0.99,
      "cost_increase_max": 0.1,
      "statistical_significance_p": 0.05
    }
  }
}
```

#### 3. Initiate Gray-Scale Rollout

```bash
# Start gray-scale (1%)
kubectl argo rollouts set image vcai-rollout \
  vcai=gcr.io/PROJECT/vcai:VERSION_ID \
  -n platform-v2-stable

# Monitor the rollout
kubectl argo rollouts get rollout vcai-rollout -n platform-v2-stable -w
```

#### 4. Monitor Each Phase

```bash
# Check health during rollout
./scripts/check_rollout_health.sh vcai-rollout platform-v2-stable

# Promote to next phase
kubectl argo rollouts promote vcai-rollout -n platform-v2-stable
```

#### 5. Complete Promotion

```bash
# Verify full rollout
kubectl argo rollouts status vcai-rollout -n platform-v2-stable

# Update baseline
curl -X POST http://lifecycle-controller:8080/baseline/update \
  -H "Content-Type: application/json" \
  -d '{"version_id": "VERSION_ID", "promoted_by": "operator@example.com"}'
```

---

## Rollback Procedures

### Automatic Rollback Triggers

The system automatically rolls back when:

| Trigger          | Condition              | Action             |
| ---------------- | ---------------------- | ------------------ |
| SLO Violation    | 3 consecutive failures | Abort rollout      |
| Security Failure | Pass rate < 95%        | Abort + Quarantine |
| Error Spike      | Error rate > 10%       | Abort rollout      |
| Latency Spike    | P95 > 3x threshold     | Abort rollout      |
| Cost Budget      | Daily limit exceeded   | Pause evaluation   |

### Manual Rollback

#### Immediate Rollback (Gray-Scale in Progress)

```bash
# Abort current rollout
kubectl argo rollouts abort vcai-rollout -n platform-v2-stable

# Revert to previous stable version
kubectl argo rollouts undo vcai-rollout -n platform-v2-stable

# Verify rollback
kubectl argo rollouts status vcai-rollout -n platform-v2-stable
```

#### Rollback After Full Deployment

```bash
# Set image back to previous version
kubectl argo rollouts set image vcai-rollout \
  vcai=gcr.io/PROJECT/vcai:PREVIOUS_VERSION \
  -n platform-v2-stable

# Skip analysis (emergency)
kubectl argo rollouts promote vcai-rollout -n platform-v2-stable --full
```

### Post-Rollback Actions

1. **Document the rollback**:

   ```bash
   curl -X POST http://lifecycle-controller:8080/audit/rollback \
     -H "Content-Type: application/json" \
     -d '{
       "version_id": "VERSION_ID",
       "reason": "SLO_VIOLATION",
       "notes": "P95 latency exceeded 5000ms",
       "triggered_by": "operator@example.com"
     }'
   ```

2. **Move to V3 Quarantine**:

   ```bash
   curl -X POST http://lifecycle-controller:8080/versions/VERSION_ID/downgrade \
     -H "Content-Type: application/json" \
     -d '{"reason": "Rollback after gray-scale SLO violation"}'
   ```

3. **Create repair issue**:
   - Go to GitHub Issues
   - Use template: "V3 Repair Request"
   - Include rollback reason and metrics snapshot

---

## Emergency Procedures

### P0 Incident: Complete Service Outage

1. **Immediate Actions** (0-5 minutes):

   ```bash
   # Scale up V2 stable deployment
   kubectl scale deployment vcai-stable -n platform-v2-stable --replicas=10

   # Abort any ongoing rollouts
   kubectl argo rollouts abort vcai-rollout -n platform-v2-stable

   # Disable shadow traffic to V1
   kubectl patch configmap traffic-routing-config \
     -n platform-control-plane \
     --type=merge \
     -p '{"data":{"shadow_percentage":"0"}}'
   ```

2. **Diagnosis** (5-15 minutes):

   ```bash
   # Check pod status
   kubectl get pods -n platform-v2-stable -l app=vcai

   # Check recent events
   kubectl get events -n platform-v2-stable --sort-by='.lastTimestamp'

   # Check logs
   kubectl logs -n platform-v2-stable -l app=vcai --tail=100
   ```

3. **Recovery**:
   - If pods crashing: `kubectl rollout undo deployment/vcai-stable`
   - If resource exhaustion: Scale up nodes or reduce replicas
   - If external dependency: Enable circuit breaker / use fallback

### P1 Incident: SLO Degradation

1. **Identify the issue**:

   ```bash
   # Check which version is serving
   kubectl argo rollouts get rollout vcai-rollout -n platform-v2-stable

   # Compare canary vs stable metrics
   ./scripts/check_rollout_health.sh vcai-rollout platform-v2-stable
   ```

2. **If canary is the issue**:

   ```bash
   kubectl argo rollouts abort vcai-rollout -n platform-v2-stable
   ```

3. **If stable is the issue**:

   ```bash
   # Check for recent changes
   kubectl rollout history deployment/vcai-stable -n platform-v2-stable

   # Rollback to previous revision
   kubectl rollout undo deployment/vcai-stable -n platform-v2-stable
   ```

### P2 Incident: Cost Budget Exceeded

1. **Pause shadow evaluation**:

   ```bash
   kubectl patch configmap traffic-routing-config \
     -n platform-control-plane \
     --type=merge \
     -p '{"data":{"shadow_percentage":"0"}}'
   ```

2. **Scale down V1**:

   ```bash
   kubectl scale deployment/vcai-service -n platform-v1-exp --replicas=0
   kubectl scale deployment/crai-service -n platform-v1-exp --replicas=0
   ```

3. **Review and adjust budget**:
   - Review cost breakdown in Grafana dashboard
   - Adjust routing policy to use cheaper models
   - Increase daily budget if justified

---

## Troubleshooting Guide

### Common Issues

#### 1. Shadow Traffic Not Reaching V1

**Symptoms**: V1 metrics show 0 requests

**Check**:

```bash
# Verify gateway configuration
kubectl get configmap traffic-routing-config -n platform-control-plane -o yaml

# Check nginx logs
kubectl logs -n platform-gateway -l app=nginx-gateway --tail=50
```

**Fix**:

```bash
# Ensure mirror is enabled
kubectl patch configmap traffic-routing-config \
  -n platform-control-plane \
  --type=merge \
  -p '{"data":{"shadow_percentage":"100"}}'

# Restart gateway
kubectl rollout restart deployment/nginx-gateway -n platform-gateway
```

#### 2. OPA Decision Failures

**Symptoms**: Promotions stuck, OPA errors in logs

**Check**:

```bash
# Check OPA health
kubectl get pods -n platform-control-plane -l app=opa

# Check OPA logs
kubectl logs -n platform-control-plane -l app=opa --tail=50

# Test OPA endpoint
kubectl exec -n platform-control-plane deployment/opa -- \
  curl localhost:8181/health
```

**Fix**:

```bash
# Restart OPA
kubectl rollout restart deployment/opa -n platform-control-plane

# Verify policy loaded
kubectl exec -n platform-control-plane deployment/opa -- \
  curl localhost:8181/v1/policies
```

#### 3. Rollout Stuck in Paused State

**Symptoms**: `kubectl argo rollouts get` shows "Paused"

**Check**:

```bash
# Check analysis run status
kubectl get analysisrun -n platform-v2-stable -l rollout-name=vcai-rollout

# Get analysis details
kubectl describe analysisrun -n platform-v2-stable $(kubectl get analysisrun -n platform-v2-stable -l rollout-name=vcai-rollout -o jsonpath='{.items[0].metadata.name}')
```

**Fix**:

```bash
# If analysis failed, abort
kubectl argo rollouts abort vcai-rollout -n platform-v2-stable

# If paused for approval, promote
kubectl argo rollouts promote vcai-rollout -n platform-v2-stable
```

#### 4. V3 Re-evaluation Not Progressing

**Symptoms**: Quarantined version stuck

**Check**:

```bash
# Check quarantine status
kubectl exec -n platform-control-plane deployment/lifecycle-controller -- \
  curl localhost:8080/quarantine/VERSION_ID

# Check re-evaluation requests
kubectl get pods -n platform-v3-legacy
```

**Fix**:

```bash
# Manually trigger re-evaluation
curl -X POST http://lifecycle-controller:8080/quarantine/VERSION_ID/reevaluate \
  -H "Content-Type: application/json" \
  -d '{"modified_parameters": {}, "reason": "Manual retry"}'
```

---

## Appendix

### Useful Commands

```bash
# Get current system status
kubectl get pods -A -l tier=platform

# View all rollouts
kubectl argo rollouts list -A

# Check SLO metrics
curl "http://prometheus:9090/api/v1/query?query=slo:v2:error_budget_remaining"

# View lifecycle events
curl http://lifecycle-controller:8080/history?limit=50

# Check cost budget
curl http://lifecycle-controller:8080/budget/status
```

### Contact

- **On-call**: #platform-oncall (Slack)
- **Escalation**: platform-leads@example.com
- **Runbook Updates**: Create PR to `docs/operations/`
