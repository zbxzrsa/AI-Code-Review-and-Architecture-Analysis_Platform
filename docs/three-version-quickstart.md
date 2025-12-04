# Three-Version Architecture Quick Start

Get the three-version self-evolving architecture running in 10 minutes.

## Prerequisites

- Docker & Docker Compose
- kubectl (for Kubernetes deployment)
- Node.js 18+ (for frontend development)
- Python 3.11+ (for backend services)

## Quick Start (Local Development)

### 1. Start Core Infrastructure

```bash
# Start databases and supporting services
docker-compose up -d postgres redis

# Wait for services to be ready
docker-compose logs -f postgres redis
```

### 2. Start Backend Services

```bash
# Option A: Docker Compose (recommended)
docker-compose up -d vcai-service crai-service lifecycle-controller

# Option B: Local Python
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 3. Start Frontend

```bash
cd frontend
npm install
npm run dev
```

### 4. Access the Platform

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **Grafana**: http://localhost:3001 (admin/admin)

---

## Kubernetes Deployment

### 1. Apply Base Manifests

```bash
# Create namespaces and base resources
kubectl apply -k kubernetes/base/

# Verify namespaces
kubectl get ns | grep platform
```

Expected output:

```
platform-v1-exp         Active
platform-v2-stable      Active
platform-v3-legacy      Active
platform-control-plane  Active
platform-monitoring     Active
```

### 2. Deploy V2 Production

```bash
# Deploy V2 stable (production)
kubectl apply -k kubernetes/overlays/v2-stable/

# Check deployment status
kubectl get pods -n platform-v2-stable

# Check Argo Rollout
kubectl argo rollouts get rollout vcai-rollout -n platform-v2-stable
```

### 3. Deploy V1 Experiment

```bash
# Deploy V1 experiment
kubectl apply -k kubernetes/overlays/v1-exp/

# Check pods (may scale to zero if no shadow traffic)
kubectl get pods -n platform-v1-exp
```

### 4. Enable Shadow Traffic

```bash
# Update ingress to enable shadow traffic
kubectl patch ingress vcai-ingress -n platform-v2-stable \
  --type=merge \
  -p '{"metadata":{"annotations":{"nginx.ingress.kubernetes.io/mirror-uri":"/api/v1"}}}'
```

### 5. Verify Deployment

```bash
# Run verification script
python scripts/verify_deployment.py

# Or use make
make check-health
```

---

## Key Operations

### Promote V1 to V2 (Gray-Scale)

```bash
# 1. Check evaluation results
curl http://localhost:8080/evaluate/status/v1-new

# 2. If approved, start gray-scale rollout
kubectl argo rollouts set image vcai-rollout \
  vcai=gcr.io/coderev-platform/vcai:v1-new \
  -n platform-v2-stable

# 3. Monitor rollout
kubectl argo rollouts get rollout vcai-rollout -n platform-v2-stable -w

# 4. Promote to next phase (if healthy)
kubectl argo rollouts promote vcai-rollout -n platform-v2-stable
```

### Rollback

```bash
# Abort current rollout
kubectl argo rollouts abort vcai-rollout -n platform-v2-stable

# Or use UI
# Go to Admin → Version Comparison → Click "Rollback"
```

### Check SLOs

```bash
# Query Prometheus
curl "http://localhost:9090/api/v1/query?query=slo:v2:error_rate:ratio_rate5m"

# Or use make
make stats
```

---

## Monitoring Dashboards

### Grafana Dashboards

1. Open http://localhost:3001
2. Login with admin/admin
3. Go to Dashboards → Three-Version Comparison

### Key Metrics to Watch

| Metric             | Threshold | Alert                 |
| ------------------ | --------- | --------------------- |
| V2 Error Rate      | < 2%      | V2ErrorRateSLOBreach  |
| V2 P95 Latency     | < 3000ms  | V2LatencySLOBreach    |
| V1 Accuracy Delta  | ≥ +2%     | (for promotion)       |
| Security Pass Rate | ≥ 99%     | V2SecurityPassRateLow |

---

## Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/platform
REDIS_URL=redis://localhost:6379

# Feature Flags
SHADOW_TRAFFIC_ENABLED=true
SHADOW_PERCENTAGE=100
```

### Promotion Thresholds

Edit `services/lifecycle-controller/policies/lifecycle.rego`:

```rego
thresholds := {
    "p95_latency_ms": 3000,
    "error_rate": 0.02,
    "accuracy_delta": 0.02,
    "security_pass_rate": 0.99,
    "cost_increase_max": 0.10,
    "statistical_significance_p": 0.05
}
```

---

## Troubleshooting

### No Shadow Traffic to V1

```bash
# Check ingress annotations
kubectl get ingress vcai-ingress -n platform-v2-stable -o yaml | grep mirror

# Check V1 pods are running
kubectl get pods -n platform-v1-exp

# Check V1 HPA
kubectl get hpa -n platform-v1-exp
```

### OPA Decisions Failing

```bash
# Check OPA health
kubectl get pods -n platform-control-plane -l app=opa

# Test OPA endpoint
kubectl exec -n platform-control-plane deployment/opa -- \
  curl localhost:8181/health

# Check policy syntax
opa check services/lifecycle-controller/policies/
```

### Rollout Stuck

```bash
# Check analysis run
kubectl get analysisrun -n platform-v2-stable

# Get analysis details
kubectl describe analysisrun -n platform-v2-stable \
  $(kubectl get analysisrun -n platform-v2-stable -o jsonpath='{.items[-1].metadata.name}')

# Manual abort if needed
kubectl argo rollouts abort vcai-rollout -n platform-v2-stable
```

---

## Next Steps

1. **Configure gold-sets**: Edit `services/evaluation-pipeline/gold_sets.yaml`
2. **Customize policies**: Modify `kubernetes/policies/kyverno-policies.yaml`
3. **Set up alerts**: Import `monitoring/observability/prometheus-alerts.yaml`
4. **Enable offline mode**: See `docs/deployment/private-offline-deployment.md`

---

## Useful Commands

```bash
# Make commands
make help           # Show all commands
make k8s-status     # Show deployment status
make rollout-status # Watch Argo Rollout
make gold-set-eval  # Run gold-set evaluation
make stats          # Show metrics

# kubectl shortcuts
alias kv1='kubectl -n platform-v1-exp'
alias kv2='kubectl -n platform-v2-stable'
alias kv3='kubectl -n platform-v3-legacy'
```
