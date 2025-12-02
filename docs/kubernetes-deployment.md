# Kubernetes Deployment Guide

## Overview

Production-grade Kubernetes deployment with namespaces, network policies, horizontal pod autoscaling, and ingress configuration.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Ingress Controller                       │
│                    (ingress-nginx)                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
┌───────▼──┐  ┌──────▼───┐  ┌────▼──────┐
│V2 Stable │  │V1 Exp    │  │V3 Quarantine
│Namespace │  │Namespace │  │Namespace
└───────┬──┘  └──────┬───┘  └────┬──────┘
        │           │            │
    ┌───┴───┐   ┌───┴───┐   ┌────┴──┐
    │       │   │       │   │       │
 Auth   Analysis AI-Orch  V1-API  V3-API
Service Service Service  Service Service
    │       │   │       │   │       │
    └───┬───┘   └───┬───┘   └────┬──┘
        │           │            │
        └───────────┼────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
    PostgreSQL              Redis
    Neo4j                   Kafka
    MinIO                   OPA
```

---

## Namespaces

### Platform V2 Stable (Production)

- **Name**: `platform-v2-stable`
- **Environment**: production
- **Replicas**: 3+ per service
- **Network Policy**: Strict isolation
- **SLO**: p95 < 3s, error rate < 2%

### Platform V1 Experimentation

- **Name**: `platform-v1-exp`
- **Environment**: experimental
- **Replicas**: 1-2 per service
- **Network Policy**: Relaxed
- **Purpose**: Testing new models

### Platform V3 Quarantine

- **Name**: `platform-v3-quarantine`
- **Environment**: quarantine
- **Replicas**: 1 per service
- **Network Policy**: Read-only
- **Purpose**: Archive failed experiments

### Platform Infrastructure

- **Name**: `platform-infrastructure`
- **Components**: PostgreSQL, Redis, Neo4j, MinIO, Kafka, OPA
- **Network Policy**: Allow all internal

### Platform Monitoring

- **Name**: `platform-monitoring`
- **Components**: Prometheus, Grafana, Loki, Tempo, OTEL Collector
- **Network Policy**: Allow scraping

---

## Network Policies

### V2 Production Isolation

- **Ingress**: Only from ingress-nginx
- **Egress**: Only to databases, message queues, policy engine
- **Purpose**: Strict security boundary

### V1 Experimentation

- **Ingress**: Only from ingress-nginx
- **Egress**: Allow all (for experimentation)
- **Purpose**: Flexible testing environment

### V3 Quarantine

- **Ingress**: Only from ingress-nginx (read-only)
- **Egress**: Only to databases (read-only)
- **Purpose**: Immutable archive

---

## Deployments

### Auth Service

```yaml
Replicas: 3
CPU: 250m request, 1000m limit
Memory: 512Mi request, 2Gi limit
Probes: Liveness (30s), Readiness (10s)
```

### Analysis Service

```yaml
Replicas: 3
CPU: 500m request, 2000m limit
Memory: 1Gi request, 4Gi limit
Probes: Liveness (30s), Readiness (10s)
```

### AI Orchestrator

```yaml
Replicas: 2
CPU: 1000m request, 4000m limit
Memory: 2Gi request, 8Gi limit
Node Selector: workload-type=compute
Probes: Liveness (30s), Readiness (10s)
```

---

## Horizontal Pod Autoscaling

### Auth Service HPA

```yaml
Min Replicas: 3
Max Replicas: 20
CPU Target: 70%
Memory Target: 80%
Scale Up: 100% every 15s
Scale Down: 50% every 60s
```

### Analysis Service HPA

```yaml
Min Replicas: 3
Max Replicas: 50
CPU Target: 70%
Memory Target: 80%
Scale Up: 100% every 15s, +4 pods every 15s
Scale Down: 50% every 60s
```

### AI Orchestrator HPA

```yaml
Min Replicas: 2
Max Replicas: 30
CPU Target: 75%
Memory Target: 80%
Scale Up: 100% every 15s, +2 pods every 15s
Scale Down: 50% every 60s
```

---

## Ingress Configuration

### Production Ingress

- **Host**: api.coderev.example.com
- **TLS**: Let's Encrypt (cert-manager)
- **Rate Limit**: 100 requests/second
- **Paths**:
  - `/api/auth` → auth-service
  - `/api/analyze` → analysis-service
  - `/api/orchestrate` → ai-orchestrator
  - `/health` → auth-service

### V1 Experimentation Ingress

- **Host**: v1.coderev.example.com
- **TLS**: Let's Encrypt
- **Paths**: `/` → v1-api

### V3 Quarantine Ingress

- **Host**: v3.coderev.example.com
- **TLS**: Let's Encrypt
- **Paths**: `/` → v3-api

---

## Deployment Steps

### Prerequisites

```bash
# Install kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml

# Install ingress-nginx
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace ingress-nginx --create-namespace
```

### Deploy Infrastructure

```bash
# Create namespaces
kubectl apply -f kubernetes/namespaces.yaml

# Create secrets
kubectl create secret generic database-credentials \
  --from-literal=url=postgresql://... \
  -n platform-v2-stable

kubectl create secret generic jwt-secrets \
  --from-literal=secret-key=<strong-random-key> \
  -n platform-v2-stable

# Create config maps
kubectl create configmap redis-config \
  --from-literal=url=redis://redis:6379 \
  -n platform-v2-stable

# Apply network policies
kubectl apply -f kubernetes/network-policies.yaml

# Apply deployments
kubectl apply -f kubernetes/deployments.yaml

# Apply HPA and Ingress
kubectl apply -f kubernetes/hpa-ingress.yaml
```

### Verify Deployment

```bash
# Check namespaces
kubectl get namespaces

# Check deployments
kubectl get deployments -n platform-v2-stable

# Check pods
kubectl get pods -n platform-v2-stable

# Check services
kubectl get svc -n platform-v2-stable

# Check ingress
kubectl get ingress -n platform-v2-stable

# Check HPA
kubectl get hpa -n platform-v2-stable

# Check network policies
kubectl get networkpolicies -n platform-v2-stable
```

---

## Monitoring & Observability

### Prometheus Scraping

```yaml
Scrape Interval: 15s
Targets:
  - All services (port 8000, path /metrics)
  - Prometheus (port 9090)
  - OTEL Collector (port 8888)
```

### Grafana Dashboards

- Service Health
- Request Metrics
- Database Performance
- Pod Resource Usage
- Network I/O

### Loki Logs

```
Query: {namespace="platform-v2-stable"}
Retention: 30 days
```

### Tempo Traces

```
Sampling: 10%
Retention: 7 days
```

---

## Scaling

### Manual Scaling

```bash
# Scale deployment
kubectl scale deployment auth-service \
  --replicas=5 \
  -n platform-v2-stable

# Check HPA status
kubectl describe hpa auth-service-hpa \
  -n platform-v2-stable
```

### Automatic Scaling

HPA automatically scales based on:

- CPU utilization (70% target)
- Memory utilization (80% target)
- Custom metrics (requests per second)

---

## Rolling Updates

```bash
# Update image
kubectl set image deployment/auth-service \
  auth-service=gcr.io/PROJECT_ID/auth-service:v2.4.0 \
  -n platform-v2-stable

# Check rollout status
kubectl rollout status deployment/auth-service \
  -n platform-v2-stable

# Rollback if needed
kubectl rollout undo deployment/auth-service \
  -n platform-v2-stable
```

---

## Troubleshooting

### Pod Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n platform-v2-stable

# Check logs
kubectl logs <pod-name> -n platform-v2-stable

# Check events
kubectl get events -n platform-v2-stable
```

### Network Issues

```bash
# Test connectivity
kubectl exec -it <pod-name> -n platform-v2-stable -- \
  curl http://auth-service/health

# Check network policies
kubectl get networkpolicies -n platform-v2-stable
```

### Resource Issues

```bash
# Check resource usage
kubectl top pods -n platform-v2-stable

# Check node resources
kubectl top nodes

# Check HPA metrics
kubectl get hpa -n platform-v2-stable -w
```

---

## Security

### Pod Security Policy

```yaml
- runAsNonRoot: true
- runAsUser: 1000
- allowPrivilegeEscalation: false
- readOnlyRootFilesystem: true
- capabilities: drop ALL
```

### Network Policies

- Strict ingress/egress rules
- Namespace isolation
- Service-to-service communication only

### Secrets Management

- Use Kubernetes secrets
- Rotate regularly
- Use external secret management (Vault, AWS Secrets Manager)

---

## Backup & Disaster Recovery

### Database Backup

```bash
# Backup PostgreSQL
kubectl exec -it postgres-pod -n platform-infrastructure -- \
  pg_dump -U postgres code_review_platform > backup.sql

# Restore PostgreSQL
kubectl exec -it postgres-pod -n platform-infrastructure -- \
  psql -U postgres code_review_platform < backup.sql
```

### Persistent Volume Backup

```bash
# List PVCs
kubectl get pvc -n platform-infrastructure

# Create snapshot
kubectl exec -it <pod> -n platform-infrastructure -- \
  tar czf /backup/data.tar.gz /data
```

---

## Performance Tuning

### Resource Requests/Limits

```yaml
Auth Service:
  Requests: CPU 250m, Memory 512Mi
  Limits: CPU 1000m, Memory 2Gi

Analysis Service:
  Requests: CPU 500m, Memory 1Gi
  Limits: CPU 2000m, Memory 4Gi

AI Orchestrator:
  Requests: CPU 1000m, Memory 2Gi
  Limits: CPU 4000m, Memory 8Gi
```

### Pod Disruption Budgets

```yaml
minAvailable: 2
maxUnavailable: 1
```

---

## Cost Optimization

- Use node affinity for workload placement
- Implement pod disruption budgets
- Use reserved instances for baseline load
- Spot instances for batch processing

---

## Future Enhancements

- [ ] Service mesh (Istio)
- [ ] Advanced traffic management
- [ ] Canary deployments
- [ ] Blue-green deployments
- [ ] GitOps (ArgoCD)
- [ ] Multi-region deployment
