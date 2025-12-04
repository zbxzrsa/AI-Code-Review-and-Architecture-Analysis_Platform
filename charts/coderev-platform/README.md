# CodeRev Platform Helm Chart

Deploy the AI Code Review Platform with three-version self-evolving architecture to Kubernetes.

## Prerequisites

- Kubernetes 1.28+
- Helm 3.12+
- kubectl configured
- Argo Rollouts (optional, for gray-scale deployments)

## Installation

### Add dependencies

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo add argo https://argoproj.github.io/argo-helm
helm repo update
```

### Install the chart

```bash
# Basic installation
helm install coderev ./charts/coderev-platform

# With custom values
helm install coderev ./charts/coderev-platform -f values-production.yaml

# Specific namespace
helm install coderev ./charts/coderev-platform -n coderev --create-namespace
```

### Upgrade

```bash
helm upgrade coderev ./charts/coderev-platform -f values-production.yaml
```

### Uninstall

```bash
helm uninstall coderev
```

## Configuration

### Version Configuration

```yaml
versions:
  v1:
    enabled: true
    namespace: platform-v1-exp
    replicas: 1
    autoscaling:
      enabled: true
      minReplicas: 0 # Scale to zero when idle
      maxReplicas: 10

  v2:
    enabled: true
    namespace: platform-v2-stable
    replicas: 3
    autoscaling:
      enabled: true
      minReplicas: 3
      maxReplicas: 20

  v3:
    enabled: true
    namespace: platform-v3-legacy
    replicas: 1
```

### AI Models

```yaml
aiModels:
  openai:
    enabled: true
    apiKeySecret: openai-secret

  anthropic:
    enabled: true
    apiKeySecret: anthropic-secret

  # For offline deployment
  local:
    enabled: false
    codellama:
      enabled: true
      modelPath: /models/codellama-34b
```

### Compliance Modes

```yaml
compliance:
  mode: standard # standard, financial, hipaa, fedramp, gdpr

  audit:
    enabled: true
    retentionDays: 365
```

### Gray-Scale Rollout

```yaml
gateway:
  grayScale:
    enabled: true
    phases: [1, 5, 25, 50, 100]
    pauseBetweenPhases: true
```

## Common Configurations

### Production

```yaml
# values-production.yaml
versions:
  v2:
    replicas: 5
    autoscaling:
      minReplicas: 5
      maxReplicas: 50

monitoring:
  prometheus:
    enabled: true
  grafana:
    enabled: true

compliance:
  mode: standard
  audit:
    enabled: true
```

### HIPAA Compliant

```yaml
# values-hipaa.yaml
compliance:
  mode: hipaa
  audit:
    enabled: true
    retentionDays: 2190 # 6 years

aiModels:
  openai:
    enabled: false
  anthropic:
    enabled: false
  local:
    enabled: true

networkPolicies:
  enabled: true
  strictIsolation: true
```

### Air-Gapped

```yaml
# values-airgap.yaml
global:
  imageRegistry: internal-registry.local

aiModels:
  openai:
    enabled: false
  anthropic:
    enabled: false
  local:
    enabled: true
    codellama:
      enabled: true
    deepseek:
      enabled: true

semanticCache:
  enabled: true
```

## Verifying Installation

```bash
# Check all pods
kubectl get pods -A | grep platform

# Check services
kubectl get svc -n platform-v2-stable

# Check ingress
kubectl get ingress -n platform-v2-stable

# Test health endpoint
curl https://api.coderev.example.com/health
```

## Troubleshooting

### Pods not starting

```bash
kubectl describe pod -n platform-v2-stable <pod-name>
kubectl logs -n platform-v2-stable <pod-name>
```

### Database connection issues

```bash
kubectl get secret database-credentials -n platform-v2-stable -o yaml
kubectl exec -it -n platform-v2-stable <pod-name> -- env | grep DATABASE
```

### OPA not evaluating

```bash
kubectl logs -n platform-control-plane -l app=opa
kubectl exec -n platform-control-plane deployment/opa -- curl localhost:8181/health
```

## Values Reference

| Parameter                     | Description                 | Default    |
| ----------------------------- | --------------------------- | ---------- |
| `versions.v1.enabled`         | Enable V1 experiment        | `true`     |
| `versions.v1.replicas`        | V1 replica count            | `1`        |
| `versions.v2.enabled`         | Enable V2 production        | `true`     |
| `versions.v2.replicas`        | V2 replica count            | `3`        |
| `versions.v3.enabled`         | Enable V3 legacy            | `true`     |
| `vcai.image.tag`              | VCAI image tag              | `latest`   |
| `lifecycleController.enabled` | Enable lifecycle controller | `true`     |
| `evaluationPipeline.enabled`  | Enable evaluation pipeline  | `true`     |
| `opa.enabled`                 | Enable OPA policy engine    | `true`     |
| `argoRollouts.enabled`        | Enable Argo Rollouts        | `true`     |
| `postgresql.enabled`          | Deploy PostgreSQL           | `true`     |
| `redis.enabled`               | Deploy Redis                | `true`     |
| `compliance.mode`             | Compliance mode             | `standard` |
| `networkPolicies.enabled`     | Enable network policies     | `true`     |

See `values.yaml` for complete configuration options.
