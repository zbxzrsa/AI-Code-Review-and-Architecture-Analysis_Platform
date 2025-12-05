# Deployment Guide

## Local Development with Docker Compose

### Prerequisites

- Docker and Docker Compose installed
- Python 3.10+
- Git

### Quick Start

1. **Clone the repository**

```bash
git clone <repository-url>
cd AI-Code-Review-and-Architecture-Analysis_Platform
```

2. **Set environment variables**

```bash
# Create .env file
cat > .env << EOF
PRIMARY_AI_API_KEY=sk-your-openai-key
SECONDARY_AI_API_KEY=sk-ant-your-anthropic-key
EOF
```

3. **Start all services**

```bash
docker-compose up -d
```

4. **Initialize databases**

```bash
docker-compose exec postgres psql -U postgres -d platform -f /docker-entrypoint-initdb.d/01-init.sql
```

5. **Verify services**

```bash
# V2 Production API
curl http://localhost:8001/

# V1 Experimentation API
curl http://localhost:8002/

# V3 Quarantine API
curl http://localhost:8003/

# Prometheus
curl http://localhost:9090/

# Grafana
open http://localhost:3000
```

### Service Ports

| Service    | Port | Purpose                     |
| ---------- | ---- | --------------------------- |
| V2 API     | 8001 | Production endpoint         |
| V2 Metrics | 9001 | Prometheus metrics          |
| V1 API     | 8002 | Experimentation endpoint    |
| V1 Metrics | 9002 | Prometheus metrics          |
| V3 API     | 8003 | Quarantine endpoint         |
| V3 Metrics | 9003 | Prometheus metrics          |
| PostgreSQL | 5432 | Database                    |
| Prometheus | 9090 | Metrics collection          |
| Grafana    | 3000 | Visualization (admin/admin) |

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Docker registry access
- PostgreSQL instance (external or in-cluster)

### Step 1: Build Docker Images

```bash
# Build V2 Production
docker build -t your-registry/platform-v2:latest backend/v2-production/
docker push your-registry/platform-v2:latest

# Build V1 Experimentation
docker build -t your-registry/platform-v1:latest backend/v1-experimentation/
docker push your-registry/platform-v1:latest

# Build V3 Quarantine
docker build -t your-registry/platform-v3:latest backend/v3-quarantine/
docker push your-registry/platform-v3:latest
```

### Step 2: Create Namespaces

```bash
kubectl apply -f kubernetes/namespaces/namespaces.yaml
```

### Step 3: Configure Secrets and ConfigMaps

```bash
# Update secrets with actual values
kubectl apply -f kubernetes/config/secrets.yaml
kubectl apply -f kubernetes/config/configmap.yaml
```

**Important**: Update the following in `kubernetes/config/secrets.yaml`:

- `db_password`: Secure database password
- `primary_ai_api_key`: OpenAI API key
- `secondary_ai_api_key`: Anthropic API key

### Step 4: Apply Network Policies

```bash
kubectl apply -f kubernetes/network-policies/isolation.yaml
```

### Step 5: Deploy Services

```bash
# Deploy V2 Production
kubectl apply -f kubernetes/deployments/v2-deployment.yaml

# Deploy V1 Experimentation
kubectl apply -f kubernetes/deployments/v1-deployment.yaml

# Deploy V3 Quarantine
kubectl apply -f kubernetes/deployments/v3-deployment.yaml
```

### Step 6: Verify Deployments

```bash
# Check all namespaces
kubectl get namespaces

# Check V2 deployment
kubectl -n platform-v2-stable get pods
kubectl -n platform-v2-stable get svc
kubectl -n platform-v2-stable get hpa

# Check V1 deployment
kubectl -n platform-v1-exp get pods
kubectl -n platform-v1-exp get svc

# Check V3 deployment
kubectl -n platform-v3-quarantine get pods
kubectl -n platform-v3-quarantine get svc

# Check network policies
kubectl -n platform-v2-stable get networkpolicies
kubectl -n platform-v1-exp get networkpolicies
kubectl -n platform-v3-quarantine get networkpolicies
```

### Step 7: Port Forwarding (for local testing)

```bash
# V2 Production API
kubectl -n platform-v2-stable port-forward svc/platform-v2-api 8001:8000

# V1 Experimentation API
kubectl -n platform-v1-exp port-forward svc/platform-v1-api 8002:8000

# V3 Quarantine API
kubectl -n platform-v3-quarantine port-forward svc/platform-v3-api 8003:8000
```

## Database Setup

### PostgreSQL Installation

```bash
# Using Helm (recommended)
helm repo add bitnami https://charts.bitnami.com/bitnami
helm install postgres bitnami/postgresql \
  --set auth.username=platform_user \
  --set auth.password=secure_password \
  --set auth.database=platform
```

### Initialize Schemas

```bash
# Connect to PostgreSQL
psql -h localhost -U postgres -d platform

# Run initialization script
\i database/schemas/init.sql

# Verify schemas
\dn

# Verify tables
\dt production.*
\dt experiments_v1.*
\dt quarantine.*
```

## Monitoring Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "platform-v2"
    static_configs:
      - targets: ["platform-v2-api:9090"]

  - job_name: "platform-v1"
    static_configs:
      - targets: ["platform-v1-api:9090"]

  - job_name: "platform-v3"
    static_configs:
      - targets: ["platform-v3-api:9090"]
```

### Grafana Dashboards

1. Access Grafana: http://localhost:3000
2. Login: admin/admin
3. Add Prometheus data source: http://prometheus:9090
4. Import dashboards from `monitoring/grafana/dashboards/`

## API Testing

### V2 Production - Code Review

```bash
curl -X POST http://localhost:8001/api/v1/code-review/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def hello():\n    print(\"Hello, World!\")",
    "language": "python",
    "focus_areas": ["security", "performance"],
    "include_architecture_analysis": true
  }'
```

### V1 Experimentation - Create Experiment

```bash
curl -X POST http://localhost:8002/api/v1/experiments/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test GPT-4 with new prompt",
    "description": "Testing improved prompt template",
    "primary_model": "gpt-4",
    "secondary_model": "claude-3-opus-20240229",
    "prompt_template": "Review this code: {code}",
    "routing_strategy": "primary",
    "tags": ["gpt-4", "prompt-v2"]
  }'
```

### V1 Experimentation - Run Experiment

```bash
curl -X POST http://localhost:8002/api/v1/experiments/run/{experiment_id} \
  -H "Content-Type: application/json" \
  -d '{
    "code_samples": ["def test(): pass"],
    "language": "python"
  }'
```

### V3 Quarantine - List Records

```bash
curl http://localhost:8003/api/v1/quarantine/records
```

## Troubleshooting

### Pods not starting

```bash
# Check pod logs
kubectl -n platform-v2-stable logs <pod-name>

# Describe pod for events
kubectl -n platform-v2-stable describe pod <pod-name>

# Check resource availability
kubectl describe nodes
```

### Database connection issues

```bash
# Test database connectivity
kubectl -n platform-v2-stable exec -it <pod-name> -- \
  psql -h postgres -U platform_user -d production -c "SELECT 1"
```

### Network policy issues

```bash
# Verify network policies are applied
kubectl -n platform-v2-stable get networkpolicies

# Test connectivity between pods
kubectl -n platform-v2-stable exec -it <pod-name> -- \
  curl http://platform-v2-api:8000/health/live
```

### Metrics not appearing

```bash
# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Verify metrics endpoint
curl http://localhost:9001/metrics
```

## Production Checklist

- [ ] Database backups configured
- [ ] SSL/TLS certificates installed
- [ ] Ingress controller configured
- [ ] Monitoring and alerting set up
- [ ] Log aggregation configured
- [ ] RBAC policies reviewed
- [ ] Network policies tested
- [ ] Disaster recovery plan documented
- [ ] Load testing completed
- [ ] Security audit performed
- [ ] Documentation updated
- [ ] Team trained on operations

## Scaling Considerations

### Horizontal Scaling

- V2: HPA automatically scales based on metrics
- V1: Manual scaling for controlled experimentation
- V3: Static single replica

### Vertical Scaling

- Adjust resource requests/limits in deployment manifests
- Monitor actual usage before adjusting

### Database Scaling

- Use read replicas for V3 (read-only)
- Connection pooling for V2 (production)
- Separate connection pool for V1 (experimentation)

## Backup and Recovery

### Database Backups

```bash
# Backup all schemas
pg_dump -U postgres platform > backup.sql

# Backup specific schema
pg_dump -U postgres -n production platform > backup_v2.sql

# Restore
psql -U postgres platform < backup.sql
```

### Version Rollback

```bash
# Rollback V2 deployment
kubectl -n platform-v2-stable rollout undo deployment/platform-v2-api

# Check rollout history
kubectl -n platform-v2-stable rollout history deployment/platform-v2-api
```
