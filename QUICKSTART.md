# Quick Start Guide

Get the AI Code Review Platform running in 5 minutes!

## Prerequisites

- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- 8GB RAM minimum
- 10GB disk space

## Option 1: Docker Compose (Recommended for Local Development)

### Step 1: Clone and Setup

```bash
cd AI-Code-Review-and-Architecture-Analysis_Platform
```

### Step 2: Configure Environment

```bash
cat > .env << EOF
PRIMARY_AI_API_KEY=sk-your-openai-key
SECONDARY_AI_API_KEY=sk-ant-your-anthropic-key
EOF
```

**Note**: Get API keys from:

- OpenAI: https://platform.openai.com/api-keys
- Anthropic: https://console.anthropic.com/

### Step 3: Start Services

```bash
docker-compose up -d
```

### Step 4: Verify Services

```bash
# Check all containers are running
docker-compose ps

# Expected output:
# NAME                    STATUS
# platform-postgres       Up
# platform-v2-api         Up
# platform-v1-api         Up
# platform-v3-api         Up
# platform-prometheus     Up
# platform-grafana        Up
```

### Step 5: Access Services

| Service                | URL                   | Credentials |
| ---------------------- | --------------------- | ----------- |
| V2 Production API      | http://localhost:8001 | None        |
| V1 Experimentation API | http://localhost:8002 | None        |
| V3 Quarantine API      | http://localhost:8003 | None        |
| Prometheus             | http://localhost:9090 | None        |
| Grafana                | http://localhost:3000 | admin/admin |

## Option 2: Kubernetes (Production)

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- Docker registry access

### Step 1: Build Docker Images

```bash
docker build -t your-registry/platform-v2:latest backend/v2-production/
docker build -t your-registry/platform-v1:latest backend/v1-experimentation/
docker build -t your-registry/platform-v3:latest backend/v3-quarantine/

docker push your-registry/platform-v2:latest
docker push your-registry/platform-v1:latest
docker push your-registry/platform-v3:latest
```

### Step 2: Create Namespaces

```bash
kubectl apply -f kubernetes/namespaces/namespaces.yaml
```

### Step 3: Configure Secrets

```bash
# Edit with your actual values
kubectl apply -f kubernetes/config/secrets.yaml
kubectl apply -f kubernetes/config/configmap.yaml
```

### Step 4: Deploy Services

```bash
kubectl apply -f kubernetes/network-policies/isolation.yaml
kubectl apply -f kubernetes/deployments/v2-deployment.yaml
kubectl apply -f kubernetes/deployments/v1-deployment.yaml
kubectl apply -f kubernetes/deployments/v3-deployment.yaml
```

### Step 5: Verify Deployment

```bash
kubectl get pods -n platform-v2-stable
kubectl get pods -n platform-v1-exp
kubectl get pods -n platform-v3-quarantine
```

## First Test: Code Review

### Using V2 Production API

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

**Expected Response:**

```json
{
  "review_id": "review_1701505200000",
  "timestamp": "2025-12-02T12:10:00Z",
  "language": "python",
  "issues": [],
  "suggestions": ["Add type hints", "Add docstring"],
  "confidence_score": 0.95,
  "analysis_time_ms": 1250.5,
  "model_used": "gpt-4"
}
```

## First Experiment: Create and Run

### Step 1: Create Experiment

```bash
curl -X POST http://localhost:8002/api/v1/experiments/create \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Test new prompt",
    "description": "Testing improved prompt template",
    "primary_model": "gpt-4",
    "secondary_model": "claude-3-opus-20240229",
    "prompt_template": "Review this code: {code}",
    "routing_strategy": "primary",
    "tags": ["test"]
  }'
```

**Response includes `id` field** - save this for next step

### Step 2: Run Experiment

```bash
curl -X POST http://localhost:8002/api/v1/experiments/run/{experiment_id} \
  -H "Content-Type: application/json" \
  -d '{
    "code_samples": ["def test(): pass"],
    "language": "python"
  }'
```

### Step 3: Check Results

```bash
curl http://localhost:8002/api/v1/experiments/{experiment_id}
```

## Monitoring Dashboard

### Access Grafana

1. Open http://localhost:3000
2. Login: `admin` / `admin`
3. Click "Dashboards" to see available dashboards
4. Explore metrics:
   - V2 Production SLO Compliance
   - V1 Experiment Progress
   - V3 Quarantine Statistics

## Troubleshooting

### Services not starting

```bash
# Check logs
docker-compose logs platform-v2-api
docker-compose logs platform-postgres

# Restart services
docker-compose restart
```

### API returning errors

```bash
# Check if database is ready
docker-compose exec postgres psql -U postgres -d platform -c "SELECT 1"

# Check API logs
docker-compose logs platform-v2-api
```

### Port already in use

```bash
# Find process using port
lsof -i :8001

# Kill process or use different port
docker-compose down
# Edit docker-compose.yml to change ports
docker-compose up -d
```

## Next Steps

1. **Read Documentation**

   - [Architecture Guide](docs/architecture.md)
   - [API Reference](docs/api-reference.md)
   - [Deployment Guide](docs/deployment.md)

2. **Configure AI Providers**

   - Add your OpenAI API key
   - Add your Anthropic API key
   - Test with different models

3. **Create Experiments**

   - Design experiment hypothesis
   - Test new prompts
   - Evaluate results
   - Promote to V2 if successful

4. **Monitor Performance**

   - Check SLO compliance
   - Review metrics
   - Optimize configurations

5. **Deploy to Production**
   - Follow [Deployment Guide](docs/deployment.md)
   - Set up monitoring and alerting
   - Configure backups
   - Train team

## Common Commands

### Docker Compose

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f platform-v2-api

# Restart specific service
docker-compose restart platform-v2-api

# Remove all data
docker-compose down -v
```

### Kubernetes

```bash
# Check pod status
kubectl get pods -n platform-v2-stable

# View logs
kubectl logs -n platform-v2-stable <pod-name>

# Port forward
kubectl port-forward -n platform-v2-stable svc/platform-v2-api 8001:8000

# Scale deployment
kubectl scale deployment platform-v2-api -n platform-v2-stable --replicas=5

# Delete deployment
kubectl delete deployment platform-v2-api -n platform-v2-stable
```

## Getting Help

- **Documentation**: Check `docs/` directory
- **Issues**: Review [CONTRIBUTING.md](CONTRIBUTING.md)
- **API Docs**: See [API Reference](docs/api-reference.md)
- **Operations**: See [Operations Runbook](docs/operations.md)

## What's Next?

- Explore the three-version architecture
- Test different AI models
- Create your first experiment
- Monitor performance metrics
- Deploy to your Kubernetes cluster

Happy experimenting! 🚀
