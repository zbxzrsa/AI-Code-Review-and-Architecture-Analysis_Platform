# Private/Offline Deployment Guide

This guide covers deploying the AI Code Review Platform in private networks or air-gapped environments where external API access is restricted or unavailable.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Local Model Setup](#local-model-setup)
4. [Air-Gapped Installation](#air-gapped-installation)
5. [Network Configuration](#network-configuration)
6. [Caching Strategies](#caching-strategies)
7. [Updates and Maintenance](#updates-and-maintenance)
8. [Compliance Considerations](#compliance-considerations)

---

## Overview

### Deployment Modes

| Mode           | External API | Local Models | Use Case                           |
| -------------- | ------------ | ------------ | ---------------------------------- |
| **Full Cloud** | ✅ Yes       | ❌ No        | Standard SaaS deployment           |
| **Hybrid**     | ✅ Yes       | ✅ Yes       | Sensitive data with cloud fallback |
| **Private**    | ❌ No        | ✅ Yes       | Corporate network, no egress       |
| **Air-Gapped** | ❌ No        | ✅ Yes       | HIPAA, classified environments     |

### Architecture for Offline Mode

```
┌─────────────────────────────────────────────────────────────────┐
│                    AIR-GAPPED NETWORK                           │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   Application   │    │  Local Models   │                    │
│  │    Services     │◄──►│   (GPU Nodes)   │                    │
│  └────────┬────────┘    └─────────────────┘                    │
│           │                                                     │
│  ┌────────▼────────┐    ┌─────────────────┐                    │
│  │  Response Cache │    │ Embedding Cache │                    │
│  │    (Redis)      │    │   (Vector DB)   │                    │
│  └─────────────────┘    └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Update Station                        │   │
│  │  (Receives signed updates via removable media)           │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Hardware Requirements

| Component   | Minimum                | Recommended        |
| ----------- | ---------------------- | ------------------ |
| **CPU**     | 16 cores               | 32+ cores          |
| **RAM**     | 64 GB                  | 128+ GB            |
| **GPU**     | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (80GB) |
| **Storage** | 500 GB SSD             | 2 TB NVMe          |
| **Network** | 1 Gbps internal        | 10 Gbps internal   |

### Software Requirements

- Docker 24.0+
- NVIDIA Container Toolkit (for GPU support)
- Kubernetes 1.28+ (optional)
- PostgreSQL 16
- Redis 7

### Model Files (Download Before Air-Gap)

```bash
# Download models from HuggingFace before going offline
# Total size: ~100 GB

# CodeLlama 34B (4-bit quantized)
huggingface-cli download codellama/CodeLlama-34b-Instruct-hf \
  --local-dir /models/codellama-34b

# DeepSeek Coder 33B (4-bit quantized)
huggingface-cli download deepseek-ai/deepseek-coder-33b-instruct \
  --local-dir /models/deepseek-coder-33b

# Embedding model for semantic cache
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 \
  --local-dir /models/embeddings
```

---

## Local Model Setup

### 1. Model Server Configuration

Create `local-models/docker-compose.yml`:

```yaml
version: "3.8"

services:
  # vLLM for CodeLlama
  codellama-server:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8100:8000"
    volumes:
      - /models/codellama-34b:/models/codellama-34b:ro
    environment:
      - CUDA_VISIBLE_DEVICES=0
    command: >
      --model /models/codellama-34b
      --tensor-parallel-size 1
      --quantization awq
      --max-model-len 16384
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # vLLM for DeepSeek Coder
  deepseek-server:
    image: vllm/vllm-openai:latest
    runtime: nvidia
    ports:
      - "8101:8000"
    volumes:
      - /models/deepseek-coder-33b:/models/deepseek-coder-33b:ro
    environment:
      - CUDA_VISIBLE_DEVICES=1
    command: >
      --model /models/deepseek-coder-33b
      --tensor-parallel-size 1
      --quantization awq
      --max-model-len 16384
      --gpu-memory-utilization 0.9
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Embedding server
  embedding-server:
    image: ghcr.io/huggingface/text-embeddings-inference:latest
    ports:
      - "8102:80"
    volumes:
      - /models/embeddings:/models/embeddings:ro
    command: >
      --model-id /models/embeddings
      --port 80
```

### 2. Model Registry Configuration

Update `ai_core/registries/model_registry.yaml`:

```yaml
models:
  - id: codellama-34b-local
    provider: local
    version: "1.0.0"
    endpoint: "http://codellama-server:8000/v1"
    capabilities:
      - code-review
      - security-audit
      - documentation
    constraints:
      max_tokens: 16384
      cost_per_1k_input: 0.0001
      cost_per_1k_output: 0.0001
      rate_limit_rpm: 1000
    slo:
      p95_latency_ms: 5000
      availability: 99.9
    status: active
    allowed_versions: [v1, v2, v3]
    offline_only: true

  - id: deepseek-coder-33b-local
    provider: local
    version: "1.0.0"
    endpoint: "http://deepseek-server:8000/v1"
    capabilities:
      - code-review
      - code-completion
    constraints:
      max_tokens: 16384
      cost_per_1k_input: 0.0001
      cost_per_1k_output: 0.0001
    status: active
    allowed_versions: [v1, v2, v3]
    offline_only: true

# Model groups for offline
model_groups:
  - name: offline-primary
    description: "Primary models for offline deployment"
    models:
      - codellama-34b-local
      - deepseek-coder-33b-local
    timeout_cascade_ms: [5000, 5000]
```

### 3. Routing Policy for Offline Mode

```yaml
routing_policies:
  - id: offline-routing
    version: "1.0.0"
    priority: 1000 # Highest priority
    status: active

    conditions:
      environment: offline

    rules:
      - name: use-local-models
        condition:
          type: always
        actions:
          - model_group: offline-primary
            budget:
              max_cost_usd: 0.01 # Local is cheap
              max_latency_ms: 10000
            cache:
              enabled: true
              ttl_hours: 168 # 1 week
```

---

## Air-Gapped Installation

### 1. Package Preparation (On Connected Machine)

```bash
#!/bin/bash
# prepare-airgap-package.sh

PACKAGE_DIR="/tmp/airgap-package"
mkdir -p $PACKAGE_DIR/{images,charts,models,configs}

# Save Docker images
IMAGES=(
  "gcr.io/coderev-platform/vcai:latest"
  "gcr.io/coderev-platform/crai:latest"
  "gcr.io/coderev-platform/lifecycle-controller:latest"
  "vllm/vllm-openai:latest"
  "postgres:16"
  "redis:7"
  "grafana/grafana:10.0.0"
  "prom/prometheus:v2.45.0"
)

for img in "${IMAGES[@]}"; do
  echo "Saving $img..."
  docker pull $img
  docker save $img | gzip > "$PACKAGE_DIR/images/$(echo $img | tr '/:' '_').tar.gz"
done

# Copy Helm charts
cp -r charts/ $PACKAGE_DIR/charts/

# Copy model files
cp -r /models/ $PACKAGE_DIR/models/

# Copy configurations
cp -r kubernetes/overlays/ $PACKAGE_DIR/configs/
cp ai_core/compliance/compliance_modes.yaml $PACKAGE_DIR/configs/

# Create checksum file
cd $PACKAGE_DIR
find . -type f -exec sha256sum {} \; > CHECKSUMS.sha256

# Sign the package
gpg --sign CHECKSUMS.sha256

# Create final archive
cd /tmp
tar -czvf airgap-package-$(date +%Y%m%d).tar.gz airgap-package/

echo "Package ready: /tmp/airgap-package-$(date +%Y%m%d).tar.gz"
```

### 2. Transfer to Air-Gapped Environment

Transfer options:

- **Approved USB drive**: Scan for malware, verify checksums
- **Data diode**: One-way network transfer
- **Optical media**: DVD/Blu-ray for tamper evidence

### 3. Installation on Air-Gapped Network

```bash
#!/bin/bash
# install-airgap.sh

PACKAGE="/media/usb/airgap-package-20240101.tar.gz"
INSTALL_DIR="/opt/coderev"

# Verify package integrity
tar -xzf $PACKAGE -C /tmp
cd /tmp/airgap-package

# Verify GPG signature
gpg --verify CHECKSUMS.sha256.sig CHECKSUMS.sha256
if [ $? -ne 0 ]; then
  echo "ERROR: Signature verification failed!"
  exit 1
fi

# Verify checksums
sha256sum -c CHECKSUMS.sha256
if [ $? -ne 0 ]; then
  echo "ERROR: Checksum verification failed!"
  exit 1
fi

# Load Docker images
for img in images/*.tar.gz; do
  echo "Loading $img..."
  docker load < $img
done

# Copy configurations
mkdir -p $INSTALL_DIR
cp -r configs/ $INSTALL_DIR/
cp -r models/ $INSTALL_DIR/

# Apply Kubernetes manifests (if using K8s)
kubectl apply -k $INSTALL_DIR/configs/overlays/offline/

# Or start with Docker Compose
docker-compose -f $INSTALL_DIR/configs/docker-compose-offline.yml up -d

echo "Installation complete!"
```

---

## Network Configuration

### Internal DNS Setup

```yaml
# coredns/Corefile
.:53 {
hosts {
10.0.1.10 codellama-server
10.0.1.11 deepseek-server
10.0.1.12 embedding-server
10.0.1.20 postgres
10.0.1.21 redis
10.0.1.30 vcai-service
10.0.1.31 crai-service
10.0.1.40 prometheus
10.0.1.41 grafana
fallthrough
}
forward . /etc/resolv.conf
cache 300
log
errors
}
```

### Firewall Rules (No External Access)

```bash
# Block all external traffic
iptables -P OUTPUT DROP
iptables -P INPUT DROP
iptables -P FORWARD DROP

# Allow loopback
iptables -A INPUT -i lo -j ACCEPT
iptables -A OUTPUT -o lo -j ACCEPT

# Allow internal network (10.0.0.0/8)
iptables -A INPUT -s 10.0.0.0/8 -j ACCEPT
iptables -A OUTPUT -d 10.0.0.0/8 -j ACCEPT

# Allow established connections
iptables -A INPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
iptables -A OUTPUT -m state --state ESTABLISHED,RELATED -j ACCEPT
```

---

## Caching Strategies

### Semantic Response Cache

```python
# services/cache/semantic_cache.py
import hashlib
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticCache:
    """
    Cache responses based on semantic similarity of inputs.
    Useful in offline mode to reduce redundant model calls.
    """

    def __init__(self, redis_client, similarity_threshold=0.95):
        self.redis = redis_client
        self.threshold = similarity_threshold
        self.embedder = SentenceTransformer('/models/embeddings')

    def get_or_compute(self, code: str, compute_fn):
        # Compute embedding
        embedding = self.embedder.encode(code)

        # Search for similar cached responses
        cached = self._find_similar(embedding)
        if cached:
            return cached['response']

        # Compute new response
        response = compute_fn(code)

        # Cache the result
        self._store(code, embedding, response)

        return response

    def _find_similar(self, embedding):
        # Search Redis for similar embeddings
        # Returns cached response if similarity > threshold
        pass

    def _store(self, code, embedding, response):
        # Store in Redis with embedding for future similarity search
        key = f"semantic:{hashlib.sha256(code.encode()).hexdigest()}"
        self.redis.hset(key, mapping={
            'code_hash': hashlib.sha256(code.encode()).hexdigest(),
            'embedding': embedding.tobytes(),
            'response': response,
            'created_at': datetime.utcnow().isoformat()
        })
        self.redis.expire(key, 7 * 24 * 3600)  # 1 week TTL
```

### Embedding Pre-computation

```bash
# Pre-compute embeddings for common patterns
python scripts/precompute_embeddings.py \
  --patterns /data/common-code-patterns/ \
  --output /cache/embeddings/ \
  --model /models/embeddings
```

---

## Updates and Maintenance

### Secure Update Process

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Update Server   │────►│   Data Diode /   │────►│  Air-Gapped      │
│  (Connected)     │     │   Approved USB   │     │  Environment     │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                         │                        │
        ▼                         ▼                        ▼
   Build Updates            Verify & Scan            Apply Updates
   Sign Packages            Transfer Media           Verify Signatures
   Create Checksums         Document Chain           Run Tests
```

### Update Verification Script

```bash
#!/bin/bash
# verify-update.sh

UPDATE_FILE="$1"

# Extract update
tar -xzf $UPDATE_FILE -C /tmp/update

# Verify GPG signature
gpg --verify /tmp/update/SIGNATURE.asc /tmp/update/MANIFEST.json
if [ $? -ne 0 ]; then
  echo "FAILED: Invalid signature"
  exit 1
fi

# Verify checksums
cd /tmp/update
sha256sum -c CHECKSUMS.sha256
if [ $? -ne 0 ]; then
  echo "FAILED: Checksum mismatch"
  exit 1
fi

# Verify version is newer
CURRENT_VERSION=$(cat /opt/coderev/VERSION)
NEW_VERSION=$(jq -r '.version' /tmp/update/MANIFEST.json)

if [ "$NEW_VERSION" \< "$CURRENT_VERSION" ]; then
  echo "FAILED: Update version is older than current"
  exit 1
fi

echo "VERIFIED: Update is valid and ready to apply"
```

### Rollback Procedure

```bash
#!/bin/bash
# rollback.sh

# Keep last 3 versions
BACKUP_DIR="/opt/coderev/backups"

# Create backup before update
backup_current() {
  VERSION=$(cat /opt/coderev/VERSION)
  tar -czf $BACKUP_DIR/backup-$VERSION-$(date +%Y%m%d%H%M%S).tar.gz \
    /opt/coderev/configs \
    /opt/coderev/data
}

# Rollback to previous version
rollback() {
  LATEST_BACKUP=$(ls -t $BACKUP_DIR/backup-*.tar.gz | head -1)

  echo "Rolling back to: $LATEST_BACKUP"

  # Stop services
  docker-compose -f /opt/coderev/docker-compose.yml down

  # Restore backup
  tar -xzf $LATEST_BACKUP -C /

  # Restart services
  docker-compose -f /opt/coderev/docker-compose.yml up -d

  echo "Rollback complete"
}
```

---

## Compliance Considerations

### HIPAA Requirements

- ✅ **No PHI transmission**: All processing stays on-premise
- ✅ **Encryption at rest**: PostgreSQL TDE, Redis encryption
- ✅ **Access controls**: RBAC with audit logging
- ✅ **Audit trail**: Immutable logs with signatures
- ✅ **BAA not required**: No third-party data processing

### FedRAMP Requirements

- ✅ **Boundary controls**: Air-gapped deployment
- ✅ **FIPS 140-2**: Use certified crypto modules
- ✅ **Continuous monitoring**: Local Prometheus/Grafana
- ✅ **Vulnerability scanning**: Offline Trivy scans
- ✅ **Change management**: Signed update packages

### Documentation Requirements

Maintain these records:

1. **System inventory**: All hardware/software components
2. **Network diagram**: Internal network topology
3. **Access log**: Who accessed what, when
4. **Update log**: All updates applied, with signatures
5. **Incident log**: Any security events

---

## Troubleshooting

### Common Issues

#### Model Server Not Starting

```bash
# Check GPU availability
nvidia-smi

# Check container logs
docker logs codellama-server

# Verify model files
ls -la /models/codellama-34b/

# Check CUDA version compatibility
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

#### High Latency

```bash
# Check GPU utilization
watch -n 1 nvidia-smi

# Check batch queue
curl http://codellama-server:8000/v1/queue

# Increase batch size if GPU underutilized
# Reduce max_model_len if OOM errors
```

#### Cache Miss Rate High

```bash
# Check cache statistics
redis-cli INFO stats

# Verify embedding server
curl http://embedding-server:80/health

# Adjust similarity threshold
# Lower threshold = more cache hits (but potentially lower quality matches)
```

---

## Support

For offline deployment support:

- Email: enterprise@example.com
- Documentation: /docs/offline/
- Update channel: https://updates.example.com/offline/ (download before air-gap)
