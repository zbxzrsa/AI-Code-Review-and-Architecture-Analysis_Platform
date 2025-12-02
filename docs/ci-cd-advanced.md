# Advanced CI/CD Features Guide

## Overview

Comprehensive guide for advanced CI/CD features including SBOM generation, image signing, security scanning, and deployment automation.

---

## Table of Contents

1. [SBOM Generation](#sbom-generation)
2. [Image Signing](#image-signing)
3. [Security Scanning](#security-scanning)
4. [Build Optimization](#build-optimization)
5. [Deployment Strategies](#deployment-strategies)
6. [Monitoring & Observability](#monitoring--observability)
7. [Troubleshooting](#troubleshooting)

---

## SBOM Generation

### What is SBOM?

Software Bill of Materials (SBOM) is a complete list of all components, libraries, and dependencies in a software application.

### Implementation

```yaml
- name: Generate SBOM
  uses: anchore/sbom-action@v0
  with:
    image: ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/${{ matrix.service }}:${{ github.sha }}
    format: spdx-json
    output-file: sbom-${{ matrix.service }}.spdx.json
```

### Supported Formats

- **SPDX JSON**: Standard format for software supply chain
- **SPDX XML**: XML variant of SPDX
- **CycloneDX**: Alternative format for dependency tracking
- **Table**: Human-readable format

### SBOM Contents

```json
{
  "SPDXID": "SPDXRef-DOCUMENT",
  "spdxVersion": "SPDX-2.3",
  "creationInfo": {
    "created": "2024-12-02T16:00:00Z",
    "creators": ["Tool: syft-0.68.1"]
  },
  "packages": [
    {
      "SPDXID": "SPDXRef-Package",
      "name": "python",
      "versionInfo": "3.11",
      "filesAnalyzed": false,
      "downloadLocation": "NOASSERTION"
    }
  ]
}
```

### Usage

```bash
# Generate SBOM locally
syft <image> -o spdx-json > sbom.spdx.json

# View SBOM
cat sbom.spdx.json | jq '.packages | length'

# Validate SBOM
spdx-tools validate sbom.spdx.json
```

### Benefits

✅ Supply chain transparency
✅ Vulnerability tracking
✅ License compliance
✅ Dependency management
✅ Audit trail

---

## Image Signing

### What is Image Signing?

Cryptographic signing of container images ensures authenticity and integrity.

### Implementation

```yaml
- name: Install Cosign
  uses: sigstore/cosign-installer@v3
  with:
    cosign-release: "v2.2.0"

- name: Sign Docker image
  env:
    COSIGN_EXPERIMENTAL: 1
  run: |
    cosign sign --yes \
      ${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/${{ matrix.service }}@${{ steps.build.outputs.digest }}
```

### Keyless Signing

Using OIDC token for keyless signing:

```bash
# Sign with keyless approach
cosign sign --yes \
  gcr.io/project-id/service@sha256:abc123...

# Verify signature
cosign verify \
  gcr.io/project-id/service@sha256:abc123...
```

### Key Management

```bash
# Generate key pair
cosign generate-key-pair

# Store private key securely
export COSIGN_KEY=<path-to-private-key>

# Sign with key
cosign sign --key $COSIGN_KEY \
  gcr.io/project-id/service:latest

# Verify with public key
cosign verify --key cosign.pub \
  gcr.io/project-id/service:latest
```

### Verification

```bash
# Verify image signature
cosign verify \
  --certificate-identity=https://github.com/org/repo \
  --certificate-oidc-issuer=https://token.actions.githubusercontent.com \
  gcr.io/project-id/service:latest

# Check signature details
cosign verify --certificate-identity-regexp=.* \
  gcr.io/project-id/service:latest
```

### Benefits

✅ Image authenticity verification
✅ Tamper detection
✅ Supply chain security
✅ Compliance requirements
✅ Audit trail

---

## Security Scanning

### Multi-Layer Security

#### 1. Code Quality

```yaml
- name: Lint with ruff
  run: ruff check backend/ --exit-zero

- name: Format check with black
  run: black --check backend/ --exit-zero
```

#### 2. Secret Detection

```yaml
- name: Run Gitleaks
  uses: gitleaks/gitleaks-action@v2
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

#### 3. SAST (Static Application Security Testing)

```yaml
- name: Run Semgrep
  uses: returntocorp/semgrep-action@v1
  with:
    config: >-
      p/security-audit
      p/secrets
      p/owasp-top-ten
```

#### 4. Dependency Scanning

```yaml
- name: Run OWASP Dependency Check
  uses: dependency-check/Dependency-Check_Action@main
  with:
    path: "backend/"
    format: "SARIF"
```

#### 5. Container Scanning

```yaml
- name: Run Trivy
  uses: aquasecurity/trivy-action@master
  with:
    scan-type: "fs"
    scan-ref: "backend/"
    format: "sarif"
```

### Security Scanning Pipeline

```
Code Push
    ↓
Lint & Format Check
    ↓
Secret Detection (Gitleaks)
    ↓
SAST (Semgrep)
    ↓
Dependency Check (OWASP)
    ↓
Container Scan (Trivy)
    ↓
Build & Sign
    ↓
Deploy
```

---

## Build Optimization

### Docker BuildKit

```yaml
build-args: |
  BUILDKIT_INLINE_CACHE=1
  BUILD_DATE=$(date -u +'%Y-%m-%dT%H:%M:%SZ')
  VCS_REF=${{ github.sha }}
  VERSION=${{ steps.meta.outputs.version }}
```

### Layer Caching

```yaml
cache-from: type=gha
cache-to: type=gha,mode=max
```

### Multi-Stage Builds

```dockerfile
# Stage 1: Build
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
COPY . .
CMD ["python", "-m", "uvicorn", "main:app"]
```

### Build Performance

- **Cache hit rate**: 80-90%
- **Build time**: 2-5 minutes
- **Push time**: 1-2 minutes
- **Total**: 5-10 minutes

---

## Deployment Strategies

### Staging Deployment

```yaml
deploy-staging:
  if: github.ref == 'refs/heads/develop' && github.event_name == 'push'
  steps:
    - name: Deploy to GKE
      run: |
        kubectl apply -f kubernetes/namespaces.yaml
        kubectl apply -f kubernetes/deployments.yaml
        kubectl rollout status deployment/auth-service

    - name: Run smoke tests
      run: |
        kubectl run smoke-test \
          --image=${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/smoke-tests:latest
```

### Production Deployment

```yaml
deploy-production:
  if: github.ref == 'refs/heads/main' && github.event_name == 'push'
  environment:
    name: production
    url: https://api.coderev.example.com
  steps:
    - name: Deploy to GKE
      run: kubectl apply -f kubernetes/

    - name: Health checks
      run: |
        for i in {1..30}; do
          curl -f https://api.coderev.example.com/health && exit 0
          sleep 10
        done
```

### Deployment Approval

```yaml
environment:
  name: production
  url: https://api.coderev.example.com
  reviewers:
    - team-leads
  deployment_branch_policy:
    protected_branches: true
    custom_deployment_rules: false
```

---

## Monitoring & Observability

### Build Metrics

```yaml
- name: Publish build metrics
  run: |
    echo "Build Duration: ${{ job.duration }}"
    echo "Build Status: ${{ job.status }}"
    echo "Commit: ${{ github.sha }}"
```

### Deployment Notifications

```yaml
- name: Notify Slack
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "Deployment ${{ job.status }}",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "*Deployment Status*\nStatus: ${{ job.status }}\nCommit: ${{ github.sha }}"
            }
          }
        ]
      }
```

### Artifact Management

```yaml
- name: Upload SBOM reports
  uses: actions/upload-artifact@v3
  with:
    name: sbom-reports
    path: sbom-*.spdx.json
    retention-days: 30
```

---

## Troubleshooting

### Common Issues

#### 1. Build Cache Not Working

```bash
# Check cache status
docker buildx du

# Clear cache
docker buildx prune --all

# Rebuild with fresh cache
docker buildx build --no-cache .
```

#### 2. Image Signing Fails

```bash
# Check Cosign installation
cosign version

# Verify OIDC token
echo $COSIGN_EXPERIMENTAL

# Debug signing
cosign sign --debug \
  gcr.io/project-id/service:latest
```

#### 3. SBOM Generation Issues

```bash
# Check Syft installation
syft --version

# Generate SBOM manually
syft gcr.io/project-id/service:latest -o spdx-json

# Validate SBOM
spdx-tools validate sbom.spdx.json
```

#### 4. Deployment Failures

```bash
# Check GKE cluster
gcloud container clusters list

# Get cluster credentials
gcloud container clusters get-credentials cluster-name

# Check deployment status
kubectl get deployments -n platform-v2-stable

# View pod logs
kubectl logs -f deployment/auth-service -n platform-v2-stable
```

### Debug Commands

```bash
# View GitHub Actions logs
gh run view <run-id> --log

# List recent runs
gh run list --limit 10

# Cancel run
gh run cancel <run-id>

# Re-run failed jobs
gh run rerun <run-id> --failed
```

---

## Best Practices

### Security

✅ Use keyless signing for GitHub Actions
✅ Scan all images before deployment
✅ Generate SBOM for all releases
✅ Verify image signatures before pulling
✅ Use secrets management for sensitive data

### Performance

✅ Enable Docker BuildKit caching
✅ Use multi-stage builds
✅ Minimize image layers
✅ Cache dependencies
✅ Parallel job execution

### Reliability

✅ Implement health checks
✅ Use rolling updates
✅ Test in staging first
✅ Monitor deployments
✅ Maintain audit logs

### Compliance

✅ Generate SBOM for all releases
✅ Sign all container images
✅ Maintain deployment records
✅ Document security scanning
✅ Track dependency updates

---

## Advanced Topics

### GitOps Integration

```yaml
- name: Update GitOps repository
  run: |
    git clone https://github.com/org/gitops-repo
    cd gitops-repo
    kustomize edit set image service=${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/service:${{ github.sha }}
    git commit -am "Update service image"
    git push
```

### Canary Deployments

```yaml
- name: Deploy canary
  run: |
    kubectl set image deployment/service \
      service=${{ env.REGISTRY }}/${{ env.PROJECT_ID }}/service:${{ github.sha }} \
      -n platform-v2-stable --record
    kubectl rollout pause deployment/service -n platform-v2-stable
```

### Blue-Green Deployments

```yaml
- name: Deploy green environment
  run: |
    kubectl apply -f kubernetes/green-deployment.yaml
    kubectl service set selector service-green app=service-green
```

---

## References

- [Cosign Documentation](https://docs.sigstore.dev/cosign/overview/)
- [SBOM Specification](https://spdx.dev/)
- [Docker BuildKit](https://docs.docker.com/build/buildkit/)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Kubernetes Deployment](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
