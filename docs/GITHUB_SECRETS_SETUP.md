# GitHub Secrets Configuration Guide

> **Document Version**: 1.0.0  
> **Last Updated**: 2024-12-07

This document lists all the GitHub Secrets required for the CI/CD pipelines to function correctly.

---

## Required Secrets Overview

The IDE warnings about "Context access might be invalid" indicate that these secrets need to be configured in your GitHub repository settings. Navigate to:

**Settings → Secrets and variables → Actions → New repository secret**

---

## Secret Categories

### 1. Google Cloud Platform (GCP)

| Secret Name      | Description                                | Required For           |
| ---------------- | ------------------------------------------ | ---------------------- |
| `GCP_PROJECT_ID` | GCP Project ID (e.g., `my-project-123456`) | All GCP services       |
| `GCP_SA_KEY`     | Service Account JSON key for GCP auth      | GKE deployments        |
| `GCR_JSON_KEY`   | JSON key for Google Container Registry     | Docker push to GCR     |
| `GKE_SA_KEY`     | Service Account key for GKE cluster access | Kubernetes deployments |

**How to generate:**

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:github-actions@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Generate key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@$PROJECT_ID.iam.gserviceaccount.com

# Copy contents of key.json to GitHub Secret
cat key.json
```

---

### 2. Slack Notifications

| Secret Name              | Description                | Required For               |
| ------------------------ | -------------------------- | -------------------------- |
| `SLACK_WEBHOOK`          | Slack Incoming Webhook URL | Deployment notifications   |
| `SLACK_WEBHOOK_URL`      | Alternative webhook URL    | Pipeline notifications     |
| `SLACK_BOT_TOKEN`        | Slack Bot OAuth Token      | Advanced Slack integration |
| `SLACK_CHANNEL_ID`       | Target channel ID          | Message routing            |
| `SLACK_SECURITY_WEBHOOK` | Security alerts webhook    | Security scan alerts       |

**How to configure:**

1. Go to [Slack API](https://api.slack.com/apps)
2. Create a new app or use existing
3. Enable "Incoming Webhooks"
4. Create webhook for your channel
5. Copy the webhook URL to GitHub Secrets

---

### 3. Security Scanning

| Secret Name        | Description                 | Required For                      |
| ------------------ | --------------------------- | --------------------------------- |
| `SNYK_TOKEN`       | Snyk API token              | Dependency vulnerability scanning |
| `GITLEAKS_LICENSE` | Gitleaks Enterprise license | Secret detection (optional)       |

**How to get Snyk token:**

1. Sign up at [snyk.io](https://snyk.io)
2. Go to Account Settings → API Token
3. Copy token to GitHub Secrets

---

## Workflow-Specific Requirements

### ci-cd.yml

```yaml
Required Secrets:
  - GCP_PROJECT_ID # Line 16
  - GCR_JSON_KEY # Lines 143, 239
  - GCP_SA_KEY # Lines 309, 339
  - SLACK_WEBHOOK # Line 385
```

### docker-build.yml

```yaml
Required Secrets:
  - GCP_PROJECT_ID # Line 47
  - SNYK_TOKEN # Line 145
  - GCR_JSON_KEY # Line 198
  - GKE_SA_KEY # Lines 400, 447
  - SLACK_CHANNEL_ID # Line 487
  - SLACK_BOT_TOKEN # Line 494
```

### security-scanning.yml

```yaml
Required Secrets:
  - GITLEAKS_LICENSE # Line 43 (optional)
  - SLACK_SECURITY_WEBHOOK # Line 312
```

### three-version-pipeline.yml

```yaml
Required Secrets:
  - GCP_PROJECT_ID # Lines 22, 252, 467, 608
  - GCP_SA_KEY # Lines 51, 253, 468, 609
  - SLACK_WEBHOOK_URL # Lines 593, 648
```

---

## Quick Setup Script

Create a shell script to set up secrets using GitHub CLI:

```bash
#!/bin/bash
# setup-secrets.sh

# Check if gh is installed
if ! command -v gh &> /dev/null; then
    echo "GitHub CLI (gh) is required. Install from: https://cli.github.com/"
    exit 1
fi

# Authenticate
gh auth login

# Set repository (change to your repo)
REPO="your-org/AI-Code-Review-and-Architecture-Analysis_Platform"

# GCP Secrets (replace with your values)
gh secret set GCP_PROJECT_ID --repo $REPO --body "your-project-id"
gh secret set GCP_SA_KEY --repo $REPO < path/to/service-account-key.json
gh secret set GCR_JSON_KEY --repo $REPO < path/to/gcr-key.json
gh secret set GKE_SA_KEY --repo $REPO < path/to/gke-key.json

# Slack Secrets
gh secret set SLACK_WEBHOOK --repo $REPO --body "https://hooks.slack.com/..."
gh secret set SLACK_WEBHOOK_URL --repo $REPO --body "https://hooks.slack.com/..."
gh secret set SLACK_BOT_TOKEN --repo $REPO --body "xoxb-..."
gh secret set SLACK_CHANNEL_ID --repo $REPO --body "C0123456789"
gh secret set SLACK_SECURITY_WEBHOOK --repo $REPO --body "https://hooks.slack.com/..."

# Security Scanning
gh secret set SNYK_TOKEN --repo $REPO --body "your-snyk-token"
# GITLEAKS_LICENSE is optional for enterprise features

echo "All secrets configured!"
```

---

## Environment Variables (Non-Secret)

Some workflows also use environment variables that can be set in **Settings → Secrets and variables → Actions → Variables**:

| Variable       | Default           | Description        |
| -------------- | ----------------- | ------------------ |
| `REGISTRY`     | `gcr.io`          | Container registry |
| `REGION`       | `us-central1`     | GCP region         |
| `CLUSTER_NAME` | `coderev-cluster` | GKE cluster name   |
| `CLUSTER_ZONE` | `us-central1-a`   | GKE cluster zone   |

---

## Local Development (No Secrets Required)

For local development without GCP/Slack integration:

```bash
# Use Docker Compose with mock mode
MOCK_MODE=true docker-compose -f docker-compose.dev.yml up -d

# Or run directly
cd backend
MOCK_MODE=true python -m uvicorn dev_api.app:app --reload
```

---

## Security Best Practices

1. **Never commit secrets** to the repository
2. **Rotate secrets** regularly (every 90 days recommended)
3. **Use least privilege** - only grant necessary permissions
4. **Audit access** - review who has access to secrets
5. **Use environment protection** - require approvals for production

---

## Troubleshooting

### "Context access might be invalid" Warning

This IDE warning appears because the linter cannot verify if secrets exist. To resolve:

1. **Configure the secret** in GitHub repository settings
2. **Or ignore** if the workflow job is optional (e.g., Slack notifications)

### "Secret not found" Error in Workflow

1. Check secret name matches exactly (case-sensitive)
2. Verify secret is set at repository level (not just environment)
3. Check if secret has expired (for service account keys)

### Permission Denied

1. Verify service account has required IAM roles
2. Check if Workload Identity is configured correctly
3. Ensure secret value is properly formatted (especially JSON keys)

---

## Reference Links

- [GitHub Encrypted Secrets](https://docs.github.com/en/actions/security-guides/encrypted-secrets)
- [GCP Service Account Keys](https://cloud.google.com/iam/docs/creating-managing-service-account-keys)
- [Slack Webhooks](https://api.slack.com/messaging/webhooks)
- [Snyk API Token](https://docs.snyk.io/snyk-api-info/authentication-for-api)
