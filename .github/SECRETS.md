# Required GitHub Secrets

This document lists all secrets required for the CI/CD pipelines to function correctly.

## Configuration

Configure these secrets in your GitHub repository:
**Settings → Secrets and variables → Actions → New repository secret**

---

## Required Secrets

### Google Cloud Platform

| Secret Name      | Description                                 | Required By                                             |
| ---------------- | ------------------------------------------- | ------------------------------------------------------- |
| `GCP_PROJECT_ID` | GCP Project ID                              | ci-cd.yml, docker-build.yml, three-version-pipeline.yml |
| `GCP_SA_KEY`     | GCP Service Account JSON key for deployment | three-version-pipeline.yml                              |
| `GCR_JSON_KEY`   | Google Container Registry JSON key          | ci-cd.yml, docker-build.yml                             |
| `GKE_SA_KEY`     | GKE Service Account key for cluster access  | docker-build.yml                                        |

### Security Scanning

| Secret Name        | Description                                    | Required By           |
| ------------------ | ---------------------------------------------- | --------------------- |
| `SNYK_TOKEN`       | Snyk API token for vulnerability scanning      | docker-build.yml      |
| `GITLEAKS_LICENSE` | Gitleaks license key (optional for enterprise) | security-scanning.yml |

### Notifications

| Secret Name              | Description                                  | Required By                |
| ------------------------ | -------------------------------------------- | -------------------------- |
| `SLACK_WEBHOOK`          | Slack webhook URL for general notifications  | ci-cd.yml                  |
| `SLACK_WEBHOOK_URL`      | Slack webhook URL for pipeline notifications | three-version-pipeline.yml |
| `SLACK_BOT_TOKEN`        | Slack bot token for rich notifications       | docker-build.yml           |
| `SLACK_CHANNEL_ID`       | Slack channel ID for notifications           | docker-build.yml           |
| `SLACK_SECURITY_WEBHOOK` | Slack webhook for security alerts            | security-scanning.yml      |

---

## Secret Setup Guide

### 1. GCP Service Account

```bash
# Create service account
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions"

# Grant necessary roles
gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/container.developer"

gcloud projects add-iam-policy-binding PROJECT_ID \
  --member="serviceAccount:github-actions@PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Generate key
gcloud iam service-accounts keys create key.json \
  --iam-account=github-actions@PROJECT_ID.iam.gserviceaccount.com

# Use the content of key.json for GCP_SA_KEY, GCR_JSON_KEY, GKE_SA_KEY
```

### 2. Slack Webhook

1. Go to https://api.slack.com/apps
2. Create a new app or select existing
3. Enable Incoming Webhooks
4. Create a webhook for your channel
5. Copy the webhook URL

### 3. Snyk Token

1. Log in to https://snyk.io
2. Go to Settings → API Token
3. Copy the token

---

## Environment Variables

These are configured in the workflow files (not secrets):

| Variable       | Value             | Description        |
| -------------- | ----------------- | ------------------ |
| `REGISTRY`     | `gcr.io`          | Container registry |
| `REGION`       | `us-central1`     | GCP region         |
| `CLUSTER_NAME` | `coderev-cluster` | GKE cluster name   |
| `CLUSTER_ZONE` | `us-central1-a`   | GKE cluster zone   |

---

## Troubleshooting

### "Context access might be invalid" warnings

These IDE warnings appear because the IDE cannot validate GitHub secrets. They are **not errors** - the secrets are accessed correctly at runtime if configured properly.

To resolve:

1. Ensure all secrets are configured in repository settings
2. Verify secret names match exactly (case-sensitive)
3. Check that secrets have valid values

### Pipeline fails with authentication errors

1. Verify the service account has necessary permissions
2. Check that the JSON key is properly formatted
3. Ensure the key hasn't expired

### Slack notifications not working

1. Verify webhook URL is correct
2. Check Slack app is installed in the workspace
3. Ensure the channel exists and webhook is authorized
