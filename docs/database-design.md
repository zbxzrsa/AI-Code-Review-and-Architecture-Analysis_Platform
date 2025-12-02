# Database Design Documentation

## Overview

Comprehensive PostgreSQL database design with 7 schemas supporting the three-version architecture, authentication, project management, experimentation, production analysis, quarantine, and provider management.

---

## Schema Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  PostgreSQL Database                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  auth                                                       │
│  ├─→ users (email, password_hash, role, verified)          │
│  ├─→ sessions (refresh_token_hash, expires_at)             │
│  ├─→ invitations (code_hash, role, max_uses)               │
│  ├─→ audit_logs (action, resource, status)                 │
│  └─→ password_resets (token_hash, expires_at)              │
│                                                             │
│  projects                                                   │
│  ├─→ projects (name, owner_id, settings)                   │
│  ├─→ versions (tag: v1/v2/v3, model_config)                │
│  ├─→ baselines (metric_key, threshold, operator)           │
│  ├─→ policies (name, rego_code, active)                    │
│  ├─→ project_policies (project_id, policy_id)              │
│  └─→ version_history (action, from_tag, to_tag)            │
│                                                             │
│  experiments_v1                                             │
│  ├─→ experiments (config, dataset_id, status)              │
│  ├─→ evaluations (metrics, ai_verdict, human_override)     │
│  ├─→ promotions (from_version_id, to_version_id, status)   │
│  ├─→ blacklist (config_hash, reason, evidence)             │
│  └─→ comparison_reports (v1_exp_id, v2_ver_id, metrics)    │
│                                                             │
│  production                                                 │
│  ├─→ analysis_sessions (user_id, project_id, status)       │
│  ├─→ analysis_tasks (type, status, result, duration_ms)    │
│  ├─→ artifacts (type, s3_uri, sha256, size_bytes)          │
│  └─→ code_review_results (language, score, model_used)     │
│                                                             │
│  quarantine                                                 │
│  ├─→ quarantine_records (config_hash, reason, evidence)    │
│  └─→ reevaluation_requests (status, approved_by)           │
│                                                             │
│  providers                                                  │
│  ├─→ providers (name, model_name, cost_per_1k_tokens)      │
│  ├─→ user_providers (encrypted_api_key, encrypted_dek)     │
│  ├─→ provider_health (is_healthy, response_time_ms)        │
│  ├─→ user_quotas (daily_limit, monthly_limit)              │
│  ├─→ usage_tracking (requests_count, tokens_used, cost)    │
│  └─→ cost_alerts (alert_type, threshold_percentage)        │
│                                                             │
│  audits                                                     │
│  ├─→ audit_log (entity, action, payload, signature)        │
│  │   ├─ Partitioned by month                               │
│  │   └─ Immutable with chain validation                     │
│  └─→ statistics (view)                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Schema Details

### 1. Auth Schema

**Purpose**: User authentication, sessions, and authorization

**Tables**:

#### users

- `id` (UUID, PK) - User identifier
- `email` (VARCHAR, unique) - Email address with format validation
- `password_hash` (VARCHAR) - Argon2id hash
- `role` (VARCHAR) - admin, user, viewer
- `verified` (BOOLEAN) - Email verification status
- `totp_secret` (VARCHAR) - 2FA secret
- `created_at`, `updated_at`, `last_login` (TIMESTAMPTZ)
- `failed_login_attempts` (INTEGER) - For lockout
- `locked_until` (TIMESTAMPTZ) - Account lockout expiry
- `settings` (JSONB) - User preferences

**Indexes**:

- email (unique)
- role
- verified

#### sessions

- `id` (UUID, PK)
- `user_id` (UUID, FK) - Reference to users
- `refresh_token_hash` (VARCHAR, unique) - Hashed refresh token
- `expires_at` (TIMESTAMPTZ) - Session expiry
- `device_info` (JSONB) - Device fingerprint
- `created_at`, `last_used` (TIMESTAMPTZ)

**Indexes**:

- user_id
- expires_at
- refresh_token_hash

#### invitations

- `id` (UUID, PK)
- `code_hash` (VARCHAR, unique) - Hashed invitation code
- `role` (VARCHAR) - Role to assign
- `max_uses` (INTEGER) - Usage limit
- `uses` (INTEGER) - Current usage count
- `expires_at` (TIMESTAMPTZ) - Expiry

**Indexes**:

- code_hash
- expires_at

#### audit_logs

- `id` (UUID, PK)
- `user_id` (UUID, FK) - Who performed action
- `action` (VARCHAR) - Action type
- `resource` (VARCHAR) - Resource affected
- `status` (VARCHAR) - success, failure
- `details` (JSONB) - Additional details
- `ip_address`, `user_agent` (VARCHAR)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- user_id
- action
- created_at

#### password_resets

- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `token_hash` (VARCHAR, unique) - Hashed reset token
- `expires_at` (TIMESTAMPTZ)
- `used` (BOOLEAN)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- user_id
- token_hash
- expires_at

---

### 2. Projects Schema

**Purpose**: Project management, versions, baselines, and policies

**Tables**:

#### projects

- `id` (UUID, PK)
- `name` (VARCHAR) - Project name
- `description` (TEXT)
- `owner_id` (UUID, FK) - Project owner
- `settings` (JSONB) - Project configuration
- `archived` (BOOLEAN)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- owner_id
- archived

#### versions

- `id` (UUID, PK)
- `project_id` (UUID, FK)
- `tag` (VARCHAR) - v1, v2, v3
- `model_config` (JSONB) - AI model configuration
- `changelog` (TEXT)
- `promoted_at` (TIMESTAMPTZ)
- `promoted_by` (UUID, FK)
- `created_at`, `updated_at` (TIMESTAMPTZ)
- Unique constraint: (project_id, tag)

**Indexes**:

- project_id
- tag
- promoted_at

#### baselines

- `id` (UUID, PK)
- `project_id` (UUID, FK)
- `metric_key` (VARCHAR) - Metric name
- `threshold` (DECIMAL) - Threshold value
- `operator` (VARCHAR) - >, <, >=, <=, =
- `snapshot_id` (UUID) - Reference snapshot
- `active` (BOOLEAN)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- project_id
- active

#### policies

- `id` (UUID, PK)
- `name` (VARCHAR, unique) - Policy name
- `description` (TEXT)
- `rego_code` (TEXT) - OPA Rego code
- `active` (BOOLEAN)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- active

#### project_policies

- `id` (UUID, PK)
- `project_id` (UUID, FK)
- `policy_id` (UUID, FK)
- `created_at` (TIMESTAMPTZ)
- Unique constraint: (project_id, policy_id)

**Indexes**:

- project_id
- policy_id

#### version_history

- `id` (UUID, PK)
- `version_id` (UUID, FK)
- `changed_by` (UUID, FK) - User who made change
- `action` (VARCHAR) - promoted, degraded, updated
- `from_tag`, `to_tag` (VARCHAR)
- `reason` (TEXT)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- version_id
- changed_by
- action

---

### 3. Experiments V1 Schema

**Purpose**: V1 experimentation zone data

**Tables**:

#### experiments

- `id` (UUID, PK)
- `name` (VARCHAR)
- `description` (TEXT)
- `version` (VARCHAR) - Always 'v1'
- `config` (JSONB) - Experiment configuration
- `dataset_id` (UUID) - Dataset reference
- `status` (VARCHAR) - created, running, completed, failed, evaluating, promoted, quarantined
- `created_by` (UUID, FK)
- `created_at`, `updated_at`, `started_at`, `completed_at` (TIMESTAMPTZ)

**Indexes**:

- status
- created_by
- dataset_id
- created_at

#### evaluations

- `id` (UUID, PK)
- `experiment_id` (UUID, FK)
- `metrics` (JSONB) - Evaluation metrics
- `ai_verdict` (VARCHAR) - pass, fail, manual_review
- `ai_confidence` (DECIMAL)
- `human_override` (VARCHAR)
- `override_reason` (TEXT)
- `evaluated_by` (VARCHAR) - ai, human
- `evaluator_id` (UUID, FK)
- `evaluated_at`, `created_at` (TIMESTAMPTZ)

**Indexes**:

- experiment_id
- ai_verdict
- evaluated_at

#### promotions

- `id` (UUID, PK)
- `from_version_id` (UUID)
- `to_version_id` (UUID)
- `status` (VARCHAR) - pending, approved, rejected, completed
- `reason` (TEXT)
- `approver_id` (UUID, FK)
- `promoted_at` (TIMESTAMPTZ)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- status
- from_version_id
- to_version_id

#### blacklist

- `id` (UUID, PK)
- `config_hash` (VARCHAR, unique) - SHA256 of config
- `reason` (TEXT)
- `evidence` (JSONB)
- `quarantined_at` (TIMESTAMPTZ)
- `quarantined_by` (UUID, FK)
- `review_status` (VARCHAR) - pending, reviewed, appealed
- `reviewed_by` (UUID, FK)
- `reviewed_at` (TIMESTAMPTZ)

**Indexes**:

- config_hash
- review_status

#### comparison_reports

- `id` (UUID, PK)
- `v1_experiment_id` (UUID, FK)
- `v2_version_id` (UUID)
- `dataset_id` (UUID)
- `metrics` (JSONB)
- `recommendation` (VARCHAR)
- `confidence` (DECIMAL)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- v1_experiment_id
- v2_version_id

---

### 4. Production Schema

**Purpose**: V2 production zone analysis data

**Tables**:

#### analysis_sessions

- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `project_id` (UUID, FK)
- `version` (VARCHAR) - v1, v2, v3
- `status` (VARCHAR) - created, queued, running, completed, failed, cancelled
- `started_at`, `finished_at` (TIMESTAMPTZ)
- `error_message` (TEXT)
- `metadata` (JSONB)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- user_id
- project_id
- status
- started_at

#### analysis_tasks

- `id` (UUID, PK)
- `session_id` (UUID, FK)
- `type` (VARCHAR) - static, dynamic, graph, ai
- `status` (VARCHAR) - pending, running, completed, failed, skipped
- `result` (JSONB)
- `error_message` (TEXT)
- `started_at`, `finished_at` (TIMESTAMPTZ)
- `duration_ms` (INTEGER)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- session_id
- type
- status

#### artifacts

- `id` (UUID, PK)
- `session_id` (UUID, FK)
- `type` (VARCHAR) - report, patch, log, metrics
- `s3_uri` (TEXT) - S3 location
- `sha256` (VARCHAR) - Checksum
- `size_bytes` (BIGINT)
- `metadata` (JSONB)
- `created_at` (TIMESTAMPTZ)
- `expires_at` (TIMESTAMPTZ) - For temporary artifacts

**Indexes**:

- session_id
- type
- expires_at

#### code_review_results

- `id` (UUID, PK)
- `session_id` (UUID, FK)
- `code_language` (VARCHAR)
- `code_length` (INTEGER)
- `overall_score` (DECIMAL)
- `vulnerabilities_count`, `issues_count` (INTEGER)
- `model_used` (VARCHAR)
- `analysis_time_ms` (INTEGER)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- session_id
- code_language

---

### 5. Quarantine Schema

**Purpose**: V3 quarantine zone (read-only archive)

**Tables**:

#### quarantine_records

- `id` (UUID, PK)
- `experiment_id` (UUID)
- `config_hash` (VARCHAR, unique)
- `config` (JSONB)
- `reason` (TEXT)
- `evidence` (JSONB)
- `failure_metrics` (JSONB)
- `quarantined_at` (TIMESTAMPTZ)
- `quarantined_by` (UUID, FK)
- `review_status` (VARCHAR) - quarantined, under_review, approved_retry
- `reviewed_at` (TIMESTAMPTZ)
- `reviewed_by` (UUID, FK)
- `review_notes` (TEXT)
- `created_at` (TIMESTAMPTZ)

**Indexes**:

- config_hash
- review_status
- quarantined_at

#### reevaluation_requests

- `id` (UUID, PK)
- `quarantine_id` (UUID, FK)
- `requested_by` (UUID, FK)
- `reason` (TEXT)
- `status` (VARCHAR) - pending, approved, rejected, completed
- `approved_by` (UUID, FK)
- `approved_at` (TIMESTAMPTZ)
- `completed_at` (TIMESTAMPTZ)
- `result` (JSONB)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- quarantine_id
- status
- requested_by

---

### 6. Providers Schema

**Purpose**: AI provider management and usage tracking

**Tables**:

#### providers

- `id` (UUID, PK)
- `name` (VARCHAR, unique)
- `provider_type` (VARCHAR) - openai, anthropic, huggingface, local
- `model_name` (VARCHAR)
- `api_endpoint` (VARCHAR)
- `is_active` (BOOLEAN)
- `is_platform_provided` (BOOLEAN)
- `cost_per_1k_tokens` (DECIMAL)
- `max_tokens` (INTEGER)
- `timeout_seconds` (INTEGER)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- is_active
- provider_type

#### user_providers

- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `provider_name` (VARCHAR)
- `provider_type` (VARCHAR)
- `model_name` (VARCHAR)
- `encrypted_api_key` (TEXT) - AES-256-GCM encrypted
- `encrypted_dek` (TEXT) - KMS encrypted DEK
- `key_last_4_chars` (VARCHAR)
- `is_active` (BOOLEAN)
- `created_at`, `updated_at` (TIMESTAMPTZ)
- Unique constraint: (user_id, provider_type)

**Indexes**:

- user_id
- is_active

#### provider_health

- `id` (UUID, PK)
- `provider_id` (UUID, FK)
- `is_healthy` (BOOLEAN)
- `last_check_at` (TIMESTAMPTZ)
- `last_error` (TEXT)
- `consecutive_failures` (INTEGER)
- `response_time_ms` (DECIMAL)
- `success_rate` (DECIMAL)
- `updated_at` (TIMESTAMPTZ)

**Indexes**:

- provider_id
- is_healthy

#### user_quotas

- `id` (UUID, PK)
- `user_id` (UUID, unique FK)
- `daily_limit` (INTEGER)
- `monthly_limit` (INTEGER)
- `daily_cost_limit` (DECIMAL)
- `monthly_cost_limit` (DECIMAL)
- `created_at`, `updated_at` (TIMESTAMPTZ)

**Indexes**:

- user_id

#### usage_tracking

- `id` (BIGSERIAL, PK)
- `user_id` (UUID, FK)
- `provider` (VARCHAR)
- `model` (VARCHAR)
- `date` (DATE)
- `requests_count` (INTEGER)
- `tokens_input`, `tokens_output` (INTEGER)
- `cost_usd` (DECIMAL)
- `created_at` (TIMESTAMPTZ)
- Unique constraint: (user_id, provider, model, date)

**Indexes**:

- user_id, date
- date
- provider

#### cost_alerts

- `id` (UUID, PK)
- `user_id` (UUID, FK)
- `alert_type` (VARCHAR) - daily_80, daily_90, daily_100, monthly_80, monthly_90, monthly_100
- `threshold_percentage` (INTEGER)
- `triggered_at` (TIMESTAMPTZ)
- `acknowledged` (BOOLEAN)
- `acknowledged_at` (TIMESTAMPTZ)

**Indexes**:

- user_id
- triggered_at
- acknowledged

---

### 7. Audits Schema

**Purpose**: Immutable audit trail with cryptographic chaining

**Tables**:

#### audit_log (Partitioned by month)

- `id` (BIGSERIAL)
- `entity` (VARCHAR) - Entity type
- `action` (VARCHAR) - Action performed
- `actor_id` (UUID, FK) - Who performed action
- `payload` (JSONB) - Action details
- `signature` (TEXT) - SHA256 signature
- `prev_hash` (TEXT) - Hash of previous entry (chain)
- `ts` (TIMESTAMPTZ) - Timestamp
- Primary key: (id, ts)

**Partitions**:

- Monthly partitions (2024-12 through 2025-12)

**Indexes**:

- entity
- actor_id
- ts
- action

**Features**:

- Immutable (triggers prevent UPDATE/DELETE)
- Chain validation (prev_hash verification)
- Partitioned for performance

---

## Key Features

### Security

- ✅ Argon2id password hashing
- ✅ AES-256-GCM API key encryption
- ✅ KMS master key management
- ✅ Audit trail with cryptographic chaining
- ✅ Immutable audit logs
- ✅ Email format validation

### Performance

- ✅ Strategic indexing on frequently queried columns
- ✅ Partitioned audit logs by month
- ✅ JSONB for flexible schema
- ✅ Foreign key constraints for referential integrity

### Data Integrity

- ✅ Unique constraints where appropriate
- ✅ Check constraints for enum values
- ✅ Cryptographic chain validation
- ✅ Immutable audit logs

### Scalability

- ✅ Partitioned audit logs
- ✅ Efficient indexes
- ✅ BIGSERIAL for high-volume tables

---

## Initialization

### Prerequisites

- PostgreSQL 13+
- psql command-line tool
- Bash shell

### Setup

```bash
# Make script executable
chmod +x database/init-db.sh

# Run initialization
./database/init-db.sh

# Or with custom settings
DB_HOST=prod-db.example.com \
DB_NAME=ai_platform \
DB_USER=admin \
./database/init-db.sh
```

### Environment Variables

- `DB_HOST` - PostgreSQL host (default: localhost)
- `DB_PORT` - PostgreSQL port (default: 5432)
- `DB_NAME` - Database name (default: ai_code_review)
- `DB_USER` - Database user (default: postgres)
- `DB_PASSWORD` - Database password (default: postgres)

---

## Maintenance

### Backup

```bash
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > backup.sql
```

### Restore

```bash
psql -h $DB_HOST -U $DB_USER $DB_NAME < backup.sql
```

### Partition Management

```sql
-- Create new partition for next month
CREATE TABLE audits.audit_log_2026_01 PARTITION OF audits.audit_log
    FOR VALUES FROM ('2026-01-01') TO ('2026-02-01');
```

---

## Monitoring

### Check Schema Size

```sql
SELECT schemaname, pg_size_pretty(sum(pg_total_relation_size(schemaname||'.'||tablename))::bigint) as size
FROM pg_tables
GROUP BY schemaname
ORDER BY sum(pg_total_relation_size(schemaname||'.'||tablename)) DESC;
```

### Check Index Usage

```sql
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

### Check Audit Log Size

```sql
SELECT pg_size_pretty(pg_total_relation_size('audits.audit_log'));
```

---

## Future Enhancements

- [ ] Materialized views for reporting
- [ ] Time-series extension (TimescaleDB) for usage_tracking
- [ ] Full-text search on audit logs
- [ ] Row-level security (RLS)
- [ ] Automated partition creation
- [ ] Compression for old audit partitions
