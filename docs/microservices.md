# Microservices Layer Documentation

## Overview

Comprehensive microservices architecture with four specialized services:

1. **Auth Service** - Authentication and authorization
2. **Project Service** - Project and version management
3. **Repo Service** - GitHub/GitLab integration
4. **Analysis Service** - Code analysis orchestration

---

## Auth Service

### Responsibilities

#### 1. User Registration

- Invitation code validation
- Admin code: `ZBXzbx123` (hashed in secrets manager)
- Optional user code for open registration
- Rate limiting on registration

#### 2. JWT Authentication

- Access tokens (15 minutes)
- Refresh tokens (7 days)
- Token refresh endpoint
- Token revocation

#### 3. Email Verification

- AWS SES/Postmark integration
- Verification token (24-hour expiry)
- Resend verification email
- Email confirmation required for login

#### 4. Password Reset

- Time-limited reset tokens (1 hour)
- Email-based reset flow
- Password strength validation
- Reset history tracking

#### 5. Role-Based Access Control (RBAC)

- Admin role (full access)
- User role (standard access)
- Viewer role (read-only access)
- Role assignment via invitation codes

#### 6. 2FA Support

- TOTP (Time-based One-Time Password)
- QR code generation
- Backup codes
- WebAuthn ready (future)

### Database Schema

```sql
-- Users table
users:
  id (UUID, PK)
  email (String, unique)
  password_hash (String, Argon2id)
  role (Enum: admin, user, viewer)
  verified (Boolean)
  totp_secret (String, nullable)
  created_at (DateTime)
  updated_at (DateTime)
  last_login (DateTime, nullable)
  failed_login_attempts (Integer)
  locked_until (DateTime, nullable)

-- Sessions table
sessions:
  id (UUID, PK)
  user_id (UUID, FK)
  refresh_token_hash (String, unique)
  expires_at (DateTime)
  device_info (JSON)
  created_at (DateTime)
  last_used (DateTime)

-- Invitations table
invitations:
  id (UUID, PK)
  code_hash (String, unique)
  role (Enum: admin, user, viewer)
  max_uses (Integer)
  uses (Integer)
  expires_at (DateTime)
  created_at (DateTime)

-- Audit logs table
audit_logs:
  id (UUID, PK)
  user_id (UUID, nullable)
  action (String)
  resource (String)
  status (String: success, failure)
  details (JSON)
  ip_address (String)
  user_agent (String)
  created_at (DateTime)

-- Password resets table
password_resets:
  id (UUID, PK)
  user_id (UUID, FK)
  token_hash (String, unique)
  expires_at (DateTime)
  used (Boolean)
  created_at (DateTime)
```

### Security Features

#### Password Hashing

- **Algorithm**: Argon2id (not bcrypt)
- **Memory**: 64MB
- **Time Cost**: 3 iterations
- **Parallelism**: 4 threads

#### Rate Limiting

- **Login**: 5 attempts per 15 minutes per IP
- **Registration**: 10 per hour per IP
- **Password Reset**: 3 per hour per email
- **Email Verification**: 5 per hour per email

#### Account Lockout

- **Threshold**: 10 failed login attempts
- **Duration**: 30 minutes
- **Manual unlock**: Admin only

#### Audit Logging

- All authentication events logged
- IP address tracking
- User agent logging
- Failure reason recording

### API Endpoints

```
POST /api/v1/auth/register
  - Email, password, invitation code
  - Returns: user_id, email

POST /api/v1/auth/login
  - Email, password
  - Returns: access_token, refresh_token

POST /api/v1/auth/refresh
  - Refresh token
  - Returns: new access_token

POST /api/v1/auth/logout
  - Revoke refresh token

POST /api/v1/auth/verify-email
  - Verification token
  - Returns: success

POST /api/v1/auth/resend-verification
  - Email
  - Returns: success

POST /api/v1/auth/forgot-password
  - Email
  - Returns: success

POST /api/v1/auth/reset-password
  - Reset token, new password
  - Returns: success

POST /api/v1/auth/2fa/setup
  - Returns: secret, qr_code

POST /api/v1/auth/2fa/verify
  - TOTP token
  - Returns: backup_codes

GET /api/v1/users/me
  - Returns: user profile

PUT /api/v1/users/me
  - Update user profile

POST /api/v1/users/change-password
  - Old password, new password
```

---

## Project Service

### Responsibilities

#### 1. Project Management

- Create, read, update, delete projects
- Project ownership and permissions
- Project archival
- Settings management

#### 2. Version Lifecycle

- Create versions (v1, v2, v3)
- Version promotion (v1 → v2)
- Version degradation (v2 → v3)
- Version comparison

#### 3. Baseline Configuration

- Define metric thresholds
- Operator support (>, <, >=, <=, ==, !=)
- Snapshot references
- Threshold history

#### 4. Policy Management

- OPA policy storage
- Policy activation/deactivation
- Policy-project associations
- Policy versioning

#### 5. Change History

- Track all version changes
- Record promotion/degradation reasons
- Audit trail with user attribution
- Rollback capability

### Database Schema

```sql
-- Projects table
projects:
  id (UUID, PK)
  name (String)
  description (Text, nullable)
  owner_id (UUID, FK)
  created_at (DateTime)
  updated_at (DateTime)
  settings (JSON)
  archived (Boolean)

-- Versions table
versions:
  id (UUID, PK)
  project_id (UUID, FK)
  tag (Enum: v1, v2, v3)
  model_config (JSON)
  changelog (Text, nullable)
  promoted_at (DateTime, nullable)
  created_at (DateTime)
  updated_at (DateTime)

-- Baselines table
baselines:
  id (UUID, PK)
  project_id (UUID, FK)
  metric_key (String)
  threshold (String)
  operator (String)
  snapshot_id (String, nullable)
  created_at (DateTime)
  updated_at (DateTime)

-- Policies table
policies:
  id (UUID, PK)
  name (String, unique)
  description (Text, nullable)
  rego_code (Text)
  active (Boolean)
  created_at (DateTime)
  updated_at (DateTime)

-- Project-Policy associations
project_policies:
  id (UUID, PK)
  project_id (UUID, FK)
  policy_id (UUID, FK)
  created_at (DateTime)

-- Version history
version_history:
  id (UUID, PK)
  version_id (UUID, FK)
  changed_by (UUID, FK)
  action (String: promoted, degraded, updated)
  from_tag (Enum, nullable)
  to_tag (Enum, nullable)
  reason (Text, nullable)
  created_at (DateTime)
```

### API Endpoints

```
POST /api/v1/projects
  - Create project
  - Returns: project_id

GET /api/v1/projects/{id}
  - Get project details

PUT /api/v1/projects/{id}
  - Update project

DELETE /api/v1/projects/{id}
  - Archive project

GET /api/v1/projects/{id}/versions
  - List all versions

POST /api/v1/versions
  - Create new version

GET /api/v1/versions/{id}
  - Get version details

POST /api/v1/versions/{id}/promote
  - Promote v1→v2 (admin only)

POST /api/v1/versions/{id}/degrade
  - Degrade v2→v3 (admin only)

GET /api/v1/versions/compare
  - Compare two versions

GET /api/v1/baselines
  - List baselines

POST /api/v1/baselines
  - Create baseline

PUT /api/v1/baselines/{id}
  - Update baseline

GET /api/v1/policies
  - List policies

POST /api/v1/policies
  - Create policy

PUT /api/v1/policies/{id}
  - Update policy
```

---

## Repo Service

### Responsibilities

#### 1. GitHub Integration

- OAuth App installation flow
- Webhook event handling
- Check Runs API for PR status
- Suggestions API for code fixes
- Octokit.js for API interactions

#### 2. GitLab Integration

- Project access tokens
- System hooks for events
- Merge Request API
- Discussion threads API

#### 3. Repository Management

- Repository indexing
- Webhook registration
- Access token management
- Repository activation/deactivation

#### 4. Pull Request Handling

- PR event listening (opened, synchronized, closed)
- PR status updates
- Inline comments
- Branch protection enforcement

#### 5. File Tree Caching

- Cache file structure
- Commit SHA tracking
- Cache expiration
- Efficient retrieval

### Database Schema

```sql
-- Repositories table
repositories:
  id (UUID, PK)
  project_id (UUID, FK)
  provider (Enum: github, gitlab)
  repo_url (String, unique)
  repo_name (String)
  owner (String)
  webhook_secret (String)
  access_token (String, encrypted)
  webhook_id (String, nullable)
  active (Boolean)
  created_at (DateTime)
  updated_at (DateTime)

-- Pull requests table
pull_requests:
  id (UUID, PK)
  repo_id (UUID, FK)
  pr_number (Integer)
  pr_title (String)
  pr_url (String)
  status (Enum: pending, analyzing, completed, failed)
  analysis_id (UUID, nullable)
  author (String)
  branch (String)
  base_branch (String)
  created_at (DateTime)
  updated_at (DateTime)
  analyzed_at (DateTime, nullable)

-- PR comments table
pr_comments:
  id (UUID, PK)
  pr_id (UUID, FK)
  file_path (String)
  line_number (Integer)
  comment (Text)
  author (Enum: ai, user)
  severity (String: critical, high, medium, low, info)
  category (String)
  suggestion (Text, nullable)
  external_id (String, nullable)
  created_at (DateTime)
  updated_at (DateTime)

-- File cache table
file_cache:
  id (UUID, PK)
  repo_id (UUID, FK)
  branch (String)
  file_tree (JSON)
  commit_sha (String)
  created_at (DateTime)
  expires_at (DateTime)
```

### Webhook Integration

#### GitHub Webhook Events

- `pull_request` - PR opened, synchronized, closed
- `pull_request_review` - Review submitted
- `issue_comment` - Comment on PR
- `push` - Branch updated

#### GitLab Webhook Events

- `merge_request` - MR opened, updated, closed
- `note` - Comment on MR
- `push` - Branch updated

### API Endpoints

```
POST /api/v1/repositories
  - Register repository

GET /api/v1/repositories/{id}
  - Get repository details

PUT /api/v1/repositories/{id}
  - Update repository

DELETE /api/v1/repositories/{id}
  - Deactivate repository

GET /api/v1/pull-requests
  - List pull requests

GET /api/v1/pull-requests/{id}
  - Get PR details

POST /api/v1/pull-requests/{id}/comments
  - Add comment to PR

GET /api/v1/pull-requests/{id}/comments
  - List PR comments

POST /api/v1/webhooks/github
  - GitHub webhook endpoint

POST /api/v1/webhooks/gitlab
  - GitLab webhook endpoint
```

---

## Analysis Service

### Responsibilities

#### 1. Session Management

- Create analysis sessions
- Track session lifecycle
- Handle session failures
- Session result aggregation

#### 2. Static Analysis

- Semgrep (security rules)
- ESLint/Prettier (JavaScript/TypeScript)
- ruff/black (Python)
- Bandit (Python security)
- Tree-sitter (AST parsing)
- radon (cyclomatic complexity)

#### 3. Dynamic Analysis

- pytest with coverage.py
- cProfile (CPU profiling)
- line_profiler (line-level profiling)
- memory_profiler (memory analysis)

#### 4. Graph Analysis

- Neo4j Cypher queries
- Dependency analysis
- Call graph construction
- Coupling metrics

#### 5. Result Management

- Report generation
- S3 artifact storage
- SHA256 checksums
- Metadata tracking

#### 6. Cache Management

- Result caching
- Code hash-based lookup
- Cache expiration
- Repeated analysis optimization

### Database Schema

```sql
-- Analysis sessions table
analysis_sessions:
  id (UUID, PK)
  user_id (UUID, FK)
  project_id (UUID, FK)
  version (String: v1, v2, v3)
  status (Enum: created, queued, running, completed, failed, cancelled)
  started_at (DateTime)
  finished_at (DateTime, nullable)
  error_message (Text, nullable)
  metadata (JSON)

-- Analysis tasks table
analysis_tasks:
  id (UUID, PK)
  session_id (UUID, FK)
  type (Enum: static, dynamic, graph)
  status (Enum: pending, running, completed, failed, skipped)
  result (JSON, nullable)
  error_message (Text, nullable)
  started_at (DateTime, nullable)
  finished_at (DateTime, nullable)
  duration_seconds (Integer, nullable)
  created_at (DateTime)

-- Artifacts table
artifacts:
  id (UUID, PK)
  session_id (UUID, FK)
  type (Enum: report, patch, log, metrics)
  s3_uri (String)
  sha256 (String)
  size_bytes (Integer)
  metadata (JSON)
  created_at (DateTime)

-- Analysis cache table
analysis_cache:
  id (UUID, PK)
  project_id (UUID, FK)
  code_hash (String, unique)
  language (String)
  result (JSON)
  created_at (DateTime)
  expires_at (DateTime)
```

### Analysis Pipeline

```
┌─────────────────────────────────────────────────────────┐
│            Analysis Session Created                     │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Check Cache                                            │
│  ├─→ Hit → Return cached result                         │
│  └─→ Miss → Continue                                    │
│                                                         │
│  Parallel Analysis Tasks                                │
│  ├─→ Static Analysis                                    │
│  │   ├─→ Semgrep (security)                             │
│  │   ├─→ ESLint/ruff (style)                            │
│  │   ├─→ Tree-sitter (AST)                              │
│  │   └─→ radon (complexity)                             │
│  ├─→ Dynamic Analysis                                   │
│  │   ├─→ pytest (coverage)                              │
│  │   ├─→ cProfile (CPU)                                 │
│  │   └─→ memory_profiler (memory)                       │
│  └─→ Graph Analysis                                     │
│      ├─→ Dependencies                                   │
│      ├─→ Call graphs                                    │
│      └─→ Coupling metrics                               │
│                                                         │
│  Aggregate Results                                      │
│  ├─→ Combine findings                                   │
│  ├─→ Generate report                                    │
│  └─→ Store artifacts                                    │
│                                                         │
│  Cache Results                                          │
│  └─→ Store for future use                               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### API Endpoints

```
POST /api/v1/analysis/sessions
  - Create analysis session
  - Returns: session_id

GET /api/v1/analysis/sessions/{id}
  - Get session details

GET /api/v1/analysis/sessions/{id}/tasks
  - List session tasks

GET /api/v1/analysis/sessions/{id}/artifacts
  - List session artifacts

POST /api/v1/analysis/sessions/{id}/cancel
  - Cancel session

GET /api/v1/analysis/cache
  - Check cache for code hash

POST /api/v1/analysis/cache
  - Store analysis result in cache
```

---

## Service Communication

### Event-Driven Architecture

```
Auth Service
├─→ user.registered
├─→ user.verified
├─→ user.login
└─→ user.password_reset

Project Service
├─→ project.created
├─→ version.promoted
├─→ version.degraded
└─→ policy.updated

Repo Service
├─→ pr.opened
├─→ pr.synchronized
├─→ pr.closed
└─→ comment.added

Analysis Service
├─→ analysis.started
├─→ analysis.completed
├─→ analysis.failed
└─→ cache.hit
```

### Inter-Service Communication

```
Repo Service → Analysis Service
  - Trigger analysis on PR events

Analysis Service → Repo Service
  - Post comments on PR

Project Service → Auth Service
  - Verify user permissions

Repo Service → Project Service
  - Get project configuration
```

---

## Deployment

### Docker Compose

```yaml
services:
  auth-service:
    image: platform-auth-service:latest
    ports:
      - "8001:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - SECRET_KEY=...
    depends_on:
      - postgres

  project-service:
    image: platform-project-service:latest
    ports:
      - "8002:8000"
    environment:
      - DATABASE_URL=postgresql://...
    depends_on:
      - postgres

  repo-service:
    image: platform-repo-service:latest
    ports:
      - "8003:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - GITHUB_APP_ID=...
      - GITLAB_TOKEN=...
    depends_on:
      - postgres

  analysis-service:
    image: platform-analysis-service:latest
    ports:
      - "8004:8000"
    environment:
      - DATABASE_URL=postgresql://...
      - S3_BUCKET=...
    depends_on:
      - postgres
```

### Kubernetes

Each service has its own deployment with:

- Resource limits and requests
- Health checks
- Horizontal Pod Autoscaling
- Service mesh integration ready

---

## Monitoring

### Prometheus Metrics

- `auth_requests_total` - Total authentication requests
- `failed_logins_total` - Failed login attempts
- `analysis_duration_seconds` - Analysis execution time
- `cache_hit_ratio` - Cache hit percentage

### Grafana Dashboards

- Authentication metrics
- Project lifecycle
- Repository integration
- Analysis performance

---

## Security Considerations

1. **Secrets Management**: Use AWS Secrets Manager or HashiCorp Vault
2. **Database Encryption**: Encrypt sensitive fields at rest
3. **API Authentication**: JWT tokens with short expiry
4. **Rate Limiting**: Per-user and per-IP limits
5. **Audit Logging**: All operations logged
6. **CORS**: Strict origin validation
7. **Input Validation**: All inputs validated
8. **SQL Injection**: Parameterized queries

---

## Future Enhancements

- [ ] GraphQL API support
- [ ] gRPC for inter-service communication
- [ ] Service mesh (Istio) integration
- [ ] Advanced caching (Redis)
- [ ] Message queue (RabbitMQ/Kafka)
- [ ] Distributed tracing (Jaeger)
- [ ] Advanced security (OAuth2, SAML)
- [ ] Multi-tenancy support
