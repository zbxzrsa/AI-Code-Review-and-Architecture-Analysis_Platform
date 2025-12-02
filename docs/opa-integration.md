# OPA (Open Policy Agent) Integration Guide

## Overview

Comprehensive integration of Open Policy Agent for policy-based access control, quota management, and compliance enforcement.

---

## Architecture

```
FastAPI Application
    ↓
OPA Client
    ↓
OPA Server (Port 8181)
    ↓
Policy Engine
    ├─ access.rego (Access Control)
    ├─ quotas.rego (Quota Management)
    ├─ alerts.rego (Alert Rules)
    └─ audit.rego (Audit Compliance)
```

---

## Setup

### Docker Compose

```yaml
services:
  opa:
    image: openpolicyagent/opa:latest
    ports:
      - "8181:8181"
    volumes:
      - ./backend/shared/security/policies:/policies
    command: run --server --bundle /policies
    environment:
      - OPA_LOG_LEVEL=info
```

### Manual Installation

```bash
# Download OPA
curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_x86_64

# Make executable
chmod +x opa

# Run OPA server
./opa run --server
```

---

## Policy Files

### 1. Access Control (access.rego)

Defines who can access what resources and perform what actions.

**Key Rules**:

- Version access control (users → v2 only)
- Resource-specific access (experiments, projects, API keys)
- Promotion rules (accuracy > 0.85, error_rate < 0.05)
- Quarantine rules (admin only)

**Usage**:

```python
opa_client = OPAClient()

# Check permission
can_access = opa_client.check_permission(
    user={"id": "user123", "role": "user"},
    resource={"type": "version", "version": "v2"},
    action="read"
)

# Check version access
can_access_v2 = opa_client.check_version_access(
    user={"id": "user123", "role": "user"},
    version="v2"
)

# Check promotion
can_promote = opa_client.can_promote_version(
    user={"id": "admin123", "role": "admin"},
    metrics={
        "accuracy": 0.87,
        "error_rate": 0.03,
        "cost_increase": 0.15
    }
)
```

### 2. Quota Management (quotas.rego)

Defines quota limits and cost control rules.

**Default Quotas**:

- Admin: 10,000 daily requests, $1,000 daily cost limit
- User: 1,000 daily requests, $100 daily cost limit
- Viewer: 100 daily requests, $10 daily cost limit

**Usage**:

```python
# Check execution allowed
can_execute = opa_client.can_execute_analysis(
    user={"id": "user123", "role": "user"},
    analysis_config={"complexity": 0.5},
    current_usage={
        "daily_requests": 500,
        "daily_cost": 50,
        "concurrent_analyses": 3
    }
)

# Get quota status
quota_status = opa_client.get_quota_status("user123")
# Returns: {
#     "daily_requests": {"used": 500, "limit": 1000, "percentage": 50},
#     "daily_cost": {"used": 50, "limit": 100, "percentage": 50},
#     ...
# }
```

### 3. Alerts (alerts.rego)

Defines when alerts should be triggered.

**Alert Types**:

- Baseline violations (accuracy drop, error rate increase)
- Provider failures (consecutive failures ≥ 3)
- Quota alerts (approaching/exceeding limits)
- Security alerts (failed auth, unauthorized access)
- Experiment alerts (failures, timeouts, quarantine)

**Alert Severity**:

- Critical: On-call engineer
- Warning: Team notification
- Info: Log only

**Usage**:

```python
# Check if alert should be triggered
should_alert = opa_client.should_alert_on_violation(
    event={"type": "baseline_violated"},
    metrics={"metric_type": "accuracy", "deviation": 0.20}
)

# Get alert severity
severity = opa_client.get_alert_severity(
    event={"alert_name": "alert_daily_cost_exceeded"}
)
```

### 4. Audit (audit.rego)

Defines what actions should be audited.

**Audit Rules**:

- All admin actions
- Sensitive operations (promote, quarantine, delete)
- Access to sensitive resources (API keys, user data)
- Failed access attempts
- Authentication events

**Retention Policies**:

- Authentication: 90 days
- Authorization: 90 days
- Data Access: 1 year
- Sensitive Operations: 1 year
- Compliance: 7 years

**Usage**:

```python
# Check if action should be audited
should_audit = opa_client.should_audit_action(
    user={"id": "admin123", "role": "admin"},
    action="promote_version",
    resource={"type": "version", "id": "v1-exp-123"}
)
```

---

## FastAPI Integration

### Basic Setup

```python
from fastapi import FastAPI, Depends, HTTPException
from backend.shared.security.opa_client import OPAClient
from backend.shared.security.auth import CurrentUser

app = FastAPI()
opa = OPAClient(opa_url="http://localhost:8181")

# Check OPA health
if not opa.is_available():
    logger.error("OPA server not available")
```

### Protected Endpoints

```python
# Version-specific endpoint
@app.post("/analyze")
async def analyze_code(
    request: AnalysisRequest,
    user = Depends(CurrentUser.get_current_user)
):
    # Check version access
    if not opa.check_version_access(user, request.version):
        raise HTTPException(
            status_code=403,
            detail="Access denied to this version"
        )

    # Check quota
    usage = get_user_usage(user["id"])
    if not opa.can_execute_analysis(user, request.dict(), usage):
        raise HTTPException(
            status_code=429,
            detail="Quota exceeded"
        )

    return await analyze(request, user)

# Promotion endpoint
@app.post("/versions/{version_id}/promote")
async def promote_version(
    version_id: str,
    metrics: PromotionMetrics,
    admin = Depends(RoleBasedAccess.require_admin)
):
    # Check promotion policy
    if not opa.can_promote_version(admin, metrics.dict()):
        raise HTTPException(
            status_code=403,
            detail="Version does not meet promotion criteria"
        )

    return await version_service.promote(version_id, admin["id"])

# API key access
@app.get("/api-keys/{key_id}")
async def get_api_key(
    key_id: str,
    user = Depends(CurrentUser.get_current_user)
):
    key = db.get_api_key(key_id)

    # Check access
    if not opa.can_access_api_key(user, key.owner_id):
        raise HTTPException(
            status_code=403,
            detail="Access denied"
        )

    return key
```

### Audit Integration

```python
@app.post("/versions/{version_id}/promote")
async def promote_version(
    version_id: str,
    metrics: PromotionMetrics,
    admin = Depends(RoleBasedAccess.require_admin)
):
    # Check if should audit
    should_audit = opa.should_audit_action(
        user=admin,
        action="promote_version",
        resource={"type": "version", "id": version_id}
    )

    # Perform action
    result = await version_service.promote(version_id, admin["id"])

    # Audit if needed
    if should_audit:
        audit_log.record(
            user_id=admin["id"],
            action="promote_version",
            resource_id=version_id,
            status="success"
        )

    return result
```

---

## Policy Development

### Testing Policies

```bash
# Start OPA with test data
opa run --server

# Test policy in REPL
opa eval -d access.rego -i input.json 'data.code_review.access.allow'

# Test with curl
curl -X POST http://localhost:8181/v1/data/code_review/access/allow \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "user": {"id": "user123", "role": "user"},
      "resource": {"type": "version", "version": "v2"},
      "action": "read"
    }
  }'
```

### Policy Updates

```python
# Load new policy
policy_code = open("access.rego").read()
opa.load_policy("access", policy_code)

# Get current policy
current = opa.get_policy("access")

# Delete policy
opa.delete_policy("access")
```

---

## Monitoring

### Health Checks

```python
# Check OPA availability
if opa.is_available():
    logger.info("OPA is healthy")
else:
    logger.error("OPA is unavailable")
```

### Metrics

```python
# Track policy decisions
@app.middleware("http")
async def track_policy_decisions(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    # Record metric
    policy_decision_duration.observe(duration)

    return response
```

---

## Best Practices

1. **Policy Organization**

   - Separate policies by concern
   - Use clear naming conventions
   - Document policy intent

2. **Testing**

   - Test policies before deployment
   - Use test data and scenarios
   - Validate policy logic

3. **Performance**

   - Cache policy decisions where appropriate
   - Use efficient policy logic
   - Monitor policy evaluation time

4. **Security**

   - Regularly review policies
   - Audit policy changes
   - Implement policy versioning

5. **Maintenance**
   - Document policy changes
   - Keep policies updated
   - Monitor policy effectiveness

---

## Troubleshooting

### OPA Not Responding

```python
# Check connection
if not opa.is_available():
    logger.error("OPA server not available")
    # Fall back to default deny
    allow = False
```

### Policy Errors

```bash
# Check policy syntax
opa parse access.rego

# Validate policy
opa check access.rego

# Debug policy
opa eval -d access.rego 'data.code_review.access'
```

### Performance Issues

```python
# Cache policy decisions
@functools.lru_cache(maxsize=1000)
def check_permission_cached(user_id, resource_id, action):
    return opa.check_permission(user, resource, action)
```

---

## Future Enhancements

- [ ] Policy versioning
- [ ] A/B testing policies
- [ ] Policy performance optimization
- [ ] Advanced audit analytics
- [ ] Policy recommendation engine
- [ ] Integration with external systems
