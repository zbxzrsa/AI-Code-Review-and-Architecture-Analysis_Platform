# Security Implementation Guide

## Overview

Comprehensive security implementation covering authentication, authorization, provider health tracking, and audit logging.

---

## Authentication

### JWT Tokens

**Access Token**:

- Expires in 15 minutes (configurable)
- Contains: user_id, role, type, exp, iat
- Used for API requests

**Refresh Token**:

- Expires in 7 days (configurable)
- Contains: user_id, type, exp, iat
- Used to obtain new access tokens

### Token Creation

```python
from backend.shared.security.auth import TokenManager

# Create both tokens
access_token, refresh_token = TokenManager.create_tokens(
    user_id="user123",
    role="admin"
)

# Create access token only
access_token = TokenManager.create_access_token(
    user_id="user123",
    role="admin"
)

# Create refresh token only
refresh_token = TokenManager.create_refresh_token(
    user_id="user123"
)
```

### Token Verification

```python
# Verify access token
payload = TokenManager.verify_token(token, token_type="access")

# Verify refresh token
payload = TokenManager.verify_token(token, token_type="refresh")

# Refresh access token
new_access_token = TokenManager.refresh_access_token(refresh_token)
```

### Token Configuration

Set environment variables:

```bash
JWT_SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
```

---

## Authorization

### Role-Based Access Control (RBAC)

**Roles**:

- `admin` (level 3) - Full access
- `user` (level 2) - Standard access
- `viewer` (level 1) - Read-only access
- `guest` (level 0) - No access

### Role Hierarchy

```
admin (3)
  ↓
user (2)
  ↓
viewer (1)
  ↓
guest (0)
```

Higher level roles inherit permissions of lower levels.

### Using Role-Based Access

```python
from fastapi import Depends
from backend.shared.security.auth import RoleBasedAccess, CurrentUser

# Require admin role
@app.post("/admin/settings")
async def update_settings(
    user = Depends(RoleBasedAccess.require_admin)
):
    """Only admins can access."""
    return {"status": "updated"}

# Require user role or higher
@app.post("/projects")
async def create_project(
    user = Depends(RoleBasedAccess.require_user)
):
    """Users and admins can access."""
    return {"project_id": "123"}

# Get current user (any role)
@app.get("/profile")
async def get_profile(
    user = Depends(CurrentUser.get_current_user)
):
    """Any authenticated user can access."""
    return {"user_id": user["id"], "role": user["role"]}
```

### Fine-Grained Permissions

```python
from backend.shared.security.auth import PermissionManager

# Define permissions
permissions = {
    "projects": {
        "admin": ["create", "read", "update", "delete"],
        "user": ["create", "read", "update"],
        "viewer": ["read"],
        "guest": []
    },
    "settings": {
        "admin": ["read", "update"],
        "user": ["read"],
        "viewer": [],
        "guest": []
    }
}

# Check permission
has_permission = PermissionManager.check_permission(
    user=user,
    resource="projects",
    action="delete",
    permissions=permissions
)

# Require permission
@app.delete("/projects/{id}")
async def delete_project(
    id: str,
    user = Depends(
        PermissionManager.require_permission(
            "projects", "delete", permissions
        )
    )
):
    """Only users with delete permission can access."""
    return {"deleted": id}
```

---

## Provider Health Tracking

### Health Status

**Statuses**:

- `healthy` - Provider is working normally
- `degraded` - Provider has issues but is operational
- `unhealthy` - Provider is not operational

### Health Tracking

```python
from backend.shared.security.provider_health import ProviderHealthTracker
from backend.shared.cache.redis_client import RedisClient

redis = RedisClient()
health_tracker = ProviderHealthTracker(redis)

# Mark provider as healthy
health_tracker.mark_provider_healthy(
    provider="openai",
    response_time_ms=45.2
)

# Mark provider as degraded
health_tracker.mark_provider_degraded(
    provider="openai",
    reason="High latency"
)

# Mark provider as unhealthy
health_tracker.mark_provider_unhealthy(
    provider="openai",
    reason="Connection timeout",
    duration=300  # 5 minutes
)

# Check provider health
is_healthy = health_tracker.is_provider_healthy("openai")
is_available = health_tracker.is_provider_available("openai")

# Get provider status
status = health_tracker.get_provider_status("openai")
# Returns: {
#     "status": "healthy",
#     "last_check": "2024-12-02T10:00:00Z",
#     "response_time_ms": 45.2,
#     "consecutive_failures": 0
# }
```

### Failure Tracking

```python
# Record provider failure
health_tracker.record_provider_failure(
    provider="openai",
    error="Connection timeout",
    response_time_ms=5000
)

# Record provider success
health_tracker.record_provider_success(
    provider="openai",
    response_time_ms=45.2
)

# Get provider statistics
stats = health_tracker.get_provider_stats("openai")
# Returns: {
#     "provider": "openai",
#     "status": "healthy",
#     "consecutive_failures": 0,
#     "response_time_ms": 45.2
# }

# Get all providers statistics
all_stats = health_tracker.get_all_providers_stats([
    "openai", "anthropic", "huggingface"
])
```

### Health Checks

```python
# Perform health check
async def check_openai():
    # Custom health check logic
    return True

result = await health_tracker.perform_health_check(
    provider="openai",
    check_func=check_openai
)

# Schedule periodic health checks
providers = {
    "openai": check_openai,
    "anthropic": check_anthropic,
    "huggingface": check_huggingface
}

await health_tracker.schedule_health_checks(
    providers=providers,
    interval=300  # 5 minutes
)
```

### Failover Management

```python
# Get healthy provider from list
healthy_provider = health_tracker.get_healthy_provider(
    providers=["openai", "anthropic", "huggingface"],
    prefer_healthy=True
)

# Get fallback chain ordered by health
fallback_chain = health_tracker.get_fallback_chain(
    primary="openai",
    fallbacks=["anthropic", "huggingface", "local-model"]
)
# Returns: ["openai", "anthropic", "huggingface", "local-model"]
# (ordered by health status)
```

---

## Session Management

### Creating Sessions

```python
from backend.shared.security.auth import SessionManager

session_manager = SessionManager(redis)

# Create session
session = session_manager.create_session(
    user_id="user123",
    role="admin",
    device_info={"user_agent": "Mozilla/5.0...", "ip": "192.168.1.1"}
)
# Returns: {
#     "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
#     "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
#     "token_type": "bearer",
#     "expires_in": 900
# }
```

### Session Operations

```python
# Get session data
session = session_manager.get_session(user_id="user123", token=access_token)

# Update session activity
session_manager.update_session_activity(user_id="user123", token=access_token)

# Invalidate single session
session_manager.invalidate_session(user_id="user123", token=access_token)

# Invalidate all sessions
session_manager.invalidate_all_sessions(user_id="user123")
```

---

## Security Audit

### Logging Auth Events

```python
from backend.shared.security.auth import SecurityAudit

# Log authentication event
SecurityAudit.log_auth_event(
    user_id="user123",
    event_type="login",
    status="success",
    details={"ip": "192.168.1.1", "device": "web"}
)

# Log failed authentication
SecurityAudit.log_auth_event(
    user_id="user123",
    event_type="login",
    status="failure",
    details={"reason": "invalid_password"}
)
```

### Logging Access Events

```python
# Log successful access
SecurityAudit.log_access_event(
    user_id="user123",
    resource="projects",
    action="delete",
    status="success",
    details={"project_id": "proj456"}
)

# Log denied access
SecurityAudit.log_access_event(
    user_id="user123",
    resource="admin_settings",
    action="update",
    status="denied",
    details={"reason": "insufficient_permissions"}
)
```

---

## API Endpoint Examples

### Login Endpoint

```python
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel

app = FastAPI()

class LoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/login")
async def login(request: LoginRequest):
    # Verify credentials (check against database)
    user = verify_credentials(request.email, request.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Create session
    session = session_manager.create_session(
        user_id=user.id,
        role=user.role
    )

    return session
```

### Refresh Token Endpoint

```python
class RefreshRequest(BaseModel):
    refresh_token: str

@app.post("/auth/refresh")
async def refresh_token(request: RefreshRequest):
    try:
        new_access_token = TokenManager.refresh_access_token(
            request.refresh_token
        )
        return {
            "access_token": new_access_token,
            "token_type": "bearer"
        }
    except HTTPException as e:
        raise e
```

### Logout Endpoint

```python
@app.post("/auth/logout")
async def logout(user = Depends(CurrentUser.get_current_user)):
    # Invalidate session
    session_manager.invalidate_session(
        user_id=user["id"],
        token=request.headers.get("authorization", "").split(" ")[-1]
    )
    return {"status": "logged_out"}
```

### Protected Endpoint

```python
@app.get("/admin/users")
async def list_users(
    user = Depends(RoleBasedAccess.require_admin)
):
    """Only admins can list users."""
    return {"users": [...]}
```

---

## Best Practices

1. **Secret Management**

   - Never hardcode secrets
   - Use environment variables
   - Rotate secrets regularly
   - Use strong secret keys (32+ characters)

2. **Token Security**

   - Use short expiration times (15 min for access)
   - Implement refresh token rotation
   - Revoke tokens on logout
   - Validate token signature

3. **Password Security**

   - Use Argon2id hashing
   - Enforce strong password requirements
   - Implement rate limiting on login
   - Lock accounts after failed attempts

4. **HTTPS Only**

   - Always use HTTPS in production
   - Implement HSTS headers
   - Use TLS 1.3 minimum

5. **Audit Logging**

   - Log all authentication events
   - Log all access attempts
   - Store logs securely
   - Monitor for suspicious activity

6. **Rate Limiting**

   - Limit login attempts
   - Limit API requests per user
   - Implement exponential backoff
   - Block after threshold

7. **Session Management**

   - Invalidate sessions on logout
   - Implement session timeout
   - Track device information
   - Support multi-device sessions

8. **Provider Health**
   - Monitor provider availability
   - Implement automatic failover
   - Track response times
   - Alert on failures

---

## Security Checklist

- [ ] JWT secret key configured
- [ ] Token expiration times set
- [ ] HTTPS enabled in production
- [ ] Rate limiting implemented
- [ ] Audit logging enabled
- [ ] Session management configured
- [ ] Provider health monitoring active
- [ ] RBAC rules defined
- [ ] Permissions validated
- [ ] Security headers configured
- [ ] Input validation implemented
- [ ] SQL injection prevention
- [ ] XSS prevention
- [ ] CSRF protection
- [ ] Regular security audits

---

## Troubleshooting

### Invalid Token Error

- Check token expiration
- Verify JWT secret key
- Ensure token format is correct
- Check token type (access vs refresh)

### Permission Denied

- Verify user role
- Check permission configuration
- Ensure role hierarchy is correct
- Review audit logs

### Provider Unhealthy

- Check provider connectivity
- Review error logs
- Check response times
- Verify API keys

### Session Issues

- Check Redis connection
- Verify session TTL
- Review session data
- Check token validity
