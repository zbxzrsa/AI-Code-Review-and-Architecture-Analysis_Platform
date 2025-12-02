# Security Implementation Guide

## Overview

This document describes the comprehensive security implementation for the AI Code Review Platform frontend.

## Table of Contents

1. [Token Storage Security](#token-storage-security)
2. [CSRF Protection](#csrf-protection)
3. [Rate Limiting](#rate-limiting)
4. [Two-Factor Authentication](#two-factor-authentication)
5. [Security Headers](#security-headers)
6. [Input Validation](#input-validation)
7. [Session Management](#session-management)

---

## Token Storage Security

### Problem

Storing JWT tokens in localStorage is vulnerable to XSS attacks. Any JavaScript running on the page can access localStorage.

### Solution: httpOnly Cookies

Tokens are now stored in **httpOnly cookies** instead of localStorage:

```typescript
// OLD (INSECURE):
localStorage.setItem("token", jwt);

// NEW (SECURE):
// Server sets httpOnly cookie - JavaScript cannot access it
```

### Implementation

#### Backend Requirements

```python
# FastAPI example
@app.post("/auth/login")
async def login(response: Response, credentials: LoginRequest):
    # ... validate credentials ...

    access_token = create_access_token(user)
    refresh_token = create_refresh_token(user)

    # Set httpOnly cookies
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,  # HTTPS only in production
        samesite="strict",
        max_age=900,  # 15 minutes
    )
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=604800,  # 7 days
    )

    return {"user": user, "requires_two_factor": user.two_factor_enabled}
```

#### Frontend Implementation

```typescript
// API client with cookie handling
const api = axios.create({
  baseURL: "/api",
  withCredentials: true, // CRITICAL: Send cookies with requests
});

// No more Authorization header - cookies are automatic
// No more token storage in localStorage
```

### Auth Store Changes

The auth store no longer stores tokens:

```typescript
// BEFORE
partialize: (state) => ({
  token: state.token, // REMOVED - insecure
  refreshToken: state.refreshToken, // REMOVED - insecure
  user: state.user,
});

// AFTER
partialize: (state) => ({
  // Only persist non-sensitive user data for UX
  user: state.user
    ? {
        id: state.user.id,
        name: state.user.name,
        avatar: state.user.avatar,
        role: state.user.role,
      }
    : null,
});
```

---

## CSRF Protection

### Problem

Cross-Site Request Forgery attacks can trick authenticated users into performing unwanted actions.

### Solution: Double-Submit Cookie Pattern

```typescript
// 1. Server generates CSRF token
// 2. Frontend stores in memory (NOT localStorage)
// 3. Frontend sends token in X-CSRF-Token header
// 4. Server validates token
```

### Implementation

#### CSRF Manager (Memory Storage)

```typescript
import { csrfManager } from "@/services/security";

// Get current token
const token = csrfManager.getToken();

// Set token (called after login)
csrfManager.setToken(token);

// Clear on logout
csrfManager.clearToken();

// Fetch fresh token
await csrfManager.fetchToken();
```

#### Automatic CSRF Headers

The API interceptor automatically adds CSRF tokens to state-changing requests:

```typescript
api.interceptors.request.use(async (config) => {
  if (["post", "put", "patch", "delete"].includes(config.method)) {
    const csrfToken = await csrfManager.ensureToken();
    config.headers["X-CSRF-Token"] = csrfToken;
  }
  return config;
});
```

#### Backend Requirements

```python
from fastapi import Request, HTTPException

async def validate_csrf(request: Request):
    if request.method in ['POST', 'PUT', 'PATCH', 'DELETE']:
        csrf_token = request.headers.get('X-CSRF-Token')
        session_csrf = request.session.get('csrf_token')

        if not csrf_token or csrf_token != session_csrf:
            raise HTTPException(403, "Invalid CSRF token")
```

---

## Rate Limiting

### Client-Side Rate Limiting

Prevents excessive requests before they hit the server with cooldown UI:

```typescript
import { useRateLimiter, rateLimitConfigs } from "@/hooks";
import { RateLimitAlert } from "@/components/common";

function LoginForm() {
  const {
    status,
    checkLimit,
    handleRateLimitResponse,
    executeWithBackoff,
    canSubmit,
    cooldownText,
  } = useRateLimiter("/auth/login", rateLimitConfigs.login);

  const handleSubmit = async () => {
    if (!canSubmit) return;

    try {
      // Execute with exponential backoff
      await executeWithBackoff(async () => {
        await apiService.auth.login(email, password);
      });
    } catch (error) {
      if (error.response?.status === 429) {
        handleRateLimitResponse(error.response.headers["retry-after"]);
      }
    }
  };

  return (
    <>
      <RateLimitAlert status={status} />
      <Button disabled={!canSubmit} onClick={handleSubmit}>
        {status.isLimited ? `Wait ${cooldownText}` : "Login"}
      </Button>
    </>
  );
}
```

### Pre-configured Limits

| Endpoint                | Max Requests | Window   | Description                  |
| ----------------------- | ------------ | -------- | ---------------------------- |
| `/auth/login`           | 5            | 15 min   | Login attempts per IP/email  |
| `/auth/register`        | 3            | 1 min    | Registration attempts per IP |
| `/user/password/change` | 3            | 24 hours | Password changes per user    |
| `/user/password/reset`  | 3            | 5 min    | Password reset requests      |
| `/user/api-keys`        | 10           | 1 hour   | API key generation           |
| `/auth/2fa/verify`      | 5            | 1 min    | 2FA verification attempts    |
| General API             | 100          | 1 min    | All other endpoints          |

### Exponential Backoff

Automatic retry with increasing delays:

```typescript
const backoffConfig = {
  initialDelayMs: 1000, // Start with 1 second
  maxDelayMs: 60000, // Max 1 minute
  multiplier: 2, // Double each time
  maxRetries: 5, // Max 5 retries
};

const result = await executeWithBackoff(fn, backoffConfig);
```

### Backend Rate Limiting (Redis)

```python
from app.middleware import RateLimitMiddleware, rate_limit

# Global middleware
app.add_middleware(RateLimitMiddleware)

# Per-endpoint decorator
@app.post("/api/auth/login")
@rate_limit(requests=5, window_seconds=900, per_ip=True)
async def login(request: Request):
    ...
```

### API Response Headers

```http
X-RateLimit-Limit: 5
X-RateLimit-Remaining: 3
X-RateLimit-Reset: 1699999999
Retry-After: 60
```

### 429 Response Format

```json
{
  "error": "Too Many Requests",
  "message": "Rate limit exceeded. Try again in 60 seconds.",
  "retry_after": 60
}
```

---

## Two-Factor Authentication

### Overview

The 2FA implementation uses TOTP (Time-based One-Time Password) with:

- **pyotp** for TOTP generation and verification
- **qrcode** for QR code generation
- Support for popular authenticator apps (Google Authenticator, Authy, Microsoft Authenticator, 1Password)

### Components

```typescript
import { TwoFactorVerify, TwoFactorSetup, TwoFactorDisable } from '@/components/auth';
import { TwoFactorSettings } from '@/components/settings';

// During login
<TwoFactorVerify
  onVerify={async (code, isBackupCode) => {
    const result = await apiService.auth.verify2FA(code, isBackupCode);
    return result.success;
  }}
  onCancel={() => navigate('/login')}
/>

// Setup 2FA (in settings)
<TwoFactorSetup
  onComplete={() => {
    message.success('2FA enabled!');
    fetchStatus();
  }}
  onCancel={handleCancel}
/>

// 2FA Management in Settings
<TwoFactorSettings />
```

### Setup Flow

```
User → Settings → Enable 2FA
         ↓
   Server generates secret
         ↓
   QR code displayed
         ↓
   User scans with authenticator app
         ↓
   User enters 6-digit code to verify
         ↓
   Server generates 10 backup codes
         ↓
   User downloads/saves backup codes
         ↓
   2FA is enabled!
```

### Login Flow with 2FA

```
User → Login with email/password
         ↓
   Server validates credentials
         ↓
   Server returns { requires_two_factor: true }
         ↓
   Frontend shows 2FA verification screen
         ↓
   User enters 6-digit code or backup code
         ↓
   Server validates code
         ↓
   User is logged in!
```

### API Endpoints

| Endpoint                            | Method | Description                          |
| ----------------------------------- | ------ | ------------------------------------ |
| `/auth/2fa/setup`                   | POST   | Generate TOTP secret and QR code     |
| `/auth/2fa/verify`                  | POST   | Verify TOTP code during setup        |
| `/auth/2fa/enable`                  | POST   | Enable 2FA for user                  |
| `/auth/2fa/disable`                 | POST   | Disable 2FA (requires password/code) |
| `/auth/2fa/backup-codes`            | GET    | Get backup code status               |
| `/auth/2fa/backup-codes/regenerate` | POST   | Regenerate backup codes              |
| `/auth/2fa/status`                  | GET    | Get 2FA status                       |
| `/auth/login/verify-2fa`            | POST   | Verify 2FA during login              |

### Backup Codes

- **10 codes** generated during setup
- Each code is **8 alphanumeric characters**
- Formatted as `XXXX-XXXX` for readability
- **One-time use only** - removed after successful use
- Stored as **SHA256 hashes** (not plain text)
- User should download/print immediately

```typescript
// Backup code verification
const { is_valid, used_hash } = await verifyBackupCode(code, hashedCodes);
if (is_valid) {
  // Remove used code from list
  hashedCodes.remove(used_hash);
}
```

### Security Features

1. **Rate Limiting**

   - 5 attempts per 15 minutes per user
   - Lockout after max attempts
   - Different users tracked independently

2. **Secret Encryption**

   - TOTP secrets encrypted with AES-256-GCM
   - KMS master key management
   - Never stored in plain text

3. **Code Validation**
   - 30-second time window
   - 1 interval tolerance for clock drift
   - Handles spaces and dashes in input

### Backend Implementation

```python
from app.services.two_factor import two_factor_service

# Generate secret and QR code
secret = two_factor_service.generate_secret()
qr_code = two_factor_service.generate_qr_code(secret, user.email)

# Verify TOTP code
is_valid = two_factor_service.verify_code(secret, code)

# Generate backup codes
backup_codes = two_factor_service.generate_backup_codes()
hashed_codes = [two_factor_service.hash_backup_code(c) for c in backup_codes]
```

### Troubleshooting

| Issue                | Solution                             |
| -------------------- | ------------------------------------ |
| Code always invalid  | Check device time sync               |
| QR code won't scan   | Use manual secret entry              |
| Lost authenticator   | Use backup code                      |
| No backup codes left | Contact support for account recovery |

---

## Security Headers

### Required Headers

```http
Content-Security-Policy: default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Strict-Transport-Security: max-age=31536000; includeSubDomains
```

### Vite Configuration

```typescript
// vite.config.ts
export default defineConfig({
  server: {
    headers: {
      "X-Content-Type-Options": "nosniff",
      "X-Frame-Options": "DENY",
      "X-XSS-Protection": "1; mode=block",
    },
  },
});
```

### CSP Violation Reporting

```typescript
import { setupCSPReporting } from "@/services/security";

// In App.tsx
useEffect(() => {
  setupCSPReporting();
}, []);
```

---

## Input Validation

### Password Strength

```typescript
import { validatePasswordStrength } from "@/services/security";

const result = validatePasswordStrength(password);
// {
//   score: 3,        // 0-4
//   isStrong: true,  // true if score >= 3
//   feedback: [],    // improvement suggestions
// }
```

### Password Requirements

- Minimum 8 characters
- Mixed case (uppercase and lowercase)
- At least one number
- At least one special character
- No common patterns (password, 123456, etc.)

### Email Validation

```typescript
import { isValidEmail } from "@/services/security";

if (!isValidEmail(email)) {
  throw new Error("Invalid email");
}
```

### Input Sanitization

```typescript
import { sanitizeInput } from "@/services/security";

const safe = sanitizeInput(userInput);
// Escapes HTML entities to prevent XSS
```

---

## Session Management

### Inactivity Timeout

Sessions expire after 15 minutes of inactivity:

```typescript
import { sessionSecurity } from "@/services/security";

// Start timer when user logs in
sessionSecurity.startInactivityTimer(() => {
  notify.warning("Session expired due to inactivity");
  logout();
});

// Activity is tracked automatically on mouse/keyboard events
```

### Session Verification

On page load, verify session is still valid:

```typescript
useEffect(() => {
  const verifySession = async () => {
    try {
      await apiService.auth.me();
    } catch {
      logout();
    }
  };
  verifySession();
}, []);
```

---

## Security Checklist

### Before Deployment

- [ ] All tokens stored in httpOnly cookies
- [ ] CSRF protection enabled
- [ ] Rate limiting configured
- [ ] Security headers set
- [ ] 2FA available for users
- [ ] Password strength requirements enforced
- [ ] Input sanitization in place
- [ ] Session timeout configured
- [ ] HTTPS enforced in production
- [ ] CSP configured and tested

### Monitoring

- [ ] Failed login attempts logged
- [ ] CSP violations reported
- [ ] Rate limit violations tracked
- [ ] Session anomalies detected
- [ ] API errors logged with context

---

## Usage Examples

### Secure Login Flow

```typescript
import { useSecureAuth } from "@/hooks/useSecureAuth";

const LoginPage = () => {
  const { login, verifyTwoFactor, pendingTwoFactor, isLoading } =
    useSecureAuth();

  const handleLogin = async (values) => {
    const result = await login({
      email: values.email,
      password: values.password,
    });

    if (result.requiresTwoFactor) {
      // Show 2FA verification UI
    }
  };

  if (pendingTwoFactor) {
    return (
      <TwoFactorVerify
        onVerify={verifyTwoFactor}
        onCancel={() => window.location.reload()}
      />
    );
  }

  return <LoginForm onSubmit={handleLogin} loading={isLoading} />;
};
```

### Protected API Calls

```typescript
// CSRF token is automatically added to POST/PUT/PATCH/DELETE
await apiService.projects.create({ name: "New Project" });

// httpOnly cookies are automatically sent
await apiService.auth.me();
```
