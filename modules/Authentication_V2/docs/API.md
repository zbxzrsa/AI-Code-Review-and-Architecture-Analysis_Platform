# Authentication_V2 API Reference

## Overview

Production authentication with MFA, OAuth, and enhanced security.

## Classes

### AuthManager

Core authentication with MFA support.

```python
from modules.Authentication_V2.src.auth_manager import AuthManager

auth = AuthManager(
    access_token_ttl=900,
    max_failed_attempts=5
)

# Register with MFA
result = await auth.register(
    email="user@example.com",
    password="secure123",
    enable_mfa=True
)

# Login (MFA flow)
result = await auth.login("user@example.com", "secure123")
if result.requires_mfa:
    result = await auth.verify_mfa_step(result.mfa_token, "123456")
```

### MFAService

TOTP-based multi-factor authentication.

```python
from modules.Authentication_V2.src.mfa_service import MFAService

mfa = MFAService(issuer="MyApp")

# Setup MFA
setup = mfa.setup_mfa("user-123", "user@example.com")
print(f"QR URI: {setup.qr_uri}")
print(f"Backup codes: {setup.backup_codes}")

# Verify TOTP
is_valid = mfa.verify_totp(setup.secret, "123456")

# Use backup code
is_valid = mfa.verify_backup_code("user-123", "ABCD1234")
```

### SessionManager

Enhanced session management with device tracking.

```python
from modules.Authentication_V2.src.session_manager import SessionManager, DeviceInfo

sessions = SessionManager(max_sessions_per_user=5)

# Create session with device info
device = DeviceInfo(
    device_id="dev-123",
    device_type="web",
    browser="Chrome"
)

session = await sessions.create_session(
    "user-123",
    device_info=device,
    trust_device=True
)

# Get all user sessions
user_sessions = await sessions.get_user_sessions("user-123")
```

### OAuthProvider

OAuth 2.0 integration.

```python
from modules.Authentication_V2.src.oauth_provider import OAuthProvider, OAuthConfig

oauth = OAuthProvider(redirect_uri="https://app.com/callback")

# Register Google
oauth.register_provider(OAuthConfig.google(
    client_id="...",
    client_secret="..."
))

# Get authorization URL
url, state = oauth.get_authorization_url(OAuthProviderType.GOOGLE)

# Handle callback
user = await oauth.handle_callback(code, state)
```

## Configuration

See `config/auth_config.yaml`
