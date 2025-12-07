"""
Authentication and Authorization API Endpoints

Contains all API endpoints related to:
- User login/logout
- Token management (access/refresh)
- Password reset
- Session management
- API key authentication

Module Size: ~300 lines (target < 2000)
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel, EmailStr, Field
import hashlib
import secrets

from ..config import logger

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


# =============================================================================
# Constants
# =============================================================================

BEARER_PREFIX = "Bearer "
ERR_INVALID_AUTH_HEADER = "Invalid authorization header"
ERR_INVALID_SESSION = "Invalid or expired session"


# =============================================================================
# Request/Response Models
# =============================================================================

class LoginRequest(BaseModel):
    """Login request payload."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    remember_me: bool = False


class LoginResponse(BaseModel):
    """Login response with tokens."""
    access_token: str
    refresh_token: str
    token_type: str = "Bearer"
    expires_in: int
    user: dict


class RegisterRequest(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    name: str = Field(..., min_length=2)
    invitation_code: Optional[str] = None


class RefreshTokenRequest(BaseModel):
    """Token refresh request."""
    refresh_token: str


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation."""
    token: str
    new_password: str = Field(..., min_length=8)


class ChangePasswordRequest(BaseModel):
    """Change password request."""
    current_password: str
    new_password: str = Field(..., min_length=8)


# =============================================================================
# Mock Data Store
# =============================================================================

MOCK_USERS = {
    "admin@example.com": {
        "id": "user-001",
        "email": "admin@example.com",
        "name": "Admin User",
        "role": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
    },
    "user@example.com": {
        "id": "user-002",
        "email": "user@example.com",
        "name": "Regular User",
        "role": "user",
        "password_hash": hashlib.sha256("user123".encode()).hexdigest(),
    },
}

MOCK_SESSIONS = {}
MOCK_RESET_TOKENS = {}


# =============================================================================
# Helper Functions
# =============================================================================

def generate_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    return hashlib.sha256(password.encode()).hexdigest() == password_hash


def create_access_token(user_id: str, expires_minutes: int = 15) -> tuple[str, datetime]:
    """Create access token."""
    token = generate_token()
    expires_at = datetime.now(timezone.utc) + timedelta(minutes=expires_minutes)
    return token, expires_at


# =============================================================================
# Authentication Endpoints
# =============================================================================

@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """
    Authenticate user and return tokens.

    - **email**: User email address
    - **password**: User password
    - **remember_me**: Extend session duration
    """
    user = MOCK_USERS.get(request.email)

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Create tokens
    expires_minutes = 10080 if request.remember_me else 15  # 7 days or 15 minutes
    access_token, expires_at = create_access_token(user["id"], expires_minutes)
    refresh_token = generate_token(48)

    # Store session
    MOCK_SESSIONS[access_token] = {
        "user_id": user["id"],
        "expires_at": expires_at.isoformat(),
        "refresh_token": refresh_token,
    }

    logger.info(f"User logged in: {user['email']}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_minutes * 60,
        user={
            "id": user["id"],
            "email": user["email"],
            "name": user["name"],
            "role": user["role"],
        }
    )


@router.post("/register")
async def register(request: RegisterRequest):
    """
    Register a new user account.

    - **email**: Email address (must be unique)
    - **password**: Password (min 8 characters)
    - **name**: Display name
    - **invitation_code**: Optional invitation code for restricted signup
    """
    if request.email in MOCK_USERS:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create new user
    user_id = f"user-{len(MOCK_USERS) + 1:03d}"
    MOCK_USERS[request.email] = {
        "id": user_id,
        "email": request.email,
        "name": request.name,
        "role": "user",
        "password_hash": hashlib.sha256(request.password.encode()).hexdigest(),
    }

    logger.info(f"New user registered: {request.email}")

    return {
        "message": "Registration successful",
        "user_id": user_id,
    }


@router.post("/refresh")
async def refresh_token(request: RefreshTokenRequest):
    """
    Refresh access token using refresh token.
    """
    # Find session by refresh token
    for access_token, session in MOCK_SESSIONS.items():
        if session.get("refresh_token") == request.refresh_token:
            # Create new access token
            user_id = session["user_id"]
            new_access_token, expires_at = create_access_token(user_id)
            new_refresh_token = generate_token(48)

            # Update session
            del MOCK_SESSIONS[access_token]
            MOCK_SESSIONS[new_access_token] = {
                "user_id": user_id,
                "expires_at": expires_at.isoformat(),
                "refresh_token": new_refresh_token,
            }

            return {
                "access_token": new_access_token,
                "refresh_token": new_refresh_token,
                "token_type": "Bearer",
                "expires_in": 900,
            }

    raise HTTPException(status_code=401, detail="Invalid refresh token")


@router.post("/logout")
async def logout(authorization: Optional[str] = Header(None)):
    """
    Logout and invalidate current session.
    """
    if authorization and authorization.startswith(BEARER_PREFIX):
        token = authorization.split(" ")[1]
        if token in MOCK_SESSIONS:
            del MOCK_SESSIONS[token]

    return {"message": "Logged out successfully"}


@router.post("/password/reset")
async def request_password_reset(request: PasswordResetRequest):
    """
    Request password reset email.
    """
    user = MOCK_USERS.get(request.email)

    # Always return success to prevent email enumeration
    if user:
        reset_token = generate_token()
        MOCK_RESET_TOKENS[reset_token] = {
            "user_email": request.email,
            "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        }
        logger.info(f"Password reset requested for: {request.email}")

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password/reset/confirm")
async def confirm_password_reset(request: PasswordResetConfirm):
    """
    Confirm password reset with token.
    """
    reset_info = MOCK_RESET_TOKENS.get(request.token)

    if not reset_info:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    # Check expiration
    expires_at = datetime.fromisoformat(reset_info["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        del MOCK_RESET_TOKENS[request.token]
        raise HTTPException(status_code=400, detail="Reset token has expired")

    # Update password
    email = reset_info["user_email"]
    if email in MOCK_USERS:
        MOCK_USERS[email]["password_hash"] = hashlib.sha256(request.new_password.encode()).hexdigest()

    del MOCK_RESET_TOKENS[request.token]

    return {"message": "Password reset successful"}


@router.post("/password/change")
async def change_password(
    request: ChangePasswordRequest,
    authorization: str = Header(...),
):
    """
    Change password for authenticated user.
    """
    if not authorization.startswith(BEARER_PREFIX):
        raise HTTPException(status_code=401, detail=ERR_INVALID_AUTH_HEADER)

    token = authorization.split(" ")[1]
    session = MOCK_SESSIONS.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Find user
    user_id = session["user_id"]
    user = next((u for u in MOCK_USERS.values() if u["id"] == user_id), None)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if not verify_password(request.current_password, user["password_hash"]):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    # Update password
    user["password_hash"] = hashlib.sha256(request.new_password.encode()).hexdigest()

    return {"message": "Password changed successfully"}


@router.get("/me")
async def get_current_user(authorization: str = Header(...)):
    """
    Get current authenticated user.
    """
    if not authorization.startswith(BEARER_PREFIX):
        raise HTTPException(status_code=401, detail=ERR_INVALID_AUTH_HEADER)

    token = authorization.split(" ")[1]
    session = MOCK_SESSIONS.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # Check expiration
    expires_at = datetime.fromisoformat(session["expires_at"])
    if datetime.now(timezone.utc) > expires_at:
        del MOCK_SESSIONS[token]
        raise HTTPException(status_code=401, detail="Session expired")

    # Find user
    user_id = session["user_id"]
    user = next((u for u in MOCK_USERS.values() if u["id"] == user_id), None)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    return {
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
    }


@router.get("/sessions")
async def list_sessions(authorization: str = Header(...)):
    """
    List all active sessions for current user.
    """
    if not authorization.startswith(BEARER_PREFIX):
        raise HTTPException(status_code=401, detail=ERR_INVALID_AUTH_HEADER)

    token = authorization.split(" ")[1]
    session = MOCK_SESSIONS.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    user_id = session["user_id"]

    # Get all sessions for this user
    user_sessions = [
        {
            "token_prefix": t[:8] + "...",
            "expires_at": s["expires_at"],
            "is_current": t == token,
        }
        for t, s in MOCK_SESSIONS.items()
        if s["user_id"] == user_id
    ]

    return {"sessions": user_sessions}


@router.delete("/sessions")
async def revoke_all_sessions(authorization: str = Header(...)):
    """
    Revoke all sessions except current.
    """
    if not authorization.startswith(BEARER_PREFIX):
        raise HTTPException(status_code=401, detail=ERR_INVALID_AUTH_HEADER)

    token = authorization.split(" ")[1]
    session = MOCK_SESSIONS.get(token)

    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")

    user_id = session["user_id"]

    # Remove all other sessions
    tokens_to_remove = [
        t for t, s in MOCK_SESSIONS.items()
        if s["user_id"] == user_id and t != token
    ]

    for t in tokens_to_remove:
        del MOCK_SESSIONS[t]

    return {"message": f"Revoked {len(tokens_to_remove)} sessions"}
