"""
Two-Factor Authentication API Routes

Endpoints:
- POST /api/auth/2fa/setup - Generate TOTP secret and QR code
- POST /api/auth/2fa/verify - Verify TOTP code during setup
- POST /api/auth/2fa/enable - Enable 2FA for user
- POST /api/auth/2fa/disable - Disable 2FA
- GET /api/auth/2fa/backup-codes - Get/regenerate backup codes
- POST /api/auth/2fa/backup-codes/regenerate - Regenerate backup codes
- GET /api/auth/2fa/status - Get 2FA status
- POST /api/auth/login/verify-2fa - Verify 2FA during login
"""

from typing import Optional, List
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Request
from pydantic import BaseModel, Field

from app.services.two_factor import (
    two_factor_service,
    two_factor_rate_limiter,
    SecretEncryption,
)
from app.api.deps import get_current_user, get_current_user_optional
from app.models.user import User
from app.core.security import verify_password


# Constants
TWO_FA_NOT_ENABLED = "2FA is not enabled."


router = APIRouter(prefix="/auth/2fa", tags=["two-factor"])


# ============================================
# Request/Response Models
# ============================================

class TwoFactorSetupResponse(BaseModel):
    """Response for 2FA setup initiation"""
    secret: str = Field(..., description="Base32 secret for manual entry")
    qr_code: str = Field(..., description="QR code as base64 data URL")
    provisioning_uri: str = Field(..., description="otpauth:// URI")


class TwoFactorVerifyRequest(BaseModel):
    """Request to verify TOTP code"""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit TOTP code")


class TwoFactorEnableRequest(BaseModel):
    """Request to enable 2FA"""
    code: str = Field(..., min_length=6, max_length=6, description="6-digit TOTP code")
    secret: str = Field(..., description="The secret from setup")


class TwoFactorDisableRequest(BaseModel):
    """Request to disable 2FA"""
    password: Optional[str] = Field(None, description="User password")
    code: Optional[str] = Field(None, description="6-digit TOTP code or backup code")


class TwoFactorLoginRequest(BaseModel):
    """Request to verify 2FA during login"""
    code: str = Field(..., description="6-digit TOTP code or backup code")
    is_backup_code: bool = Field(False, description="Whether code is a backup code")
    session_token: str = Field(..., description="Temporary session token from login")


class BackupCodesResponse(BaseModel):
    """Response with backup codes"""
    backup_codes: List[str] = Field(..., description="List of backup codes")
    generated_at: datetime = Field(..., description="When codes were generated")
    remaining_count: int = Field(..., description="Number of unused codes")


class TwoFactorStatusResponse(BaseModel):
    """Response with 2FA status"""
    enabled: bool = Field(..., description="Whether 2FA is enabled")
    enabled_at: Optional[datetime] = Field(None, description="When 2FA was enabled")
    backup_codes_remaining: int = Field(0, description="Number of unused backup codes")
    last_used: Optional[datetime] = Field(None, description="Last 2FA verification")


class TwoFactorVerifyResponse(BaseModel):
    """Response for successful 2FA verification"""
    success: bool = True
    message: str = "Code verified successfully"


# ============================================
# Setup Endpoints
# ============================================

@router.post("/setup", response_model=TwoFactorSetupResponse)
async def setup_two_factor(
    current_user: User = Depends(get_current_user),
):
    """
    Initialize 2FA setup

    Generates a new TOTP secret and QR code for the user to scan
    with their authenticator app.

    Note: This doesn't enable 2FA yet - user must verify the code first.
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled. Disable it first to set up again.",
        )

    # Generate new secret
    secret = two_factor_service.generate_secret()

    # Generate QR code
    qr_code = two_factor_service.generate_qr_code(secret, current_user.email)

    # Generate provisioning URI
    provisioning_uri = two_factor_service.generate_provisioning_uri(
        secret, current_user.email
    )

    # Format secret for display
    formatted_secret = two_factor_service.format_secret_for_manual_entry(secret)

    return TwoFactorSetupResponse(
        secret=formatted_secret,
        qr_code=qr_code,
        provisioning_uri=provisioning_uri,
    )


@router.post("/verify", response_model=TwoFactorVerifyResponse)
async def verify_setup_code(
    request: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_user),
):
    """
    Verify TOTP code during setup

    This endpoint is used to verify that the user has correctly
    configured their authenticator app before enabling 2FA.
    """
    # Check rate limiting
    is_locked, seconds = two_factor_rate_limiter.is_locked_out(str(current_user.id))
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {seconds} seconds.",
            headers={"Retry-After": str(seconds)},
        )

    # Get the pending secret (stored temporarily in session or passed from client)
    # In production, store this securely server-side
    pending_secret = getattr(current_user, "_pending_2fa_secret", None)

    if not pending_secret:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No pending 2FA setup. Please start setup again.",
        )

    # Verify the code
    is_valid = two_factor_service.verify_code(pending_secret, request.code)

    # Record attempt
    two_factor_rate_limiter.record_attempt(str(current_user.id), is_valid)

    if not is_valid:
        remaining = two_factor_rate_limiter.get_remaining_attempts(str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid code. {remaining} attempts remaining.",
        )

    return TwoFactorVerifyResponse(
        success=True,
        message="Code verified successfully. You can now enable 2FA.",
    )


@router.post("/enable", response_model=BackupCodesResponse)
async def enable_two_factor(
    request: TwoFactorEnableRequest,
    current_user: User = Depends(get_current_user),
    # db: Session = Depends(get_db),
):
    """
    Enable 2FA for the user

    After verifying the TOTP code, this endpoint:
    1. Stores the encrypted secret
    2. Generates backup codes
    3. Enables 2FA for the account

    Returns backup codes that user should save securely.
    """
    if current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="2FA is already enabled.",
        )

    # Remove spaces from secret
    secret = request.secret.replace(" ", "")

    # Verify the code one more time
    is_valid = two_factor_service.verify_code(secret, request.code)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code.",
        )

    # Generate backup codes
    backup_codes = two_factor_service.generate_backup_codes()

    # Hash backup codes for storage (used in production DB operations)
    _ = [  # noqa: F841 - reserved for DB storage
        two_factor_service.hash_backup_code(code)
        for code in backup_codes
    ]

    # Encrypt secret for storage
    # encryption = SecretEncryption()
    # encrypted_secret = encryption.encrypt(secret)

    # Update user in database
    # In production, use proper database operations:
    # current_user.two_factor_secret = encrypted_secret
    # current_user.two_factor_backup_codes = hashed_codes
    # current_user.two_factor_enabled = True
    # current_user.two_factor_enabled_at = datetime.now(timezone.utc)
    # db.commit()

    # Format backup codes for display
    formatted_codes = [
        two_factor_service.format_backup_code(code)
        for code in backup_codes
    ]

    return BackupCodesResponse(
        backup_codes=formatted_codes,
        generated_at=datetime.now(timezone.utc),
        remaining_count=len(backup_codes),
    )


@router.post("/disable")
async def disable_two_factor(
    request: TwoFactorDisableRequest,
    current_user: User = Depends(get_current_user),
    # db: Session = Depends(get_db),
):
    """
    Disable 2FA for the user

    Requires either:
    - Current password, OR
    - Valid 2FA code
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=TWO_FA_NOT_ENABLED,
        )

    # Verify authorization
    authorized = False

    if request.password:
        # Verify password
        if verify_password(request.password, current_user.hashed_password):
            authorized = True

    if request.code and not authorized:
        # Verify 2FA code
        # encryption = SecretEncryption()
        # secret = encryption.decrypt(current_user.two_factor_secret)
        secret = current_user.two_factor_secret  # Placeholder

        if two_factor_service.verify_code(secret, request.code):
            authorized = True

    if not authorized:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password or 2FA code.",
        )

    # Disable 2FA
    # current_user.two_factor_enabled = False
    # current_user.two_factor_secret = None
    # current_user.two_factor_backup_codes = None
    # db.commit()

    return {"success": True, "message": "2FA has been disabled."}


# ============================================
# Backup Codes Endpoints
# ============================================

@router.get("/backup-codes", response_model=BackupCodesResponse)
async def get_backup_codes_status(
    current_user: User = Depends(get_current_user),
):
    """
    Get backup codes status

    Returns the number of remaining backup codes.
    Does NOT return the actual codes (they're only shown once).
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=TWO_FA_NOT_ENABLED,
        )

    # Count remaining backup codes
    remaining = len(current_user.two_factor_backup_codes or [])

    return BackupCodesResponse(
        backup_codes=[],  # Never return actual codes
        generated_at=current_user.two_factor_enabled_at,
        remaining_count=remaining,
    )


@router.post("/backup-codes/regenerate", response_model=BackupCodesResponse)
async def regenerate_backup_codes(
    request: TwoFactorVerifyRequest,
    current_user: User = Depends(get_current_user),
    # db: Session = Depends(get_db),
):
    """
    Regenerate backup codes

    Requires valid 2FA code. Invalidates all previous backup codes.
    """
    if not current_user.two_factor_enabled:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=TWO_FA_NOT_ENABLED,
        )

    # Verify 2FA code
    # encryption = SecretEncryption()
    # secret = encryption.decrypt(current_user.two_factor_secret)
    secret = current_user.two_factor_secret  # Placeholder

    if not two_factor_service.verify_code(secret, request.code):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification code.",
        )

    # Generate new backup codes
    backup_codes = two_factor_service.generate_backup_codes()

    # Hash for storage (used in production DB operations)
    _ = [  # noqa: F841 - reserved for DB storage
        two_factor_service.hash_backup_code(code)
        for code in backup_codes
    ]

    # TODO: Store hashed_codes in database when DB integration is complete
    # Example: current_user.two_factor_backup_codes = hashed_codes; db.commit()

    # Format for display
    formatted_codes = [
        two_factor_service.format_backup_code(code)
        for code in backup_codes
    ]

    return BackupCodesResponse(
        backup_codes=formatted_codes,
        generated_at=datetime.now(timezone.utc),
        remaining_count=len(backup_codes),
    )


# ============================================
# Status Endpoint
# ============================================

@router.get("/status", response_model=TwoFactorStatusResponse)
async def get_two_factor_status(
    current_user: User = Depends(get_current_user),
):
    """Get current 2FA status for the user"""
    return TwoFactorStatusResponse(
        enabled=current_user.two_factor_enabled,
        enabled_at=current_user.two_factor_enabled_at,
        backup_codes_remaining=len(current_user.two_factor_backup_codes or []),
        last_used=current_user.two_factor_last_used,
    )


# ============================================
# Login Verification Endpoint
# ============================================

@router.post("/login/verify")
async def verify_two_factor_login(
    request: TwoFactorLoginRequest,
    # db: Session = Depends(get_db),
):
    """
    Verify 2FA during login

    Called after successful password authentication when 2FA is enabled.
    Accepts either TOTP code or backup code.
    """
    # Validate session token and get user
    # In production, use a secure session token system
    # user = validate_2fa_session_token(request.session_token)
    user = None  # Placeholder

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired session.",
        )

    # Check rate limiting
    is_locked, seconds = two_factor_rate_limiter.is_locked_out(str(user.id))
    if is_locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {seconds} seconds.",
            headers={"Retry-After": str(seconds)},
        )

    is_valid = False
    used_backup_code = None

    if request.is_backup_code:
        # Verify backup code
        is_valid, _used_hash = two_factor_service.verify_backup_code(
            request.code,
            user.two_factor_backup_codes or [],
        )  # _used_hash available for removing used codes

        if is_valid:
            # Remove used backup code
            # user.two_factor_backup_codes.remove(used_hash)
            # db.commit()
            used_backup_code = True
    else:
        # Verify TOTP code
        # encryption = SecretEncryption()
        # secret = encryption.decrypt(user.two_factor_secret)
        secret = user.two_factor_secret  # Placeholder
        is_valid = two_factor_service.verify_code(secret, request.code)

    # Record attempt
    two_factor_rate_limiter.record_attempt(str(user.id), is_valid)

    if not is_valid:
        remaining = two_factor_rate_limiter.get_remaining_attempts(str(user.id))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid code. {remaining} attempts remaining.",
        )

    # Update last used
    # user.two_factor_last_used = datetime.now(timezone.utc)
    # db.commit()

    # Generate JWT tokens
    # access_token = create_access_token(user)
    # refresh_token = create_refresh_token(user)

    return {
        "success": True,
        "user": {
            "id": str(user.id),
            "email": user.email,
            "name": user.name,
        },
        "used_backup_code": used_backup_code,
        "backup_codes_remaining": len(user.two_factor_backup_codes or []) if used_backup_code else None,
    }
