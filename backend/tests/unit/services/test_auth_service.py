"""
Auth Service Unit Tests
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from jose import jwt

from app.services.auth_service import AuthService
from app.models.user import User
from app.core.security import create_access_token, verify_password, hash_password


class TestAuthService:
    """Test AuthService class"""

    @pytest.fixture
    def auth_service(self):
        """Create AuthService instance with mocked dependencies"""
        service = AuthService()
        service.db = AsyncMock()
        service.redis = AsyncMock()
        return service

    @pytest.fixture
    def mock_user(self):
        """Create mock user"""
        return User(
            id="user-123",
            email="test@example.com",
            hashed_password=hash_password("SecurePass123!"),
            name="Test User",
            role="user",
            is_active=True,
            created_at=datetime.now(timezone.utc),
        )


class TestAuthentication(TestAuthService):
    """Test authentication methods"""

    @pytest.mark.asyncio
    async def test_authenticate_valid_credentials(self, auth_service, mock_user):
        """Test successful authentication"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        result = await auth_service.authenticate(
            email="test@example.com",
            password="SecurePass123!"
        )

        assert result is not None
        assert result.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_authenticate_invalid_password(self, auth_service, mock_user):
        """Test authentication with wrong password"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        result = await auth_service.authenticate(
            email="test@example.com",
            password="WrongPassword"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_user_not_found(self, auth_service):
        """Test authentication when user doesn't exist"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=None))
        )

        result = await auth_service.authenticate(
            email="nonexistent@example.com",
            password="password"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_authenticate_inactive_user(self, auth_service, mock_user):
        """Test authentication for inactive user"""
        mock_user.is_active = False
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        result = await auth_service.authenticate(
            email="test@example.com",
            password="SecurePass123!"
        )

        assert result is None


class TestRegistration(TestAuthService):
    """Test user registration"""

    @pytest.mark.asyncio
    async def test_register_new_user(self, auth_service):
        """Test successful user registration"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=None))
        )
        auth_service.db.add = MagicMock()
        auth_service.db.commit = AsyncMock()
        auth_service.db.refresh = AsyncMock()

        result = await auth_service.register(
            email="new@example.com",
            password="SecurePass123!",
            name="New User"
        )

        assert result is not None
        auth_service.db.add.assert_called_once()
        auth_service.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_register_existing_email(self, auth_service, mock_user):
        """Test registration with existing email"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        with pytest.raises(ValueError, match="already exists"):
            await auth_service.register(
                email="test@example.com",
                password="SecurePass123!",
                name="New User"
            )

    @pytest.mark.asyncio
    async def test_register_weak_password(self, auth_service):
        """Test registration with weak password"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=None))
        )

        with pytest.raises(ValueError, match="password"):
            await auth_service.register(
                email="new@example.com",
                password="weak",
                name="New User"
            )


class TestTokenManagement(TestAuthService):
    """Test token generation and validation"""

    @pytest.mark.asyncio
    async def test_create_tokens(self, auth_service, mock_user):
        """Test access and refresh token creation"""
        tokens = await auth_service.create_tokens(mock_user)

        assert "access_token" in tokens
        assert "refresh_token" in tokens
        assert tokens["token_type"] == "bearer"

    @pytest.mark.asyncio
    async def test_verify_valid_token(self, auth_service, mock_user):
        """Test verification of valid token"""
        tokens = await auth_service.create_tokens(mock_user)
        
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        user = await auth_service.verify_token(tokens["access_token"])

        assert user is not None
        assert user.id == mock_user.id

    @pytest.mark.asyncio
    async def test_verify_expired_token(self, auth_service):
        """Test verification of expired token"""
        # Create token that's already expired
        expired_token = jwt.encode(
            {
                "sub": "user-123",
                "exp": datetime.now(timezone.utc) - timedelta(hours=1)
            },
            "secret",
            algorithm="HS256"
        )

        with pytest.raises(Exception):
            await auth_service.verify_token(expired_token)

    @pytest.mark.asyncio
    async def test_verify_invalid_token(self, auth_service):
        """Test verification of invalid token"""
        with pytest.raises(Exception):
            await auth_service.verify_token("invalid-token")

    @pytest.mark.asyncio
    async def test_refresh_token(self, auth_service, mock_user):
        """Test token refresh"""
        tokens = await auth_service.create_tokens(mock_user)
        
        auth_service.redis.get = AsyncMock(return_value=mock_user.id)
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        new_tokens = await auth_service.refresh_tokens(tokens["refresh_token"])

        assert "access_token" in new_tokens
        assert new_tokens["access_token"] != tokens["access_token"]

    @pytest.mark.asyncio
    async def test_revoke_token(self, auth_service, mock_user):
        """Test token revocation"""
        tokens = await auth_service.create_tokens(mock_user)
        
        auth_service.redis.set = AsyncMock()

        await auth_service.revoke_token(tokens["access_token"])

        auth_service.redis.set.assert_called_once()


class TestPasswordManagement(TestAuthService):
    """Test password management"""

    @pytest.mark.asyncio
    async def test_change_password(self, auth_service, mock_user):
        """Test password change"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )
        auth_service.db.commit = AsyncMock()

        result = await auth_service.change_password(
            user_id=mock_user.id,
            current_password="SecurePass123!",
            new_password="NewSecurePass456!"
        )

        assert result is True
        auth_service.db.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_change_password_wrong_current(self, auth_service, mock_user):
        """Test password change with wrong current password"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        with pytest.raises(ValueError, match="incorrect"):
            await auth_service.change_password(
                user_id=mock_user.id,
                current_password="WrongPassword",
                new_password="NewSecurePass456!"
            )

    @pytest.mark.asyncio
    async def test_request_password_reset(self, auth_service, mock_user):
        """Test password reset request"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )
        auth_service.redis.set = AsyncMock()

        token = await auth_service.request_password_reset(mock_user.email)

        assert token is not None
        auth_service.redis.set.assert_called_once()

    @pytest.mark.asyncio
    async def test_reset_password_with_token(self, auth_service, mock_user):
        """Test password reset with valid token"""
        auth_service.redis.get = AsyncMock(return_value=mock_user.id)
        auth_service.redis.delete = AsyncMock()
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )
        auth_service.db.commit = AsyncMock()

        result = await auth_service.reset_password(
            token="valid-reset-token",
            new_password="NewSecurePass789!"
        )

        assert result is True
        auth_service.redis.delete.assert_called_once()


class TestTwoFactorAuth(TestAuthService):
    """Test 2FA functionality"""

    @pytest.mark.asyncio
    async def test_setup_2fa(self, auth_service, mock_user):
        """Test 2FA setup"""
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        result = await auth_service.setup_2fa(mock_user.id)

        assert "secret" in result
        assert "qr_code" in result

    @pytest.mark.asyncio
    async def test_verify_2fa_code(self, auth_service, mock_user):
        """Test 2FA code verification"""
        mock_user.two_factor_secret = "JBSWY3DPEHPK3PXP"
        
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        # Use mock for pyotp verification
        with patch("pyotp.TOTP") as mock_totp:
            mock_totp.return_value.verify.return_value = True
            
            result = await auth_service.verify_2fa(mock_user.id, "123456")
            
            assert result is True

    @pytest.mark.asyncio
    async def test_verify_invalid_2fa_code(self, auth_service, mock_user):
        """Test invalid 2FA code"""
        mock_user.two_factor_secret = "JBSWY3DPEHPK3PXP"
        
        auth_service.db.execute = AsyncMock(
            return_value=MagicMock(scalar=MagicMock(return_value=mock_user))
        )

        with patch("pyotp.TOTP") as mock_totp:
            mock_totp.return_value.verify.return_value = False
            
            result = await auth_service.verify_2fa(mock_user.id, "000000")
            
            assert result is False


class TestSessionManagement(TestAuthService):
    """Test session management"""

    @pytest.mark.asyncio
    async def test_create_session(self, auth_service, mock_user):
        """Test session creation"""
        auth_service.redis.set = AsyncMock()

        session_id = await auth_service.create_session(
            user_id=mock_user.id,
            device_info={"browser": "Chrome", "os": "Windows"}
        )

        assert session_id is not None
        auth_service.redis.set.assert_called()

    @pytest.mark.asyncio
    async def test_get_active_sessions(self, auth_service, mock_user):
        """Test getting active sessions"""
        auth_service.redis.keys = AsyncMock(return_value=[
            f"session:{mock_user.id}:session1",
            f"session:{mock_user.id}:session2"
        ])
        auth_service.redis.get = AsyncMock(return_value='{"device": "Chrome"}')

        sessions = await auth_service.get_active_sessions(mock_user.id)

        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_revoke_session(self, auth_service, mock_user):
        """Test session revocation"""
        auth_service.redis.delete = AsyncMock()

        await auth_service.revoke_session(mock_user.id, "session-123")

        auth_service.redis.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_revoke_all_sessions(self, auth_service, mock_user):
        """Test revoking all sessions"""
        auth_service.redis.keys = AsyncMock(return_value=[
            f"session:{mock_user.id}:session1",
            f"session:{mock_user.id}:session2"
        ])
        auth_service.redis.delete = AsyncMock()

        await auth_service.revoke_all_sessions(mock_user.id)

        assert auth_service.redis.delete.call_count == 2
