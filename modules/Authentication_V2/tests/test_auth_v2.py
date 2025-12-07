"""Tests for Authentication_V2"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth_manager import AuthManager
from src.mfa_service import MFAService
from src.session_manager import SessionManager, DeviceInfo
from src.token_service import TokenService


class TestAuthManagerV2:
    @pytest.fixture
    def auth(self):
        return AuthManager()

    @pytest.mark.asyncio
    async def test_register_with_mfa(self, auth):
        result = await auth.register("test@example.com", "password", enable_mfa=True)
        assert result.success
        assert result.user.mfa_enabled

    @pytest.mark.asyncio
    async def test_login_requires_mfa(self, auth):
        await auth.register("mfa@example.com", "password", enable_mfa=True)
        result = await auth.login("mfa@example.com", "password")

        assert not result.success
        assert result.requires_mfa
        assert result.mfa_token is not None

    @pytest.mark.asyncio
    async def test_rate_limiting(self, auth):
        # Register user
        await auth.register("rate@example.com", "password")

        # Exceed rate limit
        for _ in range(10):
            await auth.login("rate@example.com", "wrong")

        result = await auth.login("rate@example.com", "password")
        assert not result.success
        assert "Rate limit" in result.error or "locked" in result.error.lower()


class TestMFAService:
    @pytest.fixture
    def mfa(self):
        return MFAService()

    def test_setup_mfa(self, mfa):
        setup = mfa.setup_mfa("user-123", "user@example.com")

        assert setup.secret
        assert setup.qr_uri.startswith("otpauth://")
        assert len(setup.backup_codes) == 10

    def test_backup_code_verification(self, mfa):
        setup = mfa.setup_mfa("user-123", "user@example.com")
        code = setup.backup_codes[0]

        assert mfa.verify_backup_code("user-123", code)
        # Code should be consumed
        assert not mfa.verify_backup_code("user-123", code)


class TestSessionManagerV2:
    @pytest.fixture
    def sessions(self):
        return SessionManager()

    @pytest.mark.asyncio
    async def test_create_session_with_device(self, sessions):
        device = DeviceInfo(
            device_id="dev-123",
            device_type="web",
            user_agent="Mozilla/5.0",
            browser="Chrome",
        )

        session = await sessions.create_session("user-123", device_info=device)

        assert session.device.device_id == "dev-123"

    @pytest.mark.asyncio
    async def test_session_limits(self, sessions):
        sessions.max_sessions_per_user = 2

        s1 = await sessions.create_session("user-123")
        s2 = await sessions.create_session("user-123")
        s3 = await sessions.create_session("user-123")

        # First session should be invalidated
        assert await sessions.get_session(s1.session_id) is None
        assert await sessions.get_session(s3.session_id) is not None

    @pytest.mark.asyncio
    async def test_trust_device(self, sessions):
        device = DeviceInfo(device_id="trusted-dev", device_type="web")

        session = await sessions.create_session("user-123", device_info=device, trust_device=True)

        assert session.is_trusted


class TestTokenServiceV2:
    @pytest.fixture
    def tokens(self):
        return TokenService()

    def test_create_token_pair(self, tokens):
        pair = tokens.create_token_pair("user-123", "user")

        assert "access_token" in pair
        assert "refresh_token" in pair
        assert pair["token_type"] == "Bearer"

    def test_verify_access_token(self, tokens):
        pair = tokens.create_token_pair("user-123", "user")

        payload = tokens.verify_token(pair["access_token"], expected_type="access")

        assert payload is not None
        assert payload.sub == "user-123"

    def test_refresh_rotation(self, tokens):
        pair = tokens.create_token_pair("user-123", "user")

        new_pair = tokens.refresh_tokens(pair["refresh_token"])

        assert new_pair is not None
        # Old refresh token should be revoked
        assert tokens.verify_token(pair["refresh_token"]) is None

    def test_revoke_token(self, tokens):
        pair = tokens.create_token_pair("user-123", "user")

        tokens.revoke_token(pair["access_token"])

        assert tokens.verify_token(pair["access_token"]) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
