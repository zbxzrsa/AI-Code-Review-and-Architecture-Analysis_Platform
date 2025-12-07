"""Tests for Authentication_V1"""

import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auth_manager import AuthManager
from src.session_manager import SessionManager
from src.token_service import TokenService


class TestAuthManager:
    @pytest.fixture
    def auth(self):
        return AuthManager()

    @pytest.mark.asyncio
    async def test_register(self, auth):
        result = await auth.register("test@example.com", "password123")
        assert result.success
        assert result.user is not None
        assert result.access_token is not None

    @pytest.mark.asyncio
    async def test_login(self, auth):
        await auth.register("test@example.com", "password123")
        result = await auth.login("test@example.com", "password123")
        assert result.success

    @pytest.mark.asyncio
    async def test_invalid_login(self, auth):
        result = await auth.login("nonexistent@example.com", "wrong")
        assert not result.success


class TestSessionManager:
    @pytest.fixture
    def sessions(self):
        return SessionManager()

    @pytest.mark.asyncio
    async def test_create_session(self, sessions):
        session = await sessions.create_session("user-123")
        assert session.user_id == "user-123"
        assert not session.is_expired()

    @pytest.mark.asyncio
    async def test_invalidate_session(self, sessions):
        session = await sessions.create_session("user-123")
        result = await sessions.invalidate_session(session.session_id)
        assert result


class TestTokenService:
    @pytest.fixture
    def tokens(self):
        return TokenService()

    def test_create_and_verify(self, tokens):
        token = tokens.create_access_token("user-123", "user")
        payload = tokens.verify_token(token)
        assert payload is not None
        assert payload.sub == "user-123"

    def test_invalid_token(self, tokens):
        payload = tokens.verify_token("invalid-token")
        assert payload is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
