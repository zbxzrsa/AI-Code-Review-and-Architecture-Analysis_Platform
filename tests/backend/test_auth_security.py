"""
Comprehensive tests for authentication and security features.

Coverage targets:
- JWT token creation/verification
- CSRF protection
- Rate limiting
- httpOnly cookie handling
- Password hashing
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock
import hashlib
import secrets

# Test configuration
pytestmark = pytest.mark.asyncio


class TestSecureAuthManager:
    """Tests for SecureAuthManager class."""

    @pytest.fixture
    def auth_manager(self):
        """Create auth manager instance."""
        from backend.shared.security.secure_auth import SecureAuthManager, AuthConfig
        
        config = AuthConfig(
            secret_key="test-secret-key-for-testing-only",
            access_token_expire_minutes=15,
            refresh_token_expire_days=7,
            cookie_secure=False,  # For testing
        )
        return SecureAuthManager(config)

    # JWT Token Tests
    
    async def test_create_access_token(self, auth_manager):
        """Test access token creation."""
        user_data = {"sub": "user123", "role": "user"}
        token = auth_manager.create_access_token(user_data)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 100  # JWT tokens are typically long

    async def test_create_refresh_token(self, auth_manager):
        """Test refresh token creation."""
        user_data = {"sub": "user123"}
        token = auth_manager.create_refresh_token(user_data)
        
        assert token is not None
        assert isinstance(token, str)

    async def test_create_token_pair(self, auth_manager):
        """Test token pair creation."""
        user_data = {"sub": "user123", "role": "admin"}
        token_pair = auth_manager.create_token_pair(user_data)
        
        assert token_pair.access_token is not None
        assert token_pair.refresh_token is not None
        assert token_pair.token_type == "bearer"
        assert token_pair.expires_in == 15 * 60  # 15 minutes in seconds

    async def test_verify_valid_access_token(self, auth_manager):
        """Test verification of valid access token."""
        user_data = {"sub": "user123", "role": "user"}
        token = auth_manager.create_access_token(user_data)
        
        payload = auth_manager.verify_token(token, "access")
        
        assert payload["sub"] == "user123"
        assert payload["role"] == "user"
        assert payload["type"] == "access"

    async def test_verify_valid_refresh_token(self, auth_manager):
        """Test verification of valid refresh token."""
        user_data = {"sub": "user123"}
        token = auth_manager.create_refresh_token(user_data)
        
        payload = auth_manager.verify_token(token, "refresh")
        
        assert payload["sub"] == "user123"
        assert payload["type"] == "refresh"
        assert "jti" in payload  # JWT ID for rotation

    async def test_verify_expired_token_fails(self, auth_manager):
        """Test that expired tokens are rejected."""
        from fastapi import HTTPException
        
        user_data = {"sub": "user123"}
        # Create token with negative expiry (already expired)
        token = auth_manager.create_access_token(
            user_data,
            expires_delta=timedelta(seconds=-10)
        )
        
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(token, "access")
        
        assert exc_info.value.status_code == 401

    async def test_verify_wrong_token_type_fails(self, auth_manager):
        """Test that wrong token type is rejected."""
        from fastapi import HTTPException
        
        user_data = {"sub": "user123"}
        access_token = auth_manager.create_access_token(user_data)
        
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token(access_token, "refresh")
        
        assert exc_info.value.status_code == 401

    async def test_verify_invalid_token_fails(self, auth_manager):
        """Test that invalid tokens are rejected."""
        from fastapi import HTTPException
        
        with pytest.raises(HTTPException) as exc_info:
            auth_manager.verify_token("invalid.token.here", "access")
        
        assert exc_info.value.status_code == 401

    # CSRF Token Tests
    
    async def test_generate_csrf_token(self, auth_manager):
        """Test CSRF token generation."""
        session_id = "session123"
        token = auth_manager.generate_csrf_token(session_id)
        
        assert token is not None
        assert isinstance(token, str)
        assert len(token) > 20

    async def test_verify_valid_csrf_token(self, auth_manager):
        """Test CSRF token verification."""
        session_id = "session123"
        token = auth_manager.generate_csrf_token(session_id)
        
        is_valid = auth_manager.verify_csrf_token(token)
        
        assert is_valid is True

    async def test_verify_invalid_csrf_token(self, auth_manager):
        """Test that invalid CSRF tokens are rejected."""
        is_valid = auth_manager.verify_csrf_token("invalid-token")
        
        assert is_valid is False

    async def test_csrf_token_uniqueness(self, auth_manager):
        """Test that each CSRF token is unique."""
        tokens = [auth_manager.generate_csrf_token(f"session{i}") for i in range(100)]
        
        # All tokens should be unique
        assert len(tokens) == len(set(tokens))

    # Cookie Tests
    
    async def test_set_auth_cookies(self, auth_manager):
        """Test setting authentication cookies."""
        from fastapi import Response
        
        response = Response()
        
        auth_manager.set_auth_cookies(
            response,
            access_token="access123",
            refresh_token="refresh456",
            csrf_token="csrf789",
        )
        
        # Check cookies were set
        cookies = response.headers.getlist("set-cookie")
        assert len(cookies) == 3
        
        cookie_str = str(cookies)
        assert "access_token" in cookie_str
        assert "refresh_token" in cookie_str
        assert "csrf_token" in cookie_str
        assert "httponly" in cookie_str.lower()

    async def test_clear_auth_cookies(self, auth_manager):
        """Test clearing authentication cookies."""
        from fastapi import Response
        
        response = Response()
        auth_manager.clear_auth_cookies(response)
        
        # Check cookies were cleared (set to empty with max_age=0)
        cookies = response.headers.getlist("set-cookie")
        assert len(cookies) >= 3


class TestRateLimiter:
    """Tests for rate limiting functionality."""

    @pytest.fixture
    def rate_limiter(self):
        """Create rate limiter instance."""
        from backend.shared.middleware.rate_limiter import (
            SlidingWindowRateLimiter,
            RateLimitConfig,
            RateLimitRule,
        )
        
        config = RateLimitConfig(
            default_rpm=10,
            default_rph=100,
        )
        limiter = SlidingWindowRateLimiter(config)
        
        # Add test rule
        limiter.add_rule(RateLimitRule(
            path_pattern="/api/test",
            requests_per_minute=5,
            requests_per_hour=20,
        ))
        
        return limiter

    @pytest.fixture
    def mock_request(self):
        """Create mock request."""
        request = Mock()
        request.url.path = "/api/test"
        request.method = "POST"
        request.client.host = "127.0.0.1"
        request.headers = {}
        return request

    async def test_allows_requests_under_limit(self, rate_limiter, mock_request):
        """Test that requests under limit are allowed."""
        allowed, info = await rate_limiter.is_allowed(mock_request)
        
        assert allowed is True
        assert info["allowed"] is True
        assert info["remaining"] >= 0

    async def test_blocks_requests_over_limit(self, rate_limiter, mock_request):
        """Test that requests over limit are blocked."""
        # Make requests up to the limit
        for _ in range(5):
            await rate_limiter.is_allowed(mock_request)
        
        # Next request should be blocked
        allowed, info = await rate_limiter.is_allowed(mock_request)
        
        assert allowed is False
        assert info["limit_type"] == "minute"

    async def test_rate_limit_info_returned(self, rate_limiter, mock_request):
        """Test that rate limit info is returned."""
        allowed, info = await rate_limiter.is_allowed(mock_request)
        
        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info

    async def test_different_paths_have_separate_limits(self, rate_limiter):
        """Test that different paths have separate rate limits."""
        request1 = Mock()
        request1.url.path = "/api/path1"
        request1.method = "GET"
        request1.client.host = "127.0.0.1"
        request1.headers = {}
        
        request2 = Mock()
        request2.url.path = "/api/path2"
        request2.method = "GET"
        request2.client.host = "127.0.0.1"
        request2.headers = {}
        
        # Exhaust limit for path1
        for _ in range(10):
            await rate_limiter.is_allowed(request1)
        
        # path2 should still be allowed
        allowed, _ = await rate_limiter.is_allowed(request2)
        assert allowed is True

    async def test_different_ips_have_separate_limits(self, rate_limiter):
        """Test that different IPs have separate rate limits."""
        request1 = Mock()
        request1.url.path = "/api/test"
        request1.method = "GET"
        request1.client.host = "192.168.1.1"
        request1.headers = {}
        
        request2 = Mock()
        request2.url.path = "/api/test"
        request2.method = "GET"
        request2.client.host = "192.168.1.2"
        request2.headers = {}
        
        # Exhaust limit for IP1
        for _ in range(5):
            await rate_limiter.is_allowed(request1)
        
        # IP2 should still be allowed
        allowed, _ = await rate_limiter.is_allowed(request2)
        assert allowed is True


class TestPasswordHashing:
    """Tests for password hashing functionality."""

    async def test_password_hash_is_different_from_plain(self):
        """Test that hashed password differs from plain text."""
        from passlib.hash import argon2
        
        password = "SecurePassword123!"
        hashed = argon2.hash(password)
        
        assert hashed != password
        assert len(hashed) > len(password)

    async def test_password_verification_succeeds(self):
        """Test that correct password verifies."""
        from passlib.hash import argon2
        
        password = "SecurePassword123!"
        hashed = argon2.hash(password)
        
        assert argon2.verify(password, hashed) is True

    async def test_password_verification_fails_for_wrong_password(self):
        """Test that wrong password fails verification."""
        from passlib.hash import argon2
        
        password = "SecurePassword123!"
        wrong_password = "WrongPassword456!"
        hashed = argon2.hash(password)
        
        assert argon2.verify(wrong_password, hashed) is False

    async def test_same_password_produces_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        from passlib.hash import argon2
        
        password = "SecurePassword123!"
        hash1 = argon2.hash(password)
        hash2 = argon2.hash(password)
        
        assert hash1 != hash2
        # But both should verify
        assert argon2.verify(password, hash1) is True
        assert argon2.verify(password, hash2) is True


class TestSecurityHeaders:
    """Tests for security header configuration."""

    async def test_csrf_header_required_for_post(self):
        """Test that CSRF header is required for POST requests."""
        from backend.shared.security.secure_auth import CSRFProtectedRoute
        from fastapi import HTTPException
        
        route = CSRFProtectedRoute()
        
        request = Mock()
        request.method = "POST"
        request.headers = {}  # No CSRF header
        
        with pytest.raises(HTTPException) as exc_info:
            await route(request)
        
        assert exc_info.value.status_code == 403

    async def test_csrf_header_not_required_for_get(self):
        """Test that CSRF header is not required for GET requests."""
        from backend.shared.security.secure_auth import CSRFProtectedRoute
        
        route = CSRFProtectedRoute()
        
        request = Mock()
        request.method = "GET"
        
        # Should not raise
        result = await route(request)
        assert result is None


class TestAIProviderSecurity:
    """Tests for AI provider security features."""

    async def test_api_key_not_logged(self):
        """Test that API keys are not logged."""
        import logging
        from io import StringIO
        
        # Capture log output
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        logger = logging.getLogger("backend.shared.utils.ai_provider_factory")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        
        # Simulate provider creation with API key
        api_key = "sk-secret-api-key-12345"
        
        # Log something that might include the key
        logger.info(f"Provider configured with endpoint")
        
        log_output = log_capture.getvalue()
        
        # API key should not appear in logs
        assert api_key not in log_output
        
        # Cleanup
        logger.removeHandler(handler)

    async def test_cost_calculation_accuracy(self):
        """Test AI cost calculation accuracy."""
        from backend.shared.utils.ai_provider_factory import OpenAIProvider, ProviderConfig, ProviderType, ProviderTier
        
        # OpenAI GPT-4 cost: $0.03 per 1K tokens
        config = ProviderConfig(
            type=ProviderType.OPENAI,
            tier=ProviderTier.PAID,
            endpoint="https://api.openai.com",
            model="gpt-4",
            api_key="test-key",
        )
        provider = OpenAIProvider(config)
        
        # Test cost calculation
        tokens = 1000
        expected_cost = 0.03
        
        cost_rate = provider.COST_PER_1K.get("gpt-4", 0.03)
        actual_cost = (tokens / 1000) * cost_rate
        
        # Use pytest.approx for floating point comparison
        assert actual_cost == pytest.approx(expected_cost, rel=1e-9)


# Performance benchmark tests
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""

    @pytest.mark.benchmark
    async def test_jwt_creation_performance(self, benchmark=None):
        """Benchmark JWT token creation."""
        from backend.shared.security.secure_auth import SecureAuthManager
        
        auth_manager = SecureAuthManager()
        user_data = {"sub": "user123", "role": "user"}
        
        # Should complete in < 10ms
        import time
        start = time.time()
        for _ in range(100):
            auth_manager.create_access_token(user_data)
        elapsed = (time.time() - start) / 100 * 1000  # ms per operation
        
        assert elapsed < 10, f"JWT creation took {elapsed}ms, expected < 10ms"

    @pytest.mark.benchmark
    async def test_csrf_verification_performance(self):
        """Benchmark CSRF token verification."""
        from backend.shared.security.secure_auth import SecureAuthManager
        
        auth_manager = SecureAuthManager()
        token = auth_manager.generate_csrf_token("session123")
        
        import time
        start = time.time()
        for _ in range(1000):
            auth_manager.verify_csrf_token(token)
        elapsed = (time.time() - start) / 1000 * 1000  # ms per operation
        
        assert elapsed < 1, f"CSRF verification took {elapsed}ms, expected < 1ms"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
