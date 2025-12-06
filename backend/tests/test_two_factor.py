"""
Two-Factor Authentication Service Tests

Tests:
- TOTP generation and verification
- QR code generation
- Backup code management
- Rate limiting
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from app.services.two_factor import (
    TwoFactorService,
    TwoFactorRateLimiter,
    two_factor_service,
    two_factor_rate_limiter,
)


class TestTwoFactorService:
    """Test TOTP functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_generate_secret(self):
        """Test secret generation"""
        secret = self.service.generate_secret()
        
        assert secret is not None
        assert len(secret) == 32  # Base32 encoded
        assert secret.isalnum()

    def test_generate_unique_secrets(self):
        """Test that each secret is unique"""
        secrets = [self.service.generate_secret() for _ in range(100)]
        unique_secrets = set(secrets)
        
        assert len(unique_secrets) == 100

    def test_get_totp(self):
        """Test TOTP instance creation"""
        secret = self.service.generate_secret()
        totp = self.service.get_totp(secret)
        
        assert totp is not None
        assert totp.digits == 6
        assert totp.interval == 30

    def test_generate_provisioning_uri(self):
        """Test provisioning URI generation"""
        secret = "JBSWY3DPEHPK3PXP"
        email = "test@example.com"
        
        uri = self.service.generate_provisioning_uri(secret, email)
        
        assert uri.startswith("otpauth://totp/")
        assert "AI%20Code%20Review" in uri
        assert "test%40example.com" in uri
        assert secret in uri

    def test_generate_qr_code(self):
        """Test QR code generation"""
        secret = "JBSWY3DPEHPK3PXP"
        email = "test@example.com"
        
        qr_code = self.service.generate_qr_code(secret, email)
        
        assert qr_code.startswith("data:image/png;base64,")
        assert len(qr_code) > 100  # Should have actual content

    def test_verify_valid_code(self):
        """Test verification of valid code"""
        secret = self.service.generate_secret()
        
        # Get current code
        current_code = self.service.get_current_code(secret)
        
        # Verify it
        assert self.service.verify_code(secret, current_code) is True

    def test_verify_invalid_code(self):
        """Test verification of invalid code"""
        secret = self.service.generate_secret()
        
        # Wrong code
        assert self.service.verify_code(secret, "000000") is False

    def test_verify_code_with_spaces(self):
        """Test verification handles spaces"""
        secret = self.service.generate_secret()
        current_code = self.service.get_current_code(secret)
        
        # Add spaces
        spaced_code = f"{current_code[:3]} {current_code[3:]}"
        
        assert self.service.verify_code(secret, spaced_code) is True

    def test_verify_code_with_dashes(self):
        """Test verification handles dashes"""
        secret = self.service.generate_secret()
        current_code = self.service.get_current_code(secret)
        
        # Add dash
        dashed_code = f"{current_code[:3]}-{current_code[3:]}"
        
        assert self.service.verify_code(secret, dashed_code) is True

    def test_verify_empty_code(self):
        """Test empty code is rejected"""
        secret = self.service.generate_secret()
        
        assert self.service.verify_code(secret, "") is False
        assert self.service.verify_code(secret, None) is False

    def test_verify_non_numeric_code(self):
        """Test non-numeric code is rejected"""
        secret = self.service.generate_secret()
        
        assert self.service.verify_code(secret, "abcdef") is False

    def test_verify_wrong_length_code(self):
        """Test wrong length code is rejected"""
        secret = self.service.generate_secret()
        
        assert self.service.verify_code(secret, "12345") is False
        assert self.service.verify_code(secret, "1234567") is False


class TestBackupCodes:
    """Test backup code functionality"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_generate_backup_codes(self):
        """Test backup code generation"""
        codes = self.service.generate_backup_codes()
        
        assert len(codes) == 10
        for code in codes:
            assert len(code) == 8
            assert code.isalnum()

    def test_backup_codes_unique(self):
        """Test all backup codes are unique"""
        codes = self.service.generate_backup_codes()
        unique_codes = set(codes)
        
        assert len(unique_codes) == 10

    def test_hash_backup_code(self):
        """Test backup code hashing"""
        code = "ABCD1234"
        hashed = self.service.hash_backup_code(code)
        
        assert hashed is not None
        assert len(hashed) == 64  # SHA256 hex
        assert hashed != code

    def test_hash_backup_code_normalized(self):
        """Test hashing normalizes code"""
        code1 = "abcd1234"
        code2 = "ABCD1234"
        code3 = "ABCD-1234"
        code4 = "ABCD 1234"
        
        assert self.service.hash_backup_code(code1) == self.service.hash_backup_code(code2)
        assert self.service.hash_backup_code(code1) == self.service.hash_backup_code(code3)
        assert self.service.hash_backup_code(code1) == self.service.hash_backup_code(code4)

    def test_verify_backup_code_valid(self):
        """Test verification of valid backup code"""
        codes = self.service.generate_backup_codes()
        hashed_codes = [self.service.hash_backup_code(c) for c in codes]
        
        # Verify first code
        is_valid, matched_hash = self.service.verify_backup_code(codes[0], hashed_codes)
        
        assert is_valid is True
        assert matched_hash == hashed_codes[0]

    def test_verify_backup_code_invalid(self):
        """Test verification of invalid backup code"""
        codes = self.service.generate_backup_codes()
        hashed_codes = [self.service.hash_backup_code(c) for c in codes]
        
        # Verify wrong code
        is_valid, matched_hash = self.service.verify_backup_code("WRONGCODE", hashed_codes)
        
        assert is_valid is False
        assert matched_hash is None

    def test_format_backup_code(self):
        """Test backup code formatting"""
        code = "ABCD1234"
        formatted = self.service.format_backup_code(code)
        
        assert formatted == "ABCD-1234"

    def test_format_backup_code_lowercase(self):
        """Test formatting handles lowercase"""
        code = "abcd1234"
        formatted = self.service.format_backup_code(code)
        
        assert formatted == "ABCD-1234"


class TestSecretFormatting:
    """Test secret formatting"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_format_secret_for_manual_entry(self):
        """Test secret is formatted for easy manual entry"""
        secret = "JBSWY3DPEHPK3PXPJBSWY3DPEHPK3PXP"
        formatted = self.service.format_secret_for_manual_entry(secret)
        
        assert " " in formatted
        parts = formatted.split(" ")
        assert all(len(p) == 4 for p in parts[:-1])


class TestRateLimiter:
    """Test rate limiting for 2FA attempts"""

    def setup_method(self):
        """Setup fresh rate limiter for each test"""
        self.limiter = TwoFactorRateLimiter()
        self.user_id = "test-user-123"

    def test_initial_not_locked(self):
        """Test user starts not locked out"""
        is_locked, seconds = self.limiter.is_locked_out(self.user_id)
        
        assert is_locked is False
        assert seconds is None

    def test_remaining_attempts_initial(self):
        """Test initial remaining attempts"""
        remaining = self.limiter.get_remaining_attempts(self.user_id)
        
        assert remaining == 5

    def test_record_failed_attempt(self):
        """Test recording failed attempts"""
        self.limiter.record_attempt(self.user_id, success=False)
        
        remaining = self.limiter.get_remaining_attempts(self.user_id)
        assert remaining == 4

    def test_record_success_clears_attempts(self):
        """Test successful attempt clears counter"""
        # Make some failed attempts
        for _ in range(3):
            self.limiter.record_attempt(self.user_id, success=False)
        
        assert self.limiter.get_remaining_attempts(self.user_id) == 2
        
        # Success clears attempts
        self.limiter.record_attempt(self.user_id, success=True)
        
        assert self.limiter.get_remaining_attempts(self.user_id) == 5

    def test_lockout_after_max_attempts(self):
        """Test user is locked after max attempts"""
        # Make max failed attempts
        for _ in range(5):
            self.limiter.record_attempt(self.user_id, success=False)
        
        is_locked, seconds = self.limiter.is_locked_out(self.user_id)
        
        assert is_locked is True
        assert seconds is not None
        assert seconds > 0

    def test_lockout_duration(self):
        """Test lockout duration is correct"""
        for _ in range(5):
            self.limiter.record_attempt(self.user_id, success=False)
        
        _, seconds = self.limiter.is_locked_out(self.user_id)
        
        # Should be close to 15 minutes
        assert 14 * 60 <= seconds <= 15 * 60

    def test_different_users_independent(self):
        """Test rate limits are per-user"""
        user1 = "user-1"
        user2 = "user-2"
        
        # Lock out user 1
        for _ in range(5):
            self.limiter.record_attempt(user1, success=False)
        
        # User 2 should not be affected
        is_locked, _ = self.limiter.is_locked_out(user2)
        assert is_locked is False
        assert self.limiter.get_remaining_attempts(user2) == 5


class TestTwoFactorFlow:
    """Integration tests for complete 2FA flows"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_complete_setup_flow(self):
        """Test complete 2FA setup flow"""
        email = "user@example.com"
        
        # 1. Generate secret
        secret = self.service.generate_secret()
        assert secret is not None
        
        # 2. Generate QR code
        qr_code = self.service.generate_qr_code(secret, email)
        assert qr_code.startswith("data:image/png;base64,")
        
        # 3. User scans QR, gets code from authenticator
        code = self.service.get_current_code(secret)
        
        # 4. Verify the code
        assert self.service.verify_code(secret, code) is True
        
        # 5. Generate backup codes
        backup_codes = self.service.generate_backup_codes()
        assert len(backup_codes) == 10
        
        # Setup complete!

    def test_complete_login_flow_with_totp(self):
        """Test complete login flow with TOTP"""
        # Setup: User has 2FA enabled with this secret
        secret = self.service.generate_secret()
        
        # Login: User enters email/password (passed elsewhere)
        # Then enters TOTP code
        code = self.service.get_current_code(secret)
        
        # Verify TOTP
        assert self.service.verify_code(secret, code) is True

    def test_complete_login_flow_with_backup_code(self):
        """Test complete login flow with backup code"""
        # Setup: User has 2FA enabled with backup codes
        backup_codes = self.service.generate_backup_codes()
        hashed_codes = [self.service.hash_backup_code(c) for c in backup_codes]
        
        # Login: User lost their phone, uses backup code
        is_valid, matched_hash = self.service.verify_backup_code(
            backup_codes[0], hashed_codes
        )
        
        assert is_valid is True
        
        # Remove used backup code
        hashed_codes.remove(matched_hash)
        assert len(hashed_codes) == 9
        
        # Same backup code no longer works
        is_valid, _ = self.service.verify_backup_code(backup_codes[0], hashed_codes)
        assert is_valid is False

    def test_disable_flow(self):
        """Test 2FA disable flow"""
        secret = self.service.generate_secret()
        
        # User provides current TOTP code to disable
        code = self.service.get_current_code(secret)
        
        # Verify code before disabling
        assert self.service.verify_code(secret, code) is True
        
        # Disable: Clear secret and backup codes (done in API)
        # After disable, old secret should not be used


class TestEdgeCases:
    """Test edge cases and error handling"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_verify_code_timing_window(self):
        """Test code verification with time window"""
        secret = self.service.generate_secret()
        code = self.service.get_current_code(secret)
        
        # Code should be valid (within window)
        assert self.service.verify_code(secret, code) is True

    def test_empty_backup_codes_list(self):
        """Test verification against empty backup codes list"""
        is_valid, _ = self.service.verify_backup_code("ABCD1234", [])
        assert is_valid is False

    def test_malformed_secret(self):
        """Test handling of malformed secret"""
        # This would raise an error in pyotp
        with pytest.raises(Exception):
            self.service.verify_code("invalid-secret!", "123456")

    def test_unicode_in_email(self):
        """Test QR code generation with unicode email"""
        secret = self.service.generate_secret()
        email = "用户@example.com"
        
        # Should not raise
        qr_code = self.service.generate_qr_code(secret, email)
        assert qr_code is not None


# Performance tests
class TestPerformance:
    """Performance tests for 2FA operations"""

    def setup_method(self):
        """Setup for each test"""
        self.service = TwoFactorService()

    def test_secret_generation_performance(self):
        """Test secret generation is fast"""
        import time
        
        start = time.time()
        for _ in range(1000):
            self.service.generate_secret()
        elapsed = time.time() - start
        
        # Should generate 1000 secrets in under 1 second
        assert elapsed < 1.0

    def test_code_verification_performance(self):
        """Test code verification is fast"""
        import time
        
        secret = self.service.generate_secret()
        code = self.service.get_current_code(secret)
        
        start = time.time()
        for _ in range(1000):
            self.service.verify_code(secret, code)
        elapsed = time.time() - start
        
        # Should verify 1000 codes in under 1 second
        assert elapsed < 1.0
