#!/usr/bin/env python3
"""
Production Environment Validation Script
生产环境验证脚本

Run before deploying to production to ensure all security requirements are met.
在部署到生产环境之前运行，确保满足所有安全要求。

Usage: python scripts/validate-production.py
"""

import os
import sys
import re
from typing import List, Tuple
from pathlib import Path

# Color codes for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def print_header(text: str):
    """Print section header."""
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}{text}{RESET}")
    print(f"{BLUE}{'='*60}{RESET}")


def print_success(text: str):
    """Print success message."""
    print(f"{GREEN}✓ {text}{RESET}")


def print_error(text: str):
    """Print error message."""
    print(f"{RED}✗ {text}{RESET}")


def print_warning(text: str):
    """Print warning message."""
    print(f"{YELLOW}⚠ {text}{RESET}")


class ProductionValidator:
    """Validate production environment configuration."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def validate_environment(self) -> bool:
        """Validate ENVIRONMENT is set to production."""
        env = os.getenv("ENVIRONMENT", "")
        if env != "production":
            self.warnings.append(f"ENVIRONMENT is '{env}', not 'production'")
            # Return False if completely missing, True if just different
            return bool(env)  # Allow non-empty environment values
        self.passed.append("ENVIRONMENT=production")
        return True

    def validate_debug_disabled(self) -> bool:
        """Validate DEBUG is disabled."""
        debug = os.getenv("DEBUG", "false").lower()
        if debug == "true":
            self.errors.append("DEBUG must be 'false' in production")
            return False
        self.passed.append("DEBUG is disabled")
        return True

    def validate_jwt_secret(self) -> bool:
        """Validate JWT_SECRET_KEY is secure."""
        secret = os.getenv("JWT_SECRET_KEY", "")
        
        if not secret:
            self.errors.append("JWT_SECRET_KEY is not set")
            return False
        
        if len(secret) < 32:
            self.errors.append(f"JWT_SECRET_KEY is too short ({len(secret)} chars, need 32+)")
            return False
        
        # Check for common placeholder patterns
        placeholders = ["changeme", "your-", "example", "placeholder", "secret", "test"]
        for placeholder in placeholders:
            if placeholder in secret.lower():
                self.errors.append(f"JWT_SECRET_KEY contains placeholder pattern: '{placeholder}'")
                return False
        
        self.passed.append(f"JWT_SECRET_KEY is configured ({len(secret)} chars)")
        return True

    def validate_database(self) -> bool:
        """Validate database configuration."""
        db_url = os.getenv("DATABASE_URL", "")
        db_password = os.getenv("POSTGRES_PASSWORD", "")
        
        # Check for default/weak passwords
        weak_patterns = ["changeme", "password", "123456", "postgres", "admin"]
        
        for pattern in weak_patterns:
            if pattern in db_url.lower() or pattern in db_password.lower():
                self.errors.append(f"Database password contains weak pattern: '{pattern}'")
                return False
        
        if not db_url:
            self.warnings.append("DATABASE_URL is not set")
        else:
            self.passed.append("DATABASE_URL is configured")
        
        return True

    def validate_redis(self) -> bool:
        """Validate Redis configuration."""
        redis_url = os.getenv("REDIS_URL", "")
        redis_password = os.getenv("REDIS_PASSWORD", "")
        
        if not redis_password and not redis_url:
            self.warnings.append("REDIS_PASSWORD is not set")
        elif redis_password and "changeme" in redis_password.lower():
            self.warnings.append("REDIS_PASSWORD contains 'changeme'")
        else:
            self.passed.append("Redis password configured")
        
        return True

    def validate_cors(self) -> bool:
        """Validate CORS configuration."""
        cors = os.getenv("CORS_ORIGINS", "")
        
        if not cors:
            self.errors.append("CORS_ORIGINS must be set in production")
            return False
        
        if "*" in cors:
            self.errors.append("CORS_ORIGINS cannot be '*' in production")
            return False
        
        # Check for localhost in production CORS
        if "localhost" in cors or "127.0.0.1" in cors:
            self.warnings.append("CORS_ORIGINS contains localhost - verify this is intentional")
        
        # Check for HTTPS
        origins = [o.strip() for o in cors.split(",")]
        non_https = [o for o in origins if o and not o.startswith("https://")]
        if non_https:
            self.warnings.append(f"Some CORS origins are not HTTPS: {non_https}")
        
        self.passed.append(f"CORS_ORIGINS configured: {cors[:50]}...")
        return True

    def validate_ssl(self) -> bool:
        """Check for SSL/TLS configuration hints.
        
        Returns True if SSL is properly configured, False if explicitly disabled.
        Adds warnings for missing but non-critical configuration.
        """
        # Check for SSL-related environment variables
        ssl_vars = ["SSL_CERT_PATH", "SSL_KEY_PATH", "FORCE_HTTPS"]
        configured = [var for var in ssl_vars if os.getenv(var)]
        
        # Check if SSL is explicitly disabled
        force_https = os.getenv("FORCE_HTTPS", "").lower()
        if force_https == "false":
            self.warnings.append("FORCE_HTTPS is explicitly disabled")
            return False  # Explicitly disabled
        
        if not configured:
            self.warnings.append("No SSL environment variables found - ensure HTTPS is configured at load balancer/proxy")
            # Not a failure - SSL may be at proxy level
        else:
            self.passed.append(f"SSL variables found: {configured}")
        
        return len(configured) > 0 or force_https != "false"

    def validate_api_keys(self) -> bool:
        """Validate AI provider API keys."""
        openai = os.getenv("OPENAI_API_KEY", "")
        anthropic = os.getenv("ANTHROPIC_API_KEY", "")
        mock_mode = os.getenv("MOCK_MODE", "true").lower() == "true"
        
        if mock_mode:
            self.warnings.append("MOCK_MODE is enabled - AI features use mock data")
            return True
        
        if not openai and not anthropic:
            self.warnings.append("No AI provider API keys configured")
        else:
            if openai and openai.startswith("sk-"):
                self.passed.append("OpenAI API key configured")
            if anthropic and anthropic.startswith("sk-ant-"):
                self.passed.append("Anthropic API key configured")
        
        # Return True - API keys are optional if mock mode is acceptable
        return bool(openai or anthropic or mock_mode)

    def validate_secrets_not_in_code(self) -> bool:
        """Check that secrets are not hardcoded in source files."""
        project_root = Path(__file__).parent.parent
        
        patterns_to_check = [
            r'sk-[a-zA-Z0-9]{32,}',  # OpenAI keys
            r'sk-ant-[a-zA-Z0-9]{32,}',  # Anthropic keys
            r'password\s*=\s*["\'][^"\']{8,}["\']',  # Hardcoded passwords
        ]
        
        files_to_check = list(project_root.glob("**/*.py")) + \
                         list(project_root.glob("**/*.ts")) + \
                         list(project_root.glob("**/*.tsx"))
        
        # Exclude node_modules, __pycache__, .git
        files_to_check = [f for f in files_to_check 
                        if "node_modules" not in str(f) 
                        and "__pycache__" not in str(f)
                        and ".git" not in str(f)]
        
        found_secrets = []
        for file_path in files_to_check[:100]:  # Limit to first 100 files
            try:
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                for pattern in patterns_to_check:
                    if re.search(pattern, content):
                        found_secrets.append(str(file_path))
                        break
            except (OSError, UnicodeDecodeError):
                pass  # Skip unreadable files
        
        if found_secrets:
            self.warnings.append(f"Possible secrets found in {len(found_secrets)} files")
        else:
            self.passed.append("No obvious hardcoded secrets detected")
        
        return True

    def run_all_validations(self) -> bool:
        """Run all validations and return overall result."""
        print_header("Production Environment Validation")
        
        validations = [
            ("Environment", self.validate_environment),
            ("Debug Mode", self.validate_debug_disabled),
            ("JWT Secret", self.validate_jwt_secret),
            ("Database", self.validate_database),
            ("Redis", self.validate_redis),
            ("CORS", self.validate_cors),
            ("SSL/TLS", self.validate_ssl),
            ("API Keys", self.validate_api_keys),
            ("Hardcoded Secrets", self.validate_secrets_not_in_code),
        ]
        
        for name, validator in validations:
            print(f"\nChecking {name}...")
            validator()
        
        # Print summary
        print_header("Validation Summary")
        
        if self.passed:
            print(f"\n{GREEN}Passed ({len(self.passed)}):{RESET}")
            for item in self.passed:
                print_success(item)
        
        if self.warnings:
            print(f"\n{YELLOW}Warnings ({len(self.warnings)}):{RESET}")
            for item in self.warnings:
                print_warning(item)
        
        if self.errors:
            print(f"\n{RED}Errors ({len(self.errors)}):{RESET}")
            for item in self.errors:
                print_error(item)
        
        print_header("Result")
        
        if self.errors:
            print_error(f"FAILED - {len(self.errors)} critical errors must be fixed before deployment")
            return False
        elif self.warnings:
            print_warning(f"PASSED WITH WARNINGS - Review {len(self.warnings)} warnings")
            return True
        else:
            print_success("ALL CHECKS PASSED - Ready for production deployment")
            return True


def main():
    """Main entry point."""
    # Load .env file if it exists
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        print(f"Loading environment from {env_file}")
        from dotenv import load_dotenv
        load_dotenv(env_file)
    
    # Also try .env.production
    prod_env = Path(__file__).parent.parent / ".env.production"
    if prod_env.exists():
        print(f"Loading production environment from {prod_env}")
        from dotenv import load_dotenv
        load_dotenv(prod_env, override=True)
    
    validator = ProductionValidator()
    success = validator.run_all_validations()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
