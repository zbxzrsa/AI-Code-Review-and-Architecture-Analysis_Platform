#!/usr/bin/env python3
"""
Secure Initialization Script

This script should be run during first production deployment to:
- Generate secure passwords and secrets
- Update default admin password
- Create production-ready environment file

Usage:
    python secure_init.py --database-url "postgresql://..." --generate-secrets
"""

import argparse
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict

try:
    import psycopg2
    from argon2 import PasswordHasher
    HAS_DEPENDENCIES = True
except ImportError:
    HAS_DEPENDENCIES = False


def generate_secure_password(length: int = 32) -> str:
    """Generate a cryptographically secure password."""
    return secrets.token_urlsafe(length)


def generate_hex_secret(length: int = 64) -> str:
    """Generate a hex-encoded secret."""
    return secrets.token_hex(length)


def hash_password(password: str) -> str:
    """Hash password using Argon2id (recommended for password hashing)."""
    ph = PasswordHasher(
        time_cost=3,
        memory_cost=65536,  # 64MB
        parallelism=4,
    )
    return ph.hash(password)


def generate_all_secrets() -> Dict[str, str]:
    """Generate all required secrets for production deployment."""
    return {
        "ADMIN_PASSWORD": generate_secure_password(24),
        "JWT_SECRET_KEY": generate_hex_secret(64),
        "JWT_REFRESH_SECRET_KEY": generate_hex_secret(64),
        "POSTGRES_PASSWORD": generate_secure_password(32),
        "REDIS_PASSWORD": generate_secure_password(32),
        "NEO4J_PASSWORD": generate_secure_password(32),
        "MINIO_ROOT_PASSWORD": generate_secure_password(32),
        "ENCRYPTION_KEY": generate_hex_secret(32),
        "CSRF_SECRET": generate_hex_secret(32),
    }


def write_secrets_file(secrets_data: Dict[str, str], output_path: Path) -> None:
    """Write secrets to environment file."""
    with open(output_path, "w") as f:
        f.write("# Production Secrets - KEEP SECURE!\n")
        f.write(f"# Generated at: {datetime.now().isoformat()}\n")
        f.write("# WARNING: Store this file securely and never commit to version control\n\n")

        for key, value in secrets_data.items():
            f.write(f"{key}={value}\n")

    # Set restrictive permissions (owner read/write only)
    os.chmod(output_path, 0o600)


def update_admin_password(database_url: str, new_password: str) -> bool:
    """Update the default admin password in the database."""
    if not HAS_DEPENDENCIES:
        print("ERROR: psycopg2 and argon2-cffi are required for database operations")
        print("Install with: pip install psycopg2-binary argon2-cffi")
        return False

    password_hash = hash_password(new_password)

    try:
        conn = psycopg2.connect(database_url)
        with conn.cursor() as cur:
            # Update admin password
            cur.execute(
                """
                UPDATE auth.users
                SET password_hash = %s,
                    updated_at = NOW()
                WHERE email = 'admin@example.com'
                RETURNING id
                """,
                (password_hash,)
            )

            result = cur.fetchone()
            if result:
                conn.commit()
                print(f"[OK] Admin password updated for user ID: {result[0]}")
                return True
            else:
                print("[WARN] No admin user found with email 'admin@example.com'")
                return False

    except psycopg2.Error as e:
        print(f"[ERROR] Database error: {e}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()


def verify_database_connection(database_url: str) -> bool:
    """Verify database connection is working."""
    if not HAS_DEPENDENCIES:
        return False

    try:
        conn = psycopg2.connect(database_url)
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        conn.close()
        return True
    except psycopg2.Error as e:
        print(f"[ERROR] Cannot connect to database: {e}")
        return False


def print_security_checklist() -> None:
    """Print security checklist for production deployment."""
    checklist = """
=======================================================
PRODUCTION SECURITY CHECKLIST
=======================================================

Before deploying to production, ensure:

[ ] All default passwords have been changed
[ ] JWT secrets are unique and stored securely
[ ] Database passwords use strong random values
[ ] HTTPS is enabled with valid certificates
[ ] CORS origins are properly configured
[ ] Rate limiting is enabled
[ ] Audit logging is enabled
[ ] Backup encryption keys are stored safely
[ ] Environment files are not in version control
[ ] Security headers are properly configured
[ ] Input validation is enabled on all endpoints
[ ] SQL injection protection is verified
[ ] XSS protection is enabled

After deployment:

[ ] Run security scan (e.g., OWASP ZAP)
[ ] Verify audit logs are recording
[ ] Test authentication flows
[ ] Verify rate limiting is working
[ ] Check security headers with securityheaders.com
=======================================================
"""
    print(checklist)


def main():
    parser = argparse.ArgumentParser(
        description="Secure initialization script for production deployment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate secrets only:
    python secure_init.py --generate-secrets

  Generate secrets and update database:
    python secure_init.py --database-url "postgresql://user:pass@host:5432/db" --generate-secrets

  Custom output file:
    python secure_init.py --generate-secrets --output /secure/path/.env.production
        """
    )

    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection URL for updating admin password"
    )
    parser.add_argument(
        "--generate-secrets",
        action="store_true",
        help="Generate new secure secrets"
    )
    parser.add_argument(
        "--output",
        default=".env.production.secrets",
        help="Output file for generated secrets (default: .env.production.secrets)"
    )
    parser.add_argument(
        "--show-checklist",
        action="store_true",
        help="Show security checklist for production deployment"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )

    args = parser.parse_args()

    if args.show_checklist:
        print_security_checklist()
        return

    if not args.generate_secrets:
        parser.print_help()
        print("\n[INFO] Use --generate-secrets to generate new production secrets")
        return

    print("=" * 60)
    print("SECURE INITIALIZATION")
    print("=" * 60)
    print()

    # Generate secrets
    print("[1/3] Generating secure secrets...")
    secrets_data = generate_all_secrets()

    if args.dry_run:
        print("[DRY RUN] Would generate the following secrets:")
        for key in secrets_data:
            print(f"  - {key}: {'*' * 20}")
    else:
        output_path = Path(args.output)
        write_secrets_file(secrets_data, output_path)
        print(f"[OK] Secrets written to: {output_path}")
        print("[OK] File permissions set to 600 (owner read/write only)")

    # Update database if URL provided
    if args.database_url:
        print()
        print("[2/3] Updating admin password in database...")

        if args.dry_run:
            print("[DRY RUN] Would update admin password in database")
        else:
            if verify_database_connection(args.database_url):
                if update_admin_password(args.database_url, secrets_data["ADMIN_PASSWORD"]):
                    print("[OK] Admin password updated successfully")
                else:
                    print("[WARN] Failed to update admin password")
            else:
                print("[ERROR] Could not connect to database")
    else:
        print()
        print("[2/3] Skipping database update (no --database-url provided)")

    # Print summary
    print()
    print("[3/3] Summary")
    print("-" * 40)

    if not args.dry_run:
        print(f"Secrets file: {args.output}")
        print(f"Admin password: ADMIN_PASSWORD in {args.output}")
        print()
        print("IMPORTANT NEXT STEPS:")
        print("1. Copy secrets to your secure deployment environment")
        print("2. Update your deployment configuration to use these secrets")
        print("3. Delete the local secrets file after deployment")
        print("4. Run --show-checklist to see full security checklist")

    print()
    print("[OK] Secure initialization complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
