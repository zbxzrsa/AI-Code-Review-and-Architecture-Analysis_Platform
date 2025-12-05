#!/usr/bin/env python3
"""
Development Seed Data Script

Creates sample data for local development and testing.
Run with: python scripts/seed_data.py

WARNING: This will clear existing data! Only use in development.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta, timezone
from uuid import uuid4
import random
import hashlib

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

import asyncpg
from argon2 import PasswordHasher

# Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ai_codereview")

# Password hasher
ph = PasswordHasher()


# =============================================================================
# Sample Data
# =============================================================================

USERS = [
    {
        "email": "admin@example.com",
        "name": "Admin User",
        "role": "admin",
        "password": "Admin123!",
    },
    {
        "email": "developer@example.com",
        "name": "Developer User",
        "role": "developer",
        "password": "Dev123!",
    },
    {
        "email": "reviewer@example.com",
        "name": "Code Reviewer",
        "role": "reviewer",
        "password": "Review123!",
    },
    {
        "email": "test@example.com",
        "name": "Test User",
        "role": "user",
        "password": "Test123!",
    },
]

LANGUAGES = ["python", "javascript", "typescript", "java", "go", "rust", "ruby"]

PROJECT_NAMES = [
    "E-commerce Platform",
    "AI Chatbot",
    "Data Pipeline",
    "Mobile App Backend",
    "Authentication Service",
    "Analytics Dashboard",
    "Content Management System",
    "Payment Gateway",
    "Notification Service",
    "Search Engine",
]

SAMPLE_CODE = {
    "python": '''
def calculate_metrics(data: list[dict]) -> dict:
    """Calculate various metrics from data."""
    if not data:
        return {"count": 0, "avg": 0, "max": 0, "min": 0}
    
    values = [d.get("value", 0) for d in data]
    return {
        "count": len(values),
        "avg": sum(values) / len(values),
        "max": max(values),
        "min": min(values),
    }

class DataProcessor:
    def __init__(self, config: dict):
        self.config = config
        self._cache = {}
    
    async def process(self, items: list) -> list:
        results = []
        for item in items:
            result = await self._process_item(item)
            results.append(result)
        return results
    
    async def _process_item(self, item: dict) -> dict:
        # Process individual item
        return {"processed": True, **item}
''',
    "javascript": '''
class UserService {
    constructor(database) {
        this.db = database;
        this.cache = new Map();
    }

    async getUser(id) {
        if (this.cache.has(id)) {
            return this.cache.get(id);
        }
        
        const user = await this.db.users.findById(id);
        if (user) {
            this.cache.set(id, user);
        }
        return user;
    }

    async createUser(data) {
        const user = await this.db.users.create({
            ...data,
            createdAt: new Date(),
        });
        return user;
    }
}

module.exports = { UserService };
''',
    "typescript": '''
interface User {
    id: string;
    email: string;
    name: string;
    createdAt: Date;
}

interface CreateUserDTO {
    email: string;
    name: string;
    password: string;
}

export class AuthService {
    private users: Map<string, User> = new Map();

    async register(dto: CreateUserDTO): Promise<User> {
        const id = crypto.randomUUID();
        const user: User = {
            id,
            email: dto.email,
            name: dto.name,
            createdAt: new Date(),
        };
        this.users.set(id, user);
        return user;
    }

    async findById(id: string): Promise<User | undefined> {
        return this.users.get(id);
    }
}
''',
}

ISSUE_TYPES = ["security", "performance", "style", "bug", "smell"]
ISSUE_SEVERITIES = ["critical", "high", "medium", "low", "info"]


# =============================================================================
# Database Operations
# =============================================================================

async def create_connection():
    """Create database connection"""
    return await asyncpg.connect(DATABASE_URL)


async def clear_data(conn):
    """Clear existing data"""
    print("🗑️  Clearing existing data...")
    
    tables = [
        "auth.audit_logs",
        "auth.sessions",
        "auth.password_resets",
        "auth.invitations",
        "production.code_review_results",
        "production.analysis_artifacts",
        "production.analysis_tasks",
        "production.analysis_sessions",
        "projects.project_history",
        "projects.project_member",
        "projects.project_files",
        "projects.project_versions",
        "projects.projects",
        "auth.users",
    ]
    
    for table in tables:
        try:
            await conn.execute(f"DELETE FROM {table}")
        except Exception as e:
            print(f"  Warning: Could not clear {table}: {e}")
    
    print("  Done!")


async def seed_users(conn) -> list:
    """Create sample users"""
    print("👤 Creating users...")
    
    user_ids = []
    for user in USERS:
        user_id = str(uuid4())
        password_hash = ph.hash(user["password"])
        
        await conn.execute("""
            INSERT INTO auth.users (id, email, password_hash, name, role, is_active, created_at)
            VALUES ($1, $2, $3, $4, $5, true, $6)
        """, user_id, user["email"], password_hash, user["name"], user["role"], datetime.now(timezone.utc))
        
        user_ids.append(user_id)
        print(f"  Created: {user['email']} ({user['role']})")
    
    return user_ids


async def seed_projects(conn, user_ids: list) -> list:
    """Create sample projects"""
    print("📁 Creating projects...")
    
    project_ids = []
    for i, name in enumerate(PROJECT_NAMES):
        project_id = str(uuid4())
        owner_id = random.choice(user_ids)
        language = random.choice(LANGUAGES)
        
        await conn.execute("""
            INSERT INTO projects.projects (id, name, description, language, owner_id, status, created_at)
            VALUES ($1, $2, $3, $4, $5, 'active', $6)
        """, 
            project_id, 
            name, 
            f"Sample project for {name.lower()}",
            language,
            owner_id,
            datetime.now(timezone.utc) - timedelta(days=random.randint(1, 30))
        )
        
        project_ids.append((project_id, language))
        print(f"  Created: {name} ({language})")
    
    return project_ids


async def seed_project_files(conn, projects: list):
    """Create sample project files"""
    print("📄 Creating project files...")
    
    for project_id, language in projects:
        # Get sample code for language or use Python as default
        code = SAMPLE_CODE.get(language, SAMPLE_CODE["python"])
        
        file_id = str(uuid4())
        extension = {"python": "py", "javascript": "js", "typescript": "ts"}.get(language, "py")
        
        await conn.execute("""
            INSERT INTO projects.project_files (id, project_id, path, content, language, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
        """,
            file_id,
            project_id,
            f"src/main.{extension}",
            code,
            language,
            datetime.now(timezone.utc)
        )
    
    print(f"  Created {len(projects)} files")


async def seed_analysis_sessions(conn, projects: list, user_ids: list):
    """Create sample analysis sessions"""
    print("🔍 Creating analysis sessions...")
    
    session_count = 0
    for project_id, language in projects[:5]:  # Only first 5 projects
        for _ in range(random.randint(1, 3)):
            session_id = str(uuid4())
            user_id = random.choice(user_ids)
            
            await conn.execute("""
                INSERT INTO production.analysis_sessions 
                (id, project_id, user_id, status, analysis_type, created_at, completed_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
                session_id,
                project_id,
                user_id,
                random.choice(["completed", "completed", "failed"]),
                random.choice(["quick", "deep", "security"]),
                datetime.now(timezone.utc) - timedelta(hours=random.randint(1, 72)),
                datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 1)),
            )
            session_count += 1
    
    print(f"  Created {session_count} sessions")


async def seed_invitations(conn, user_ids: list):
    """Create sample invitation codes"""
    print("📨 Creating invitations...")
    
    codes = ["WELCOME2024", "DEVTEAM", "REVIEWER", "PARTNER", "EARLYBIRD"]
    
    for code in codes:
        await conn.execute("""
            INSERT INTO auth.invitations (id, code, email, role, uses_remaining, expires_at, created_by, created_at)
            VALUES ($1, $2, NULL, $3, $4, $5, $6, $7)
        """,
            str(uuid4()),
            code,
            random.choice(["user", "developer", "reviewer"]),
            random.randint(5, 20),
            datetime.now(timezone.utc) + timedelta(days=30),
            user_ids[0],  # Admin user
            datetime.now(timezone.utc),
        )
        print(f"  Created: {code}")


async def seed_audit_logs(conn, user_ids: list):
    """Create sample audit logs"""
    print("📝 Creating audit logs...")
    
    actions = ["login", "logout", "create_project", "update_project", "run_analysis"]
    
    for _ in range(50):
        await conn.execute("""
            INSERT INTO auth.audit_logs (id, user_id, action, resource_type, resource_id, ip_address, created_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
            str(uuid4()),
            random.choice(user_ids),
            random.choice(actions),
            random.choice(["user", "project", "analysis"]),
            str(uuid4()),
            f"192.168.1.{random.randint(1, 254)}",
            datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 168)),
        )
    
    print("  Created 50 log entries")


# =============================================================================
# Main
# =============================================================================

async def main():
    print("🌱 AI Code Review Platform - Seed Data")
    print("=" * 50)
    
    # Confirm in production-like environments
    env = os.getenv("ENVIRONMENT", "development")
    if env not in ["development", "test", "local"]:
        print(f"⚠️  Environment is '{env}'. This script is for development only.")
        confirm = await asyncio.to_thread(input, "Are you sure you want to continue? (yes/no): ")
        if confirm.lower() != "yes":
            print("Aborted.")
            return
    
    try:
        conn = await create_connection()
        print(f"✅ Connected to database")
        
        # Clear and seed
        await clear_data(conn)
        user_ids = await seed_users(conn)
        projects = await seed_projects(conn, user_ids)
        await seed_project_files(conn, projects)
        await seed_analysis_sessions(conn, projects, user_ids)
        await seed_invitations(conn, user_ids)
        await seed_audit_logs(conn, user_ids)
        
        await conn.close()
        
        print("=" * 50)
        print("✅ Seed data created successfully!")
        print()
        print("Test Accounts:")
        print("-" * 30)
        for user in USERS:
            print(f"  {user['email']} / {user['password']} ({user['role']})")
        print()
        print("Invitation Codes: WELCOME2024, DEVTEAM, REVIEWER, PARTNER, EARLYBIRD")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
