"""
Admin Routes / 管理员路由
"""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query
import random

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.get("/users")
async def list_users(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = None
):
    """List all users / 列出所有用户"""
    users = [
        {
            "id": f"user_{i}",
            "email": f"user{i}@example.com",
            "name": f"User {i}",
            "role": "admin" if i == 1 else "user",
            "status": "active",
            "created_at": (datetime.now() - timedelta(days=i * 10)).isoformat(),
            "last_login": (datetime.now() - timedelta(hours=i)).isoformat()
        }
        for i in range(1, 21)
    ]
    
    if search:
        users = [u for u in users if search.lower() in u["name"].lower() or search.lower() in u["email"].lower()]
    
    total = len(users)
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "items": users[start:end],
        "total": total,
        "page": page,
        "limit": limit
    }


@router.get("/users/stats")
async def get_user_stats():
    """Get user statistics / 获取用户统计"""
    return {
        "total": 156,
        "active": 142,
        "inactive": 14,
        "admins": 5,
        "new_this_month": 12,
        "growth_rate": 8.5
    }


@router.get("/providers")
async def list_providers():
    """List AI providers / 列出 AI 提供商"""
    return {
        "providers": [
            {
                "id": "openai",
                "name": "OpenAI",
                "type": "api",
                "status": "active",
                "models": ["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                "requestsToday": random.randint(100, 500),
                "costToday": round(random.uniform(5, 50), 2),
                "avgLatency": random.randint(200, 800),
                "errorRate": round(random.uniform(0, 2), 2)
            },
            {
                "id": "anthropic",
                "name": "Anthropic",
                "type": "api",
                "status": "active",
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "requestsToday": random.randint(50, 300),
                "costToday": round(random.uniform(3, 30), 2),
                "avgLatency": random.randint(300, 1000),
                "errorRate": round(random.uniform(0, 1.5), 2)
            }
        ]
    }


@router.get("/stats")
async def get_admin_stats():
    """Get admin statistics / 获取管理统计"""
    return {
        "users": {"total": 156, "active": 142},
        "projects": {"total": 89, "active": 67},
        "analyses": {"today": 234, "total": 15678},
        "api_calls": {"today": 1250, "total": 98765},
        "costs": {"today": 45.67, "month": 1234.56}
    }


@router.get("/audit")
async def get_audit_logs(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100)
):
    """Get audit logs / 获取审计日志"""
    logs = [
        {
            "id": f"audit_{i}",
            "user": f"user_{i % 5 + 1}",
            "action": ["login", "create_project", "run_analysis", "update_settings"][i % 4],
            "resource": f"resource_{i}",
            "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
            "ip": f"192.168.1.{i % 255}"
        }
        for i in range(1, 101)
    ]
    
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "items": logs[start:end],
        "total": 100,
        "page": page,
        "limit": limit
    }


@router.get("/invitations")
async def list_invitations():
    """List invitations / 列出邀请"""
    return {
        "invitations": [
            {
                "id": "inv_1",
                "email": "newuser@example.com",
                "role": "user",
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "expires_at": (datetime.now() + timedelta(days=7)).isoformat()
            }
        ]
    }


@router.get("/ai-models")
async def list_ai_models():
    """List AI models / 列出 AI 模型"""
    return {
        "models": [
            {
                "id": "gpt-4-turbo",
                "name": "GPT-4 Turbo",
                "provider": "OpenAI",
                "version": "v2",
                "status": "active",
                "accuracy": 0.94,
                "latency": 650,
                "cost_per_1k": 0.01
            },
            {
                "id": "claude-3-opus",
                "name": "Claude 3 Opus",
                "provider": "Anthropic",
                "version": "v2",
                "status": "active",
                "accuracy": 0.92,
                "latency": 800,
                "cost_per_1k": 0.015
            }
        ]
    }
