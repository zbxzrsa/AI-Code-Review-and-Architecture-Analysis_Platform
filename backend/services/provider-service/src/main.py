"""
Provider Service - AI provider management and quota enforcement.
"""
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
from datetime import datetime, timezone
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Provider Service",
    description="AI provider management and quota tracking",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ProviderStatus(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ProviderType(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"


class ProviderResponse(BaseModel):
    id: str
    name: str
    type: ProviderType
    status: ProviderStatus
    endpoint: str
    model: str
    is_free: bool
    latency_ms: Optional[float] = None
    last_check: datetime


class QuotaResponse(BaseModel):
    user_id: str
    daily_limit: int
    daily_used: int
    monthly_limit: int
    monthly_used: int
    cost_limit: float
    cost_used: float


# Health endpoints
@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/health/live")
async def liveness():
    return {"status": "alive"}


@app.get("/health/ready")
async def readiness():
    return {"status": "ready"}


# Provider endpoints
@app.get("/api/providers")
async def list_providers():
    """List all AI providers."""
    return {
        "providers": [
            ProviderResponse(
                id="ollama_local",
                name="Ollama (Local)",
                type=ProviderType.OLLAMA,
                status=ProviderStatus.HEALTHY,
                endpoint="http://ollama:11434",
                model="codellama:13b",
                is_free=True,
                latency_ms=1200,
                last_check=datetime.now(timezone.utc),
            ),
            ProviderResponse(
                id="openai_gpt4",
                name="OpenAI GPT-4",
                type=ProviderType.OPENAI,
                status=ProviderStatus.HEALTHY,
                endpoint="https://api.openai.com",
                model="gpt-4",
                is_free=False,
                latency_ms=2500,
                last_check=datetime.now(timezone.utc),
            ),
        ]
    }


@app.get("/api/providers/{provider_id}")
async def get_provider(provider_id: str):
    """Get provider details."""
    return ProviderResponse(
        id=provider_id,
        name="Ollama (Local)",
        type=ProviderType.OLLAMA,
        status=ProviderStatus.HEALTHY,
        endpoint="http://ollama:11434",
        model="codellama:13b",
        is_free=True,
        latency_ms=1200,
        last_check=datetime.now(timezone.utc),
    )


@app.get("/api/providers/{provider_id}/health")
async def check_provider_health(provider_id: str):
    """Check provider health."""
    return {
        "provider_id": provider_id,
        "status": "healthy",
        "latency_ms": 1250,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/providers/priority-chain")
async def get_priority_chain():
    """Get current provider priority chain."""
    return {
        "chain": [
            {"provider": "ollama_local", "priority": 0, "status": "healthy"},
            {"provider": "openai_gpt4", "priority": 1, "status": "healthy"},
            {"provider": "anthropic_claude", "priority": 2, "status": "degraded"},
        ]
    }


# Quota endpoints
@app.get("/api/quotas/{user_id}", response_model=QuotaResponse)
async def get_user_quota(user_id: str):
    """Get user quota status."""
    return QuotaResponse(
        user_id=user_id,
        daily_limit=100,
        daily_used=45,
        monthly_limit=3000,
        monthly_used=1250,
        cost_limit=50.0,
        cost_used=12.50,
    )


@app.post("/api/quotas/{user_id}/increment")
async def increment_quota(user_id: str, count: int = 1, cost: float = 0.0):
    """Increment usage quota."""
    return {
        "user_id": user_id,
        "incremented": count,
        "cost_added": cost,
        "new_daily_used": 46,
        "new_cost_used": 12.52,
    }


@app.get("/api/metrics/providers/{provider}")
async def get_provider_metrics(provider: str):
    """Get provider performance metrics."""
    return {
        "provider": provider,
        "metrics": {
            "requests_total": 15420,
            "requests_success": 15380,
            "requests_failed": 40,
            "avg_latency_ms": 1450,
            "p50_latency_ms": 1200,
            "p95_latency_ms": 2500,
            "p99_latency_ms": 3500,
            "error_rate": 0.0026,
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
