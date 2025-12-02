"""
Auth Service - User authentication and authorization.

Responsibilities:
- User registration with invitation code validation
- JWT-based authentication (access + refresh tokens)
- Email verification via AWS SES/Postmark
- Password reset flow with time-limited tokens
- Role-based access control (RBAC)
- 2FA support: TOTP (optional) + WebAuthn (future)
"""
import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram
import time

from src.config import settings
from src.database import init_db, get_db
from src.routers import auth, users, roles
from src.middleware import audit_logger, rate_limiter

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
auth_requests = Counter(
    'auth_requests_total',
    'Total authentication requests',
    ['endpoint', 'status']
)

auth_duration = Histogram(
    'auth_request_duration_seconds',
    'Authentication request duration'
)

failed_logins = Counter(
    'failed_logins_total',
    'Total failed login attempts',
    ['reason']
)

app = FastAPI(
    title="Auth Service",
    description="Authentication and authorization service",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware
app.add_middleware(audit_logger.AuditLoggingMiddleware)
app.add_middleware(rate_limiter.RateLimitMiddleware)

# Initialize database
@app.on_event("startup")
async def startup():
    """Initialize database on startup."""
    await init_db()
    logger.info("Auth service started")

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Auth service shutting down")

# Health checks
@app.get("/health/live", tags=["Health"])
async def liveness():
    """Liveness probe."""
    return {"status": "alive"}

@app.get("/health/ready", tags=["Health"])
async def readiness(db = Depends(get_db)):
    """Readiness probe."""
    try:
        # Test database connection
        await db.execute("SELECT 1")
        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(users.router, prefix="/api/v1/users", tags=["Users"])
app.include_router(roles.router, prefix="/api/v1/roles", tags=["Roles"])

# Metrics endpoint
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
    )
