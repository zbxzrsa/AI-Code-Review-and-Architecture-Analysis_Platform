"""
Health Check Module for Microservices

Provides standardized health check endpoints:
- /health - Liveness check (is the service running?)
- /ready - Readiness check (is the service ready to accept traffic?)
- /health/live - Alias for liveness
- /health/ready - Alias for readiness

Includes:
- Database connectivity checks
- Redis connectivity checks
- External service dependency checks
- Graceful shutdown handling
"""

import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable, Awaitable
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, APIRouter, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Health Status Models
# =============================================================================

class HealthStatus(str, Enum):
    """Health status enumeration"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ComponentHealth(BaseModel):
    """Health status of a single component"""
    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    last_check: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    """Full health check response"""
    status: HealthStatus
    service: str
    version: str = "1.0.0"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    uptime_seconds: float = 0
    components: List[ComponentHealth] = []
    details: Optional[Dict[str, Any]] = None


class LivenessResponse(BaseModel):
    """Simple liveness response"""
    status: str = "alive"
    service: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ReadinessResponse(BaseModel):
    """Readiness response"""
    status: str
    service: str
    ready: bool
    checks: Dict[str, bool] = {}
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# =============================================================================
# Health Check Registry
# =============================================================================

class HealthCheckRegistry:
    """
    Registry for health check functions
    
    Allows services to register their specific health checks
    """
    
    def __init__(self, service_name: str, version: str = "1.0.0"):
        self.service_name = service_name
        self.version = version
        self.start_time = datetime.now(timezone.utc)
        self._checks: Dict[str, Callable[[], Awaitable[ComponentHealth]]] = {}
        self._is_shutting_down = False
        self._is_ready = False
    
    def register(self, name: str):
        """Decorator to register a health check function"""
        def decorator(func: Callable[[], Awaitable[ComponentHealth]]):
            self._checks[name] = func
            return func
        return decorator
    
    def add_check(self, name: str, func: Callable[[], Awaitable[ComponentHealth]]):
        """Add a health check function"""
        self._checks[name] = func
    
    def remove_check(self, name: str):
        """Remove a health check"""
        self._checks.pop(name, None)
    
    def set_ready(self, ready: bool = True):
        """Set service readiness status"""
        self._is_ready = ready
    
    def set_shutting_down(self, shutting_down: bool = True):
        """Set shutdown status"""
        self._is_shutting_down = shutting_down
    
    @property
    def uptime_seconds(self) -> float:
        """Get service uptime in seconds"""
        return (datetime.now(timezone.utc) - self.start_time).total_seconds()
    
    async def check_component(self, name: str) -> ComponentHealth:
        """Run a single component check"""
        check_func = self._checks.get(name)
        if not check_func:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message="Check not found"
            )
        
        start_time = datetime.now(timezone.utc)
        try:
            result = await asyncio.wait_for(check_func(), timeout=5.0)
            result.latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            return result
        except asyncio.TimeoutError:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=5000,
                message="Check timed out"
            )
        except Exception as e:
            return ComponentHealth(
                name=name,
                status=HealthStatus.UNHEALTHY,
                latency_ms=(datetime.now(timezone.utc) - start_time).total_seconds() * 1000,
                message=str(e)
            )
    
    async def run_all_checks(self) -> HealthResponse:
        """Run all registered health checks"""
        components = []
        
        # Run all checks concurrently
        if self._checks:
            results = await asyncio.gather(
                *[self.check_component(name) for name in self._checks.keys()],
                return_exceptions=True
            )
            
            for result in results:
                if isinstance(result, ComponentHealth):
                    components.append(result)
                elif isinstance(result, Exception):
                    components.append(ComponentHealth(
                        name="unknown",
                        status=HealthStatus.UNHEALTHY,
                        message=str(result)
                    ))
        
        # Determine overall status
        if self._is_shutting_down:
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.UNHEALTHY for c in components):
            overall_status = HealthStatus.UNHEALTHY
        elif any(c.status == HealthStatus.DEGRADED for c in components):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
        
        return HealthResponse(
            status=overall_status,
            service=self.service_name,
            version=self.version,
            uptime_seconds=self.uptime_seconds,
            components=components,
        )
    
    async def check_readiness(self) -> ReadinessResponse:
        """Check if service is ready to accept traffic"""
        checks = {}
        
        # Run all registered checks
        for name in self._checks.keys():
            result = await self.check_component(name)
            checks[name] = result.status == HealthStatus.HEALTHY
        
        # Service is ready if not shutting down and all checks pass
        is_ready = (
            self._is_ready and 
            not self._is_shutting_down and 
            all(checks.values())
        )
        
        return ReadinessResponse(
            status="ready" if is_ready else "not_ready",
            service=self.service_name,
            ready=is_ready,
            checks=checks,
        )


# =============================================================================
# Common Health Check Functions
# =============================================================================

async def check_database(get_connection) -> ComponentHealth:
    """
    Check database connectivity
    
    Args:
        get_connection: Async function that returns a database connection
    """
    try:
        async with get_connection() as conn:
            await conn.execute("SELECT 1")
        return ComponentHealth(
            name="database",
            status=HealthStatus.HEALTHY,
            message="Connected"
        )
    except Exception as e:
        return ComponentHealth(
            name="database",
            status=HealthStatus.UNHEALTHY,
            message=f"Connection failed: {str(e)}"
        )


async def check_redis(redis_client) -> ComponentHealth:
    """
    Check Redis connectivity
    
    Args:
        redis_client: Redis client instance with ping() method
    """
    try:
        await redis_client.ping()
        return ComponentHealth(
            name="redis",
            status=HealthStatus.HEALTHY,
            message="Connected"
        )
    except Exception as e:
        return ComponentHealth(
            name="redis",
            status=HealthStatus.UNHEALTHY,
            message=f"Connection failed: {str(e)}"
        )


async def check_http_service(
    url: str,
    name: str,
    timeout: float = 3.0
) -> ComponentHealth:
    """
    Check HTTP service availability
    
    Args:
        url: Health check URL of the service
        name: Name of the service
        timeout: Request timeout in seconds
    """
    import httpx
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=timeout)
            if response.status_code == 200:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=f"HTTP {response.status_code}"
                )
            else:
                return ComponentHealth(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    message=f"HTTP {response.status_code}"
                )
    except Exception as e:
        return ComponentHealth(
            name=name,
            status=HealthStatus.UNHEALTHY,
            message=str(e)
        )


# =============================================================================
# FastAPI Router
# =============================================================================

def create_health_router(registry: HealthCheckRegistry) -> APIRouter:
    """
    Create FastAPI router with health check endpoints
    
    Args:
        registry: Health check registry instance
    """
    router = APIRouter(tags=["health"])
    
    @router.get("/health", response_model=HealthResponse)
    @router.get("/health/live", response_model=HealthResponse)
    async def liveness_check():
        """
        Liveness check - Is the service running?
        
        Returns 200 if the service is alive.
        Used by Kubernetes liveness probe.
        """
        if registry._is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "service": registry.service_name,
                    "message": "Service is shutting down"
                }
            )
        
        result = await registry.run_all_checks()
        status_code = 200 if result.status == HealthStatus.HEALTHY else 503
        return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))
    
    @router.get("/ready", response_model=ReadinessResponse)
    @router.get("/health/ready", response_model=ReadinessResponse)
    async def readiness_check():
        """
        Readiness check - Is the service ready to accept traffic?
        
        Returns 200 if ready, 503 if not ready.
        Used by Kubernetes readiness probe.
        """
        result = await registry.check_readiness()
        status_code = 200 if result.ready else 503
        return JSONResponse(status_code=status_code, content=result.model_dump(mode="json"))
    
    @router.get("/health/startup")
    async def startup_check():
        """
        Startup check - Has the service started successfully?
        
        Used by Kubernetes startup probe.
        """
        return JSONResponse(
            status_code=200 if registry._is_ready else 503,
            content={
                "status": "started" if registry._is_ready else "starting",
                "service": registry.service_name
            }
        )
    
    return router


# =============================================================================
# Graceful Shutdown Handler
# =============================================================================

class GracefulShutdown:
    """
    Graceful shutdown handler for microservices
    
    Handles:
    - SIGTERM and SIGINT signals
    - Draining connections before shutdown
    - Coordinating with health check registry
    """
    
    def __init__(
        self,
        registry: HealthCheckRegistry,
        shutdown_timeout: int = 30,
        drain_delay: int = 5,
    ):
        self.registry = registry
        self.shutdown_timeout = shutdown_timeout
        self.drain_delay = drain_delay
        self._shutdown_event = asyncio.Event()
        self._cleanup_tasks: List[Callable[[], Awaitable[None]]] = []
    
    def add_cleanup_task(self, task: Callable[[], Awaitable[None]]):
        """Add a cleanup task to run during shutdown"""
        self._cleanup_tasks.append(task)
    
    def setup_signal_handlers(self, loop: asyncio.AbstractEventLoop = None):
        """Setup signal handlers for graceful shutdown
        
        Args:
            loop: Event loop to use. If None, tries to get running loop.
        """
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running loop, get or create one (for non-async setup)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda s=sig: asyncio.create_task(self._handle_signal(s))
            )
    
    async def _handle_signal(self, sig: signal.Signals):
        """Handle shutdown signal"""
        logger.info(f"Received signal {sig.name}, initiating graceful shutdown")
        
        # Mark as shutting down (health checks will fail)
        self.registry.set_shutting_down(True)
        self.registry.set_ready(False)
        
        # Wait for load balancer to drain connections
        logger.info(f"Waiting {self.drain_delay}s for connection drain")
        await asyncio.sleep(self.drain_delay)
        
        # Run cleanup tasks
        for task in self._cleanup_tasks:
            try:
                await asyncio.wait_for(task(), timeout=5.0)
            except Exception as e:
                logger.error(f"Cleanup task failed: {e}")
        
        # Signal shutdown complete
        self._shutdown_event.set()
        
        logger.info("Graceful shutdown complete")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        await self._shutdown_event.wait()


# =============================================================================
# FastAPI Lifespan Context Manager
# =============================================================================

def create_lifespan(
    registry: HealthCheckRegistry,
    on_startup: Optional[Callable[[], Awaitable[None]]] = None,
    on_shutdown: Optional[Callable[[], Awaitable[None]]] = None,
):
    """
    Create FastAPI lifespan context manager with health check integration
    
    Args:
        registry: Health check registry
        on_startup: Optional startup callback
        on_shutdown: Optional shutdown callback
    """
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        logger.info(f"Starting {registry.service_name}")
        
        if on_startup:
            await on_startup()
        
        # Mark as ready
        registry.set_ready(True)
        logger.info(f"{registry.service_name} is ready")
        
        yield
        
        # Shutdown
        logger.info(f"Shutting down {registry.service_name}")
        registry.set_shutting_down(True)
        registry.set_ready(False)
        
        if on_shutdown:
            await on_shutdown()
        
        logger.info(f"{registry.service_name} shutdown complete")
    
    return lifespan


# =============================================================================
# Quick Setup Function
# =============================================================================

def setup_health_checks(
    app: FastAPI,
    service_name: str,
    version: str = "1.0.0",
) -> HealthCheckRegistry:
    """
    Quick setup for health checks in a FastAPI application
    
    Args:
        app: FastAPI application instance
        service_name: Name of the service
        version: Service version
    
    Returns:
        HealthCheckRegistry instance for adding custom checks
    
    Example:
        ```python
        app = FastAPI()
        registry = setup_health_checks(app, "my-service", "1.0.0")
        
        @registry.register("database")
        async def check_db():
            # Your database check logic
            return ComponentHealth(name="database", status=HealthStatus.HEALTHY)
        ```
    """
    registry = HealthCheckRegistry(service_name, version)
    router = create_health_router(registry)
    app.include_router(router)
    
    return registry
