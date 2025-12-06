"""
API Versioning System (TD-003)

Implements API versioning with:
- /v1/api routes
- Version switching mechanism
- Backward compatibility
- Migration support

Usage:
    from fastapi import FastAPI
    from backend.shared.api.versioning import create_versioned_app, VersionedRouter
    
    app = create_versioned_app()
    
    # Create versioned routers
    v1_router = VersionedRouter(version="v1")
    v2_router = VersionedRouter(version="v2")
    
    @v1_router.get("/users")
    async def get_users_v1():
        return {"version": "v1", "users": [...]}
    
    @v2_router.get("/users")
    async def get_users_v2():
        return {"version": "v2", "data": {"users": [...]}}
"""
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Set, Type, Union

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


@dataclass
class VersionInfo:
    """Information about an API version."""
    version: str
    status: str  # stable, beta, deprecated, sunset
    release_date: str
    sunset_date: Optional[str] = None
    description: str = ""
    breaking_changes: List[str] = None
    
    def __post_init__(self):
        if self.breaking_changes is None:
            self.breaking_changes = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "status": self.status,
            "release_date": self.release_date,
            "sunset_date": self.sunset_date,
            "description": self.description,
            "breaking_changes": self.breaking_changes,
        }


# Version registry
VERSION_INFO: Dict[str, VersionInfo] = {
    "v1": VersionInfo(
        version="v1",
        status="stable",
        release_date="2024-01-01",
        description="Initial stable API version",
    ),
    "v2": VersionInfo(
        version="v2",
        status="beta",
        release_date="2024-06-01",
        description="Enhanced API with improved response structure",
        breaking_changes=[
            "Response wrapper changed from flat to nested structure",
            "Pagination parameters renamed",
            "Error format standardized",
        ],
    ),
}

DEFAULT_VERSION = APIVersion.V1
SUPPORTED_VERSIONS: Set[str] = {"v1", "v2"}
DEPRECATED_VERSIONS: Set[str] = set()


class VersionedRouter(APIRouter):
    """
    Router that automatically prefixes routes with API version.
    
    Usage:
        v1_router = VersionedRouter(version="v1")
        
        @v1_router.get("/users")
        async def get_users():
            return {"users": [...]}
        
        # Route becomes: /v1/users
    """
    
    def __init__(self, version: str, **kwargs):
        if version not in SUPPORTED_VERSIONS:
            raise ValueError(f"Unsupported version: {version}. Supported: {SUPPORTED_VERSIONS}")
        
        self.version = version
        prefix = kwargs.pop("prefix", "")
        super().__init__(prefix=f"/{version}{prefix}", **kwargs)
        
        # Track endpoints for documentation
        self._endpoints: List[Dict[str, str]] = []
    
    def api_route(self, path: str, **kwargs):
        """Override to track endpoints."""
        self._endpoints.append({
            "path": f"/{self.version}{path}",
            "methods": kwargs.get("methods", ["GET"]),
        })
        return super().api_route(path, **kwargs)


class VersionNegotiator:
    """
    Handles API version negotiation from request headers or path.
    """
    
    HEADER_NAME = "X-API-Version"
    ACCEPT_HEADER_PREFIX = "application/vnd.coderev.v"
    
    @classmethod
    def get_version(cls, request: Request) -> str:
        """
        Extract API version from request.
        
        Priority:
        1. X-API-Version header
        2. Accept header (application/vnd.coderev.v1+json)
        3. URL path prefix (/v1/...)
        4. Default version
        """
        # Check X-API-Version header
        version_header = request.headers.get(cls.HEADER_NAME)
        if version_header and version_header in SUPPORTED_VERSIONS:
            return version_header
        
        # Check Accept header
        accept = request.headers.get("Accept", "")
        if cls.ACCEPT_HEADER_PREFIX in accept:
            for version in SUPPORTED_VERSIONS:
                if f"{cls.ACCEPT_HEADER_PREFIX}{version.replace('v', '')}" in accept:
                    return version
        
        # Check URL path
        path = request.url.path
        for version in SUPPORTED_VERSIONS:
            if path.startswith(f"/{version}/") or path == f"/{version}":
                return version
        
        return DEFAULT_VERSION.value
    
    @classmethod
    def is_deprecated(cls, version: str) -> bool:
        """Check if version is deprecated."""
        return version in DEPRECATED_VERSIONS
    
    @classmethod
    def get_sunset_date(cls, version: str) -> Optional[str]:
        """Get sunset date for deprecated version."""
        info = VERSION_INFO.get(version)
        return info.sunset_date if info else None


def add_version_headers(response: Response, version: str):
    """Add version-related headers to response."""
    response.headers["X-API-Version"] = version
    
    if VersionNegotiator.is_deprecated(version):
        sunset_date = VersionNegotiator.get_sunset_date(version)
        response.headers["Deprecation"] = "true"
        if sunset_date:
            response.headers["Sunset"] = sunset_date
        response.headers["X-API-Warn"] = f"API version {version} is deprecated"


class VersionedResponse(BaseModel):
    """Standard versioned API response wrapper."""
    version: str
    timestamp: str
    data: Any
    meta: Optional[Dict[str, Any]] = None
    
    @classmethod
    def create(cls, data: Any, version: str, meta: Optional[Dict] = None) -> "VersionedResponse":
        return cls(
            version=version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            data=data,
            meta=meta,
        )


def versioned_response(version: str = "v1"):
    """
    Decorator to wrap response in versioned format.
    
    Usage:
        @app.get("/users")
        @versioned_response("v1")
        async def get_users():
            return [{"id": 1, "name": "John"}]
        
        # Response: {"version": "v1", "timestamp": "...", "data": [...]}
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            return VersionedResponse.create(result, version)
        return wrapper
    return decorator


class VersionMigrator:
    """
    Handles request/response migration between versions.
    
    Usage:
        migrator = VersionMigrator()
        
        @migrator.migrate_request("v1", "v2")
        def migrate_user_request(data):
            # Transform v1 request to v2 format
            return {"user_id": data.get("id")}
        
        @migrator.migrate_response("v2", "v1")
        def migrate_user_response(data):
            # Transform v2 response to v1 format
            return {"id": data.get("user_id")}
    """
    
    def __init__(self):
        self._request_migrations: Dict[tuple, Callable] = {}
        self._response_migrations: Dict[tuple, Callable] = {}
    
    def migrate_request(self, from_version: str, to_version: str):
        """Register request migration function."""
        def decorator(func: Callable):
            self._request_migrations[(from_version, to_version)] = func
            return func
        return decorator
    
    def migrate_response(self, from_version: str, to_version: str):
        """Register response migration function."""
        def decorator(func: Callable):
            self._response_migrations[(from_version, to_version)] = func
            return func
        return decorator
    
    def transform_request(self, data: Any, from_version: str, to_version: str) -> Any:
        """Transform request data between versions."""
        migration = self._request_migrations.get((from_version, to_version))
        if migration:
            return migration(data)
        return data
    
    def transform_response(self, data: Any, from_version: str, to_version: str) -> Any:
        """Transform response data between versions."""
        migration = self._response_migrations.get((from_version, to_version))
        if migration:
            return migration(data)
        return data


def create_versioned_app(
    title: str = "API",
    description: str = "",
    default_version: str = "v1",
) -> FastAPI:
    """
    Create FastAPI app with versioning support.
    
    Usage:
        app = create_versioned_app(
            title="Code Review API",
            description="AI-powered code review",
        )
    """
    app = FastAPI(
        title=title,
        description=description,
        version=default_version,
    )
    
    # Version info endpoint
    @app.get("/versions", tags=["System"])
    async def get_versions():
        """Get available API versions."""
        return {
            "default": default_version,
            "supported": list(SUPPORTED_VERSIONS),
            "deprecated": list(DEPRECATED_VERSIONS),
            "versions": {
                v: info.to_dict() 
                for v, info in VERSION_INFO.items()
            },
        }
    
    # Version middleware
    @app.middleware("http")
    async def version_middleware(request: Request, call_next):
        # Determine version
        version = VersionNegotiator.get_version(request)
        
        # Store version in request state
        request.state.api_version = version
        
        # Process request
        response = await call_next(request)
        
        # Add version headers
        add_version_headers(response, version)
        
        return response
    
    return app


# Migration examples for common patterns

class UserMigrations:
    """Example migrations for user-related endpoints."""
    
    migrator = VersionMigrator()
    
    @staticmethod
    @migrator.migrate_request("v1", "v2")
    def migrate_user_request_v1_to_v2(data: Dict) -> Dict:
        """Migrate user request from v1 to v2."""
        return {
            "user_id": data.get("id"),
            "email_address": data.get("email"),
            "display_name": data.get("name"),
            "metadata": {},
        }
    
    @staticmethod
    @migrator.migrate_response("v2", "v1")
    def migrate_user_response_v2_to_v1(data: Dict) -> Dict:
        """Migrate user response from v2 to v1."""
        return {
            "id": data.get("user_id"),
            "email": data.get("email_address"),
            "name": data.get("display_name"),
        }


# Utility functions

def get_request_version(request: Request) -> str:
    """Get API version from request state."""
    return getattr(request.state, "api_version", DEFAULT_VERSION.value)


def deprecation_warning(
    version: str,
    message: str,
    sunset_date: Optional[str] = None
) -> Dict[str, Any]:
    """Generate deprecation warning for response."""
    warning = {
        "type": "deprecation",
        "version": version,
        "message": message,
    }
    if sunset_date:
        warning["sunset_date"] = sunset_date
    return warning
