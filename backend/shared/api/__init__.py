"""
API Utilities Module

Provides:
- API versioning (TD-003)
- Request/Response utilities
"""
from .versioning import (
    APIVersion,
    VersionInfo,
    VersionedRouter,
    VersionNegotiator,
    VersionedResponse,
    VersionMigrator,
    create_versioned_app,
    versioned_response,
    get_request_version,
    deprecation_warning,
    VERSION_INFO,
    SUPPORTED_VERSIONS,
    DEPRECATED_VERSIONS,
    DEFAULT_VERSION,
)
