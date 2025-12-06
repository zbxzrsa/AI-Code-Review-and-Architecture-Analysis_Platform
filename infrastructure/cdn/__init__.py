"""
CDN Configuration Module (Performance Optimization #4)

Provides CDN management with:
- Cache policy configuration
- Resource versioning
- Compression optimization
"""
from .cdn_config import (
    CDNConfig,
    CDNManager,
    CachePolicy,
    CacheControl,
    CompressionType,
    ResourceType,
    ResourceVersioner,
    ResourceCompressor,
    RESOURCE_TYPES,
    get_cdn_manager,
    init_cdn_manager,
    generate_nginx_cdn_config,
)
