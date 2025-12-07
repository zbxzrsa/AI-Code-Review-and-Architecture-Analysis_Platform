"""
CDN Static Resource Distribution (Performance Optimization #4)

Provides CDN configuration and resource management with:
- Cache policy management
- Resource versioning
- Compression optimization
- Cache invalidation

Expected Benefits:
- Reduce server load by 15-25%
- Faster asset delivery
- Improved global performance
"""
import hashlib
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class CacheControl(str, Enum):
    """Cache-Control header directives."""
    NO_CACHE = "no-cache"
    NO_STORE = "no-store"
    PRIVATE = "private"
    PUBLIC = "public"
    IMMUTABLE = "immutable"
    MUST_REVALIDATE = "must-revalidate"


class CompressionType(str, Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    BROTLI = "br"
    ZSTD = "zstd"


@dataclass
class CachePolicy:
    """Cache policy configuration for a resource type."""
    max_age: int  # seconds
    s_maxage: Optional[int] = None  # CDN cache time
    cache_control: CacheControl = CacheControl.PUBLIC
    immutable: bool = False
    stale_while_revalidate: Optional[int] = None
    stale_if_error: Optional[int] = None
    vary: List[str] = field(default_factory=lambda: ["Accept-Encoding"])

    def to_header(self) -> str:
        """Generate Cache-Control header value."""
        parts = [self.cache_control.value]
        parts.append(f"max-age={self.max_age}")

        if self.s_maxage:
            parts.append(f"s-maxage={self.s_maxage}")

        if self.immutable:
            parts.append("immutable")

        if self.stale_while_revalidate:
            parts.append(f"stale-while-revalidate={self.stale_while_revalidate}")

        if self.stale_if_error:
            parts.append(f"stale-if-error={self.stale_if_error}")

        return ", ".join(parts)


@dataclass
class ResourceType:
    """Configuration for a type of static resource."""
    extensions: Set[str]
    cache_policy: CachePolicy
    compress: bool = True
    min_compress_size: int = 1024  # Only compress if > 1KB
    supported_compressions: List[CompressionType] = field(
        default_factory=lambda: [CompressionType.BROTLI, CompressionType.GZIP]
    )


# Default resource type configurations
RESOURCE_TYPES = {
    "javascript": ResourceType(
        extensions={".js", ".mjs"},
        cache_policy=CachePolicy(
            max_age=31536000,  # 1 year
            s_maxage=31536000,
            immutable=True,
        ),
        compress=True,
    ),
    "css": ResourceType(
        extensions={".css"},
        cache_policy=CachePolicy(
            max_age=31536000,
            s_maxage=31536000,
            immutable=True,
        ),
        compress=True,
    ),
    "images": ResourceType(
        extensions={".png", ".jpg", ".jpeg", ".gif", ".webp", ".avif", ".svg"},
        cache_policy=CachePolicy(
            max_age=2592000,  # 30 days
            s_maxage=2592000,
        ),
        compress=False,  # Already compressed
    ),
    "fonts": ResourceType(
        extensions={".woff", ".woff2", ".ttf", ".otf", ".eot"},
        cache_policy=CachePolicy(
            max_age=31536000,
            s_maxage=31536000,
            immutable=True,
        ),
        compress=False,  # Already compressed
    ),
    "html": ResourceType(
        extensions={".html", ".htm"},
        cache_policy=CachePolicy(
            max_age=0,
            s_maxage=3600,  # CDN caches for 1 hour
            cache_control=CacheControl.NO_CACHE,
            must_revalidate=True,
            stale_while_revalidate=86400,
        ),
        compress=True,
    ),
    "json": ResourceType(
        extensions={".json"},
        cache_policy=CachePolicy(
            max_age=300,  # 5 minutes
            s_maxage=3600,
        ),
        compress=True,
    ),
    "sourcemaps": ResourceType(
        extensions={".map"},
        cache_policy=CachePolicy(
            max_age=31536000,
            s_maxage=31536000,
            immutable=True,
        ),
        compress=True,
    ),
}


@dataclass
class CDNConfig:
    """CDN configuration settings."""
    enabled: bool = True
    cdn_url: str = ""
    origin_url: str = ""

    # Cache settings
    default_max_age: int = 86400  # 1 day
    respect_origin_headers: bool = True

    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9 for gzip, 1-11 for brotli

    # Versioning
    enable_versioning: bool = True
    version_param: str = "v"
    hash_length: int = 8

    # Security
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])

    # Performance
    http2_push: bool = True
    preload_critical_resources: bool = True


class ResourceVersioner:
    """
    Handles resource versioning for cache busting.

    Generates content-based hashes for versioned URLs.
    """

    def __init__(self, config: CDNConfig):
        self._config = config
        self._version_cache: Dict[str, str] = {}

    def get_content_hash(self, content: bytes) -> str:
        """Generate content hash for versioning."""
        return hashlib.sha256(content).hexdigest()[:self._config.hash_length]

    def get_file_hash(self, file_path: str) -> str:
        """Get hash for a file."""
        if file_path in self._version_cache:
            return self._version_cache[file_path]

        try:
            with open(file_path, "rb") as f:
                content = f.read()
            hash_value = self.get_content_hash(content)
            self._version_cache[file_path] = hash_value
            return hash_value
        except Exception as e:
            logger.error(f"Failed to hash file {file_path}: {e}")
            return "unknown"

    def version_url(self, url: str, file_path: Optional[str] = None) -> str:
        """Add version parameter to URL."""
        if not self._config.enable_versioning:
            return url

        if file_path:
            version = self.get_file_hash(file_path)
        else:
            version = datetime.now().strftime("%Y%m%d%H")

        separator = "&" if "?" in url else "?"
        return f"{url}{separator}{self._config.version_param}={version}"

    def clear_cache(self):
        """Clear version cache."""
        self._version_cache.clear()


class ResourceCompressor:
    """
    Handles resource compression.

    Supports gzip and brotli compression with configurable levels.
    """

    def __init__(self, config: CDNConfig):
        self._config = config

    def compress_gzip(self, content: bytes) -> bytes:
        """Compress content with gzip."""
        import gzip
        return gzip.compress(content, compresslevel=self._config.compression_level)

    def compress_brotli(self, content: bytes) -> bytes:
        """Compress content with brotli."""
        try:
            import brotli
            return brotli.compress(
                content,
                quality=min(self._config.compression_level, 11)
            )
        except ImportError:
            logger.warning("Brotli not available, falling back to gzip")
            return self.compress_gzip(content)

    def compress(
        self,
        content: bytes,
        compression_type: CompressionType
    ) -> Tuple[bytes, float]:
        """
        Compress content with specified algorithm.

        Returns:
            Tuple of (compressed_content, compression_ratio)
        """
        if compression_type == CompressionType.NONE:
            return content, 1.0

        if compression_type == CompressionType.GZIP:
            compressed = self.compress_gzip(content)
        elif compression_type == CompressionType.BROTLI:
            compressed = self.compress_brotli(content)
        else:
            return content, 1.0

        ratio = len(compressed) / len(content) if content else 1.0
        return compressed, ratio

    def should_compress(
        self,
        content: bytes,
        resource_type: ResourceType
    ) -> bool:
        """Determine if content should be compressed."""
        if not self._config.enable_compression:
            return False

        if not resource_type.compress:
            return False

        if len(content) < resource_type.min_compress_size:
            return False

        return True


class CDNManager:
    """
    Main CDN management class.

    Handles:
    - Cache policy generation
    - Resource versioning
    - Compression
    - Cache invalidation
    """

    def __init__(self, config: Optional[CDNConfig] = None):
        self._config = config or CDNConfig()
        self._versioner = ResourceVersioner(self._config)
        self._compressor = ResourceCompressor(self._config)
        self._invalidation_queue: List[str] = []

    def get_resource_type(self, file_path: str) -> Optional[ResourceType]:
        """Get resource type configuration for a file."""
        ext = Path(file_path).suffix.lower()

        for resource_type in RESOURCE_TYPES.values():
            if ext in resource_type.extensions:
                return resource_type

        return None

    def get_cache_headers(self, file_path: str) -> Dict[str, str]:
        """Get cache headers for a resource."""
        resource_type = self.get_resource_type(file_path)

        if not resource_type:
            # Default policy for unknown types
            return {
                "Cache-Control": "public, max-age=3600",
            }

        headers = {
            "Cache-Control": resource_type.cache_policy.to_header(),
        }

        if resource_type.cache_policy.vary:
            headers["Vary"] = ", ".join(resource_type.cache_policy.vary)

        return headers

    def get_cdn_url(self, path: str, file_path: Optional[str] = None) -> str:
        """Get CDN URL for a resource."""
        if not self._config.enabled or not self._config.cdn_url:
            return path

        # Add versioning
        versioned_path = self._versioner.version_url(path, file_path)

        # Combine with CDN URL
        cdn_url = self._config.cdn_url.rstrip("/")
        return f"{cdn_url}/{versioned_path.lstrip('/')}"

    def process_resource(
        self,
        content: bytes,
        file_path: str,
        accept_encoding: str = ""
    ) -> Tuple[bytes, Dict[str, str]]:
        """
        Process resource for CDN delivery.

        Returns:
            Tuple of (content, headers)
        """
        resource_type = self.get_resource_type(file_path)
        headers = self.get_cache_headers(file_path)

        # Determine compression
        if resource_type and self._compressor.should_compress(content, resource_type):
            # Check client support
            if "br" in accept_encoding and CompressionType.BROTLI in resource_type.supported_compressions:
                content, _ratio = self._compressor.compress(content, CompressionType.BROTLI)
                headers["Content-Encoding"] = "br"
            elif "gzip" in accept_encoding and CompressionType.GZIP in resource_type.supported_compressions:
                content, _ratio = self._compressor.compress(content, CompressionType.GZIP)
                headers["Content-Encoding"] = "gzip"

        # Add content type
        content_type, _ = mimetypes.guess_type(file_path)
        if content_type:
            headers["Content-Type"] = content_type

        headers["Content-Length"] = str(len(content))

        return content, headers

    def queue_invalidation(self, paths: List[str]):
        """Queue paths for cache invalidation."""
        self._invalidation_queue.extend(paths)
        logger.info(f"Queued {len(paths)} paths for invalidation")

    def invalidate_all(self):
        """Invalidate all cached resources."""
        self._versioner.clear_cache()
        self._invalidation_queue.append("/*")
        logger.info("Queued full cache invalidation")

    def get_pending_invalidations(self) -> List[str]:
        """Get and clear pending invalidations."""
        paths = self._invalidation_queue.copy()
        self._invalidation_queue.clear()
        return paths

    def generate_preload_headers(
        self,
        resources: List[Dict[str, str]]
    ) -> List[str]:
        """
        Generate Link headers for HTTP/2 push.

        Args:
            resources: List of dicts with 'path' and 'as' keys

        Returns:
            List of Link header values
        """
        headers = []

        for resource in resources:
            path = resource.get("path", "")
            as_type = resource.get("as", "")

            if path and as_type:
                cdn_url = self.get_cdn_url(path)
                headers.append(f'<{cdn_url}>; rel=preload; as={as_type}')

        return headers

    def get_stats(self) -> Dict[str, Any]:
        """Get CDN statistics."""
        return {
            "enabled": self._config.enabled,
            "cdn_url": self._config.cdn_url,
            "versioning_enabled": self._config.enable_versioning,
            "compression_enabled": self._config.enable_compression,
            "cached_versions": len(self._versioner._version_cache),
            "pending_invalidations": len(self._invalidation_queue),
        }


# Nginx configuration generator
def generate_nginx_cdn_config(config: CDNConfig) -> str:
    """Generate Nginx configuration for CDN caching."""
    return f'''# CDN Caching Configuration
# Generated for Performance Optimization #4

# Enable gzip compression
gzip on;
gzip_vary on;
gzip_proxied any;
gzip_comp_level {config.compression_level};
gzip_types text/plain text/css text/xml application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;
gzip_min_length 1024;

# Enable brotli compression (requires ngx_brotli module)
brotli on;
brotli_comp_level {min(config.compression_level, 6)};
brotli_types text/plain text/css text/xml application/json application/javascript application/xml+rss application/atom+xml image/svg+xml;

# Static file caching
location ~* \\.(js|css|png|jpg|jpeg|gif|ico|svg|woff|woff2|ttf|eot)$ {{
    expires 1y;
    add_header Cache-Control "public, immutable";
    add_header Vary "Accept-Encoding";

    # Enable HTTP/2 server push
    http2_push_preload on;
}}

# HTML caching (short TTL)
location ~* \\.html$ {{
    expires 1h;
    add_header Cache-Control "no-cache, must-revalidate";
    add_header Vary "Accept-Encoding";
}}

# API responses (no cache)
location /api/ {{
    add_header Cache-Control "no-store, no-cache, must-revalidate";
    add_header Pragma "no-cache";
}}

# Security headers
add_header X-Content-Type-Options "nosniff" always;
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-XSS-Protection "1; mode=block" always;
'''


# Global CDN manager instance
_cdn_manager: Optional[CDNManager] = None


def get_cdn_manager() -> CDNManager:
    """Get or create global CDN manager."""
    global _cdn_manager
    if _cdn_manager is None:
        _cdn_manager = CDNManager()
    return _cdn_manager


def init_cdn_manager(config: Optional[CDNConfig] = None) -> CDNManager:
    """Initialize global CDN manager."""
    global _cdn_manager
    _cdn_manager = CDNManager(config)
    return _cdn_manager
