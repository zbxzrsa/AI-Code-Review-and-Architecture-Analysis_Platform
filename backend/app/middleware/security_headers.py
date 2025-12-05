"""
Security Headers Middleware

Adds security headers to all responses:
- Content-Security-Policy (CSP)
- X-Frame-Options
- X-Content-Type-Options
- X-XSS-Protection
- Strict-Transport-Security (HSTS)
- Referrer-Policy
- Permissions-Policy
"""

from typing import List, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from app.core.config import settings


# ============================================
# CSP Constants
# ============================================
CSP_SELF = "'self'"
CSP_UNSAFE_INLINE = "'unsafe-inline'"
CSP_UNSAFE_EVAL = "'unsafe-eval'"
CSP_NONE = "'none'"


# ============================================
# Content Security Policy Configuration
# ============================================

class CSPDirectives:
    """CSP directive builder"""
    
    def __init__(self):
        self.directives: dict[str, List[str]] = {}
    
    def add(self, directive: str, *sources: str) -> "CSPDirectives":
        if directive not in self.directives:
            self.directives[directive] = []
        self.directives[directive].extend(sources)
        return self
    
    def build(self) -> str:
        parts = []
        for directive, sources in self.directives.items():
            if sources:
                parts.append(f"{directive} {' '.join(sources)}")
            else:
                parts.append(directive)
        return "; ".join(parts)


def get_csp_policy(is_development: bool = False) -> str:
    """
    Build Content Security Policy
    
    Production: Strict CSP
    Development: Relaxed for hot reload, dev tools
    """
    csp = CSPDirectives()
    
    # Default source
    csp.add("default-src", CSP_SELF)
    
    # Script sources
    if is_development:
        csp.add("script-src", CSP_SELF, CSP_UNSAFE_INLINE, CSP_UNSAFE_EVAL)
    else:
        csp.add("script-src", CSP_SELF, CSP_UNSAFE_INLINE)  # For Monaco editor
    
    # Style sources (need unsafe-inline for Ant Design)
    csp.add("style-src", CSP_SELF, CSP_UNSAFE_INLINE)
    
    # Image sources
    csp.add("img-src", CSP_SELF, "data:", "blob:", "https:")
    
    # Font sources
    csp.add("font-src", CSP_SELF, "data:")
    
    # Connect sources (API, WebSocket)
    if is_development:
        csp.add("connect-src", CSP_SELF, "ws:", "wss:", "http://localhost:*")
    else:
        csp.add("connect-src", CSP_SELF, "wss:")
    
    # Frame ancestors (prevent clickjacking)
    csp.add("frame-ancestors", CSP_SELF)
    
    # Form actions
    csp.add("form-action", CSP_SELF)
    
    # Base URI
    csp.add("base-uri", CSP_SELF)
    
    # Object sources
    csp.add("object-src", CSP_NONE)
    
    # Upgrade insecure requests in production
    if not is_development:
        csp.add("upgrade-insecure-requests")
    
    return csp.build()


# ============================================
# Security Headers Configuration
# ============================================

class SecurityHeadersConfig:
    """Security headers configuration"""
    
    def __init__(
        self,
        csp_policy: Optional[str] = None,
        frame_options: str = "SAMEORIGIN",
        content_type_options: str = "nosniff",
        xss_protection: str = "1; mode=block",
        hsts_max_age: int = 31536000,  # 1 year
        hsts_include_subdomains: bool = True,
        hsts_preload: bool = True,
        referrer_policy: str = "strict-origin-when-cross-origin",
        permissions_policy: Optional[str] = None,
        enable_hsts: bool = True,
    ):
        self.csp_policy = csp_policy
        self.frame_options = frame_options
        self.content_type_options = content_type_options
        self.xss_protection = xss_protection
        self.hsts_max_age = hsts_max_age
        self.hsts_include_subdomains = hsts_include_subdomains
        self.hsts_preload = hsts_preload
        self.referrer_policy = referrer_policy
        self.permissions_policy = permissions_policy or self._default_permissions_policy()
        self.enable_hsts = enable_hsts
    
    def _default_permissions_policy(self) -> str:
        """Default permissions policy (restrictive)"""
        policies = [
            "geolocation=()",
            "microphone=()",
            "camera=()",
            "payment=()",
            "usb=()",
            "magnetometer=()",
            "gyroscope=()",
            "accelerometer=()",
        ]
        return ", ".join(policies)
    
    def get_hsts_value(self) -> str:
        """Build HSTS header value"""
        value = f"max-age={self.hsts_max_age}"
        if self.hsts_include_subdomains:
            value += "; includeSubDomains"
        if self.hsts_preload:
            value += "; preload"
        return value


# ============================================
# Middleware
# ============================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for adding security headers
    """
    
    def __init__(self, app, config: Optional[SecurityHeadersConfig] = None):
        super().__init__(app)
        self.config = config or SecurityHeadersConfig()
        
        # Set CSP based on environment
        if not self.config.csp_policy:
            is_dev = getattr(settings, "DEBUG", False) or \
                     getattr(settings, "ENVIRONMENT", "production") == "development"
            self.config.csp_policy = get_csp_policy(is_development=is_dev)
    
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        
        # Content Security Policy
        if self.config.csp_policy:
            response.headers["Content-Security-Policy"] = self.config.csp_policy
        
        # X-Frame-Options (prevent clickjacking)
        response.headers["X-Frame-Options"] = self.config.frame_options
        
        # X-Content-Type-Options (prevent MIME sniffing)
        response.headers["X-Content-Type-Options"] = self.config.content_type_options
        
        # X-XSS-Protection (legacy XSS protection)
        response.headers["X-XSS-Protection"] = self.config.xss_protection
        
        # Strict-Transport-Security (HTTPS only)
        # Only add in production over HTTPS
        if self.config.enable_hsts:
            is_https = request.url.scheme == "https" or \
                       request.headers.get("X-Forwarded-Proto") == "https"
            if is_https:
                response.headers["Strict-Transport-Security"] = self.config.get_hsts_value()
        
        # Referrer-Policy
        response.headers["Referrer-Policy"] = self.config.referrer_policy
        
        # Permissions-Policy
        if self.config.permissions_policy:
            response.headers["Permissions-Policy"] = self.config.permissions_policy
        
        # Additional security headers
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        
        return response


# ============================================
# Factory function for easy setup
# ============================================

def create_security_headers_middleware(
    csp_policy: Optional[str] = None,
    enable_hsts: bool = True,
    development_mode: bool = False,
) -> SecurityHeadersMiddleware:
    """
    Create security headers middleware with sensible defaults
    
    Args:
        csp_policy: Custom CSP policy (auto-generated if None)
        enable_hsts: Enable HSTS header
        development_mode: Use relaxed settings for development
    
    Returns:
        Configured SecurityHeadersMiddleware
    """
    config = SecurityHeadersConfig(
        csp_policy=csp_policy or get_csp_policy(is_development=development_mode),
        enable_hsts=enable_hsts and not development_mode,
    )
    
    # In development, allow framing for debugging
    if development_mode:
        config.frame_options = "SAMEORIGIN"
    
    return SecurityHeadersMiddleware(app=None, config=config)


# ============================================
# Usage example for FastAPI app
# ============================================
"""
from fastapi import FastAPI
from app.middleware.security_headers import SecurityHeadersMiddleware, SecurityHeadersConfig

app = FastAPI()

# Option 1: Use defaults
app.add_middleware(SecurityHeadersMiddleware)

# Option 2: Custom configuration
config = SecurityHeadersConfig(
    csp_policy="default-src 'self'; script-src 'self' 'unsafe-inline'",
    enable_hsts=True,
)
app.add_middleware(SecurityHeadersMiddleware, config=config)
"""
