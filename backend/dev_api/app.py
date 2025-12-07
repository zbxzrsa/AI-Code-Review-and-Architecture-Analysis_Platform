"""
FastAPI 应用工厂 (FastAPI Application Factory)

模块功能描述:
    创建和配置 FastAPI 应用，注册所有路由和中间件。

架构说明:
    dev_api/
    ├── app.py           - 本文件（应用工厂）
    ├── config.py        - 配置管理
    ├── middleware.py    - 自定义中间件
    ├── models.py        - Pydantic 数据模型
    ├── mock_data.py     - 开发用模拟数据
    ├── core/            - 核心基础设施
    │   ├── config.py        - 系统配置
    │   ├── dependencies.py  - 依赖注入
    │   └── middleware.py    - 中间件实现
    ├── routes/          - API 路由模块
    │   ├── admin.py         - 管理员端点
    │   ├── analysis.py      - 代码分析
    │   ├── auth.py          - 认证
    │   ├── dashboard.py     - 仪表板指标
    │   ├── oauth.py         - OAuth 集成
    │   ├── projects.py      - 项目管理
    │   ├── reports.py       - 报告和备份
    │   ├── security.py      - 安全端点
    │   ├── three_version.py - 三版本演化
    │   ├── users.py         - 用户管理
    │   └── vulnerabilities.py - 漏洞管理
    └── services/        - 业务逻辑服务
        ├── code_review_service.py    - 代码审查逻辑
        ├── vulnerability_service.py  - 漏洞处理
        └── analytics_service.py      - 分析逻辑

最后修改日期: 2024-12-07
"""

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from .config import CORS_ORIGINS, IS_PRODUCTION, logger
from .middleware import RequestSizeLimitMiddleware

# Import monitoring (optional - graceful fallback if not available)
try:
    from backend.shared.monitoring import (
        MetricsCollector,
        PrometheusMiddleware,
        get_metrics,
        get_metrics_content_type,
        PROMETHEUS_AVAILABLE,
    )
except ImportError:
    PROMETHEUS_AVAILABLE = False
    PrometheusMiddleware = None
    MetricsCollector = None
from .routes import (
    admin_router,
    analysis_router,
    auth_router,
    dashboard_router,
    oauth_router,
    projects_router,
    reports_router,
    security_router,
    three_version_router,
    users_router,
    vulnerabilities_router,
)


def create_app() -> FastAPI:
    """
    创建和配置 FastAPI 应用
    
    功能描述:
        初始化 FastAPI 应用实例，配置 CORS、中间件和路由。
    
    返回值:
        FastAPI: 配置完成的 FastAPI 应用实例
    """
    
    application = FastAPI(
        title="Dev API Server",
        description="Development API server for frontend testing with modular architecture",
        version="2.1.0",
        docs_url="/docs" if not IS_PRODUCTION else None,
        redoc_url="/redoc" if not IS_PRODUCTION else None,
    )
    
    # Add middleware
    application.add_middleware(RequestSizeLimitMiddleware)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-CSRF-Token", "X-Request-ID", "X-API-Key"],
    )
    
    # Add Prometheus metrics middleware if available
    if PROMETHEUS_AVAILABLE and PrometheusMiddleware:
        application.add_middleware(PrometheusMiddleware)
        logger.info("Prometheus metrics middleware enabled")
    
    # Register routes (alphabetically organized)
    application.include_router(admin_router)
    application.include_router(analysis_router)
    application.include_router(auth_router)
    application.include_router(dashboard_router)
    application.include_router(oauth_router)
    application.include_router(projects_router)
    application.include_router(reports_router)
    application.include_router(security_router)
    application.include_router(three_version_router)
    application.include_router(users_router)
    application.include_router(vulnerabilities_router)
    
    # Health check endpoints
    @application.get("/")
    async def root():
        return {
            "service": "Dev API Server",
            "status": "running",
            "version": "2.1.0",
            "architecture": "modular",
        }
    
    @application.get("/health")
    async def health():
        return {"status": "healthy", "version": "2.1.0"}
    
    @application.get("/metrics")
    async def metrics():
        """Prometheus metrics endpoint."""
        if PROMETHEUS_AVAILABLE:
            # Update system metrics before returning
            MetricsCollector.update_system_metrics()
            return Response(
                content=get_metrics(),
                media_type=get_metrics_content_type()
            )
        return Response(
            content="# Prometheus metrics not available\n",
            media_type="text/plain"
        )
    
    @application.get("/api/info")
    async def api_info():
        """Get API information and available routes."""
        return {
            "name": "AI Code Review Platform API",
            "version": "2.1.0",
            "modules": [
                {"name": "auth", "prefix": "/api/auth", "description": "Authentication and authorization"},
                {"name": "projects", "prefix": "/api/projects", "description": "Project management"},
                {"name": "analysis", "prefix": "/api/analysis", "description": "Code analysis"},
                {"name": "vulnerabilities", "prefix": "/api/vulnerabilities", "description": "Vulnerability scanning"},
                {"name": "admin", "prefix": "/api/admin", "description": "System administration"},
                {"name": "dashboard", "prefix": "/api/dashboard", "description": "Dashboard metrics"},
                {"name": "security", "prefix": "/api/security", "description": "Security endpoints"},
                {"name": "three-version", "prefix": "/api/three-version", "description": "Three-version evolution"},
                {"name": "reports", "prefix": "/api/reports", "description": "Reports and backups"},
                {"name": "metrics", "prefix": "/metrics", "description": "Prometheus metrics"},
            ],
            "metrics_enabled": PROMETHEUS_AVAILABLE,
        }
    
    logger.info("Dev API Server initialized with modular architecture (v2.1.0)")
    
    return application


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
