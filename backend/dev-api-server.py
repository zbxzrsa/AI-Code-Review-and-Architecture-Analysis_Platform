"""
开发环境 API 服务器入口 (Development API Server - Entry Point)

模块功能描述:
    这是一个精简的入口文件，从模块化的 dev_api 包导入应用。
    实际实现位于 backend/dev_api/ 目录下。

目录结构:
    dev_api/
    ├── __init__.py      - 包初始化
    ├── app.py           - FastAPI 应用工厂
    ├── config.py        - 配置和常量
    ├── models.py        - Pydantic 数据模型
    ├── mock_data.py     - 开发用模拟数据
    ├── middleware.py    - 自定义中间件
    └── routes/          - API 路由模块
        ├── admin.py         - 管理员端点
        ├── analysis.py      - 代码分析
        ├── dashboard.py     - 仪表板指标
        ├── oauth.py         - OAuth 集成
        ├── projects.py      - 项目管理
        ├── reports.py       - 报告和备份
        ├── security.py      - 安全端点
        ├── three_version.py - 三版本演化
        └── users.py         - 用户管理

运行方式:
    python dev-api-server.py
    或: uvicorn dev_api:app --reload --host 0.0.0.0 --port 8000

迁移说明:
    此文件从 4,492 行重构为约 80 行。
    旧版本备份于: dev-api-server.py.backup
    请参阅 dev_api/ 目录获取模块化实现。

最后修改日期: 2024-12-07
"""

import os
import sys

# Add backend to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

# Import the modular app
from dev_api import app
from dev_api.config import ENVIRONMENT, MOCK_MODE, logger

# Re-export for backward compatibility
__all__ = ["app"]


def main():
    """
    运行开发环境 API 服务器
    
    功能描述:
        启动 FastAPI 开发服务器，显示配置信息并开启热重载模式。
    """
    print("=" * 60)
    print("🚀 AI Code Review Platform - Dev API Server")
    print("=" * 60)
    print(f"🔧 Environment: {ENVIRONMENT}")
    print(f"🎭 Mock Mode: {'ENABLED (no AI keys required)' if MOCK_MODE else 'DISABLED (requires AI keys)'}")
    print("=" * 60)
    print("📦 Using modular architecture from dev_api/")
    print("=" * 60)
    print("🌐 Server:    http://localhost:8000")
    print("📖 API Docs:  http://localhost:8000/docs")
    print("❤️  Health:   http://localhost:8000/health")
    print("=" * 60)
    if MOCK_MODE:
        print("ℹ️  Running in mock mode - AI responses are simulated")
        print("   Set MOCK_MODE=false in .env to use real AI providers")
    print("=" * 60)
    
    uvicorn.run(
        "dev_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
