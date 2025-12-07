"""
共享数据库模块 (Shared Database Module)

模块功能描述:
    提供数据库连接管理和查询功能。

主要功能:
    - 使用 asyncpg 的异步连接池
    - SQLAlchemy 异步会话管理
    - 健康检查和连接监控
    - 指数退避自动重连

主要组件:
    - DatabaseManager: 数据库管理器
    - SecureQueryBuilder: 安全查询构建器
    - get_async_session: 获取异步会话

最后修改日期: 2024-12-07
"""

from .connection import (
    DatabaseManager,
    get_database,
    get_async_session,
    init_database,
    close_database,
)
from .secure_queries import SecureQueryBuilder, QueryType

__all__ = [
    "DatabaseManager",
    "get_database",
    "get_async_session",
    "init_database",
    "close_database",
    "SecureQueryBuilder",
    "QueryType",
]
