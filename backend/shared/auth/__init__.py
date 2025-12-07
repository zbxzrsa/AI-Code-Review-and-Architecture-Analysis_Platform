"""
共享认证模块 (Shared Authentication Module)

模块功能描述:
    提供 OAuth 提供者和认证工具。

主要功能:
    - OAuth 第三方登录（GitHub、GitLab 等）
    - OAuth 令牌管理
    - 用户信息获取

主要组件:
    - OAuthProviderBase: OAuth 提供者基类
    - OAuthProviderFactory: OAuth 提供者工厂
    - GitHubOAuth: GitHub OAuth 实现
    - GitLabOAuth: GitLab OAuth 实现

最后修改日期: 2024-12-07
"""

from .oauth_providers import (
    OAuthProviderBase,
    OAuthProviderFactory,
    GitHubOAuth,
    GitLabOAuth,
    OAuthToken,
    OAuthUser,
    OAuthRepository,
)

__all__ = [
    "OAuthProviderBase",
    "OAuthProviderFactory",
    "GitHubOAuth",
    "GitLabOAuth",
    "OAuthToken",
    "OAuthUser",
    "OAuthRepository",
]
