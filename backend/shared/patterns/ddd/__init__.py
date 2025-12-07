"""
领域驱动设计实现 (Domain-Driven Design Implementation)

模块功能描述:
    为平台提供清晰的领域边界和限界上下文。

子领域:
    - 代码分析（核心领域）
    - 版本管理（核心领域）
    - 用户与认证（支撑领域）
    - 提供者管理（支撑领域）
    - 审计与合规（通用领域）

主要组件:
    - BoundedContexts: 限界上下文
    - DomainModels: 领域模型
    - Aggregates: 聚合根
    - DomainEvents: 领域事件
    - Repositories: 仓储

最后修改日期: 2024-12-07
"""
from .bounded_contexts import *
from .domain_models import *
from .aggregates import *
from .domain_events import *
from .repositories import *
