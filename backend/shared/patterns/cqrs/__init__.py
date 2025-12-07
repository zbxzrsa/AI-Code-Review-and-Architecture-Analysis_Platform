"""
CQRS 命令查询职责分离模式实现 (CQRS Pattern Implementation)

模块功能描述:
    分离读写操作以提高可扩展性和性能。

主要功能:
    - 命令处理（写操作）
    - 查询处理（读操作）
    - 事件溯源
    - 读模型投影

主要组件:
    - Commands: 命令处理器
    - Queries: 查询处理器
    - EventSourcing: 事件溯源
    - ReadModels: 读模型

最后修改日期: 2024-12-07
"""
from .commands import *
from .queries import *
from .event_sourcing import *
from .read_models import *
