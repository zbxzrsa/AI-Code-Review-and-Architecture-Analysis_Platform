"""
事件驱动架构 - 架构违规和依赖风险事件处理器

增强事件总线，添加架构相关事件
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import asyncio

logger = logging.getLogger(__name__)


class ArchitectureEventType(str, Enum):
    """架构事件类型"""
    ARCHITECTURAL_VIOLATION_DETECTED = "architectural_violation_detected"
    DEPENDENCY_RISK_DETECTED = "dependency_risk_detected"
    CIRCULAR_DEPENDENCY_FOUND = "circular_dependency_found"
    MODULE_BOUNDARY_VIOLATION = "module_boundary_violation"
    DATAFLOW_ANOMALY = "dataflow_anomaly"
    QUALITY_GATE_FAILED = "quality_gate_failed"
    KPI_THRESHOLD_BREACHED = "kpi_threshold_breached"
    ARCHITECTURE_DRIFT_DETECTED = "architecture_drift_detected"


@dataclass
class ArchitectureEvent:
    """架构事件"""
    event_id: str
    event_type: ArchitectureEventType
    timestamp: datetime
    source: str
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None
    
    # 事件数据
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 关联信息
    correlation_id: Optional[str] = None
    affected_components: List[str] = field(default_factory=list)
    severity: str = "medium"  # critical, high, medium, low


class ArchitectureEventBus:
    """
    架构事件总线
    
    处理架构相关事件：
    - 架构违规检测
    - 依赖风险
    - 质量门控失败
    - KPI阈值突破
    """
    
    def __init__(self, base_event_bus=None):
        """
        初始化架构事件总线
        
        Args:
            base_event_bus: 基础事件总线（如果已有）
        """
        self.base_event_bus = base_event_bus
        self.handlers: Dict[ArchitectureEventType, List] = {}
        self.event_history: List[ArchitectureEvent] = []
        self.max_history = 10000
    
    def subscribe(
        self,
        event_type: ArchitectureEventType,
        handler: callable
    ) -> str:
        """
        订阅架构事件
        
        Args:
            event_type: 事件类型
            handler: 异步处理函数
        
        Returns:
            处理器ID
        """
        import uuid
        handler_id = str(uuid.uuid4())
        
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        
        self.handlers[event_type].append({
            "id": handler_id,
            "handler": handler
        })
        
        logger.info(f"Subscribed to {event_type.value} with handler {handler_id}")
        return handler_id
    
    async def emit(
        self,
        event_type: ArchitectureEventType,
        data: Dict[str, Any],
        source: str,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        发布架构事件
        
        Args:
            event_type: 事件类型
            data: 事件数据
            source: 事件源
            tenant_id: 租户ID
            project_id: 项目ID
            correlation_id: 关联ID
        
        Returns:
            事件ID
        """
        import uuid
        event_id = str(uuid.uuid4())
        
        event = ArchitectureEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            source=source,
            tenant_id=tenant_id,
            project_id=project_id,
            data=data,
            correlation_id=correlation_id or str(uuid.uuid4()),
            affected_components=data.get("affected_components", []),
            severity=data.get("severity", "medium")
        )
        
        # 存储到历史
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # 调用处理器
        handlers = self.handlers.get(event_type, [])
        if handlers:
            tasks = [self._safe_call_handler(h["handler"], event) for h in handlers]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        # 如果存在基础事件总线，也发布到那里
        if self.base_event_bus:
            await self.base_event_bus.emit(
                event_type.value,
                {
                    **data,
                    "architecture_event": True,
                    "event_id": event_id
                },
                source
            )
        
        logger.info(
            f"Emitted architecture event {event_type.value}",
            extra={
                "event_id": event_id,
                "source": source,
                "tenant_id": tenant_id
            }
        )
        
        return event_id
    
    async def _safe_call_handler(
        self,
        handler: callable,
        event: ArchitectureEvent
    ) -> None:
        """安全调用处理器"""
        try:
            if asyncio.iscoroutinefunction(handler):
                await handler(event)
            else:
                handler(event)
        except Exception as e:
            logger.error(
                f"Handler error for event {event.event_id}: {e}",
                exc_info=True
            )
    
    def get_event_history(
        self,
        event_type: Optional[ArchitectureEventType] = None,
        tenant_id: Optional[str] = None,
        limit: int = 100
    ) -> List[ArchitectureEvent]:
        """获取事件历史"""
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if tenant_id:
            events = [e for e in events if e.tenant_id == tenant_id]
        
        return events[-limit:]


# =============================================================================
# 预定义的事件处理器
# =============================================================================

class ArchitectureViolationHandler:
    """架构违规处理器"""
    
    def __init__(self, notification_service=None, audit_logger=None):
        self.notification_service = notification_service
        self.audit_logger = audit_logger
    
    async def handle(self, event: ArchitectureEvent) -> None:
        """处理架构违规事件"""
        logger.warning(
            f"Architectural violation detected: {event.data.get('violation_type')}",
            extra={
                "event_id": event.event_id,
                "tenant_id": event.tenant_id,
                "affected_components": event.affected_components
            }
        )
        
        # 发送通知
        if self.notification_service:
            await self.notification_service.send_alert(
                f"架构违规检测: {event.data.get('violation_type')}",
                event.data,
                event.tenant_id
            )
        
        # 记录审计
        if self.audit_logger:
            await self.audit_logger.log(
                action="architectural_violation_detected",
                entity_type="architecture",
                entity_id=event.event_id,
                tenant_id=event.tenant_id,
                changes=event.data
            )


class DependencyRiskHandler:
    """依赖风险处理器"""
    
    async def handle(self, event: ArchitectureEvent) -> None:
        """处理依赖风险事件"""
        logger.warning(
            f"Dependency risk detected: {event.data.get('risk_type')}",
            extra={
                "event_id": event.event_id,
                "tenant_id": event.tenant_id
            }
        )
        
        # TODO: 触发依赖分析
        # TODO: 更新风险评分
        # TODO: 发送通知


class QualityGateFailureHandler:
    """质量门控失败处理器"""
    
    def __init__(self, governance_service=None):
        self.governance_service = governance_service
    
    async def handle(self, event: ArchitectureEvent) -> None:
        """处理质量门控失败事件"""
        gate_id = event.data.get("gate_id")
        failed_checks = event.data.get("failed_checks", [])
        
        logger.error(
            f"Quality gate failed: {gate_id}",
            extra={
                "event_id": event.event_id,
                "gate_id": gate_id,
                "failed_checks": failed_checks,
                "tenant_id": event.tenant_id
            }
        )
        
        # TODO: 触发自动修复或回滚
        # TODO: 通知相关人员
        # TODO: 更新治理仪表板


class KPIThresholdBreachHandler:
    """KPI阈值突破处理器"""
    
    async def handle(self, event: ArchitectureEvent) -> None:
        """处理KPI阈值突破事件"""
        kpi_type = event.data.get("kpi_type")
        current_value = event.data.get("current_value")
        threshold = event.data.get("threshold")
        
        logger.warning(
            f"KPI threshold breached: {kpi_type} = {current_value} (threshold: {threshold})",
            extra={
                "event_id": event.event_id,
                "kpi_type": kpi_type,
                "current_value": current_value,
                "threshold": threshold,
                "tenant_id": event.tenant_id
            }
        )
        
        # TODO: 触发质量门控重新评估
        # TODO: 更新KPI仪表板
        # TODO: 发送告警

