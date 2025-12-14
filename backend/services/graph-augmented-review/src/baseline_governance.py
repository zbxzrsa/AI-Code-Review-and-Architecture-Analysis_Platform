"""
持续基线治理系统

功能：
- 租户感知的KPI跟踪
- 质量门控
- 审计追踪
- 合规性检查
"""

from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class KPIType(str, Enum):
    """KPI类型"""
    COMPLEXITY = "complexity"
    COVERAGE = "coverage"
    SECURITY_POSTURE = "security_posture"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    DOCUMENTATION = "documentation"


class GateStatus(str, Enum):
    """质量门控状态"""
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class KPIMetric:
    """KPI指标"""
    kpi_type: KPIType
    value: float
    threshold: float
    unit: str
    timestamp: datetime
    tenant_id: Optional[str] = None
    project_id: Optional[str] = None


@dataclass
class QualityGate:
    """质量门控"""
    gate_id: str
    name: str
    description: str
    kpi_checks: List[KPIType]
    thresholds: Dict[KPIType, float]
    tenant_id: Optional[str] = None
    enabled: bool = True
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GateResult:
    """门控结果"""
    gate_id: str
    status: GateStatus
    passed_checks: List[KPIType]
    failed_checks: List[KPIType]
    warnings: List[str]
    kpi_values: Dict[KPIType, float]
    timestamp: datetime
    tenant_id: Optional[str] = None
    audit_trail_id: Optional[str] = None


@dataclass
class AuditEntry:
    """审计条目"""
    entry_id: str
    action: str
    entity_type: str
    entity_id: str
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    changes: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class KPITracker(ABC):
    """KPI跟踪器抽象基类"""
    
    @abstractmethod
    async def track_metric(self, metric: KPIMetric) -> None:
        """跟踪指标"""
        pass
    
    @abstractmethod
    async def get_current_value(
        self,
        kpi_type: KPIType,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> Optional[float]:
        """获取当前值"""
        pass
    
    @abstractmethod
    async def get_historical_values(
        self,
        kpi_type: KPIType,
        start_date: datetime,
        end_date: datetime,
        tenant_id: Optional[str] = None
    ) -> List[KPIMetric]:
        """获取历史值"""
        pass


class BaselineGovernanceService:
    """
    持续基线治理服务
    
    功能：
    1. KPI跟踪（租户感知）
    2. 质量门控
    3. 审计追踪
    4. 合规性检查
    """
    
    def __init__(
        self,
        kpi_tracker: KPITracker,
        audit_logger=None,
        db_connection=None
    ):
        """
        初始化治理服务
        
        Args:
            kpi_tracker: KPI跟踪器
            audit_logger: 审计日志记录器
            db_connection: 数据库连接（用于持久化）
        """
        self.kpi_tracker = kpi_tracker
        self.audit_logger = audit_logger
        self.db = db_connection
        
        # 租户配置的质量门控
        self.tenant_gates: Dict[str, List[QualityGate]] = {}
        
        # 默认质量门控
        self.default_gates = self._create_default_gates()
    
    def _create_default_gates(self) -> List[QualityGate]:
        """创建默认质量门控"""
        return [
            QualityGate(
                gate_id="default_complexity",
                name="代码复杂度门控",
                description="检查代码复杂度是否超过阈值",
                kpi_checks=[KPIType.COMPLEXITY],
                thresholds={KPIType.COMPLEXITY: 10.0}  # 圈复杂度阈值
            ),
            QualityGate(
                gate_id="default_coverage",
                name="测试覆盖率门控",
                description="检查测试覆盖率是否达到要求",
                kpi_checks=[KPIType.COVERAGE],
                thresholds={KPIType.COVERAGE: 80.0}  # 80%覆盖率
            ),
            QualityGate(
                gate_id="default_security",
                name="安全态势门控",
                description="检查安全评分是否达标",
                kpi_checks=[KPIType.SECURITY_POSTURE],
                thresholds={KPIType.SECURITY_POSTURE: 0.9}  # 90%安全评分
            ),
            QualityGate(
                gate_id="comprehensive",
                name="综合质量门控",
                description="检查所有关键KPI",
                kpi_checks=[
                    KPIType.COMPLEXITY,
                    KPIType.COVERAGE,
                    KPIType.SECURITY_POSTURE,
                    KPIType.MAINTAINABILITY
                ],
                thresholds={
                    KPIType.COMPLEXITY: 10.0,
                    KPIType.COVERAGE: 80.0,
                    KPIType.SECURITY_POSTURE: 0.9,
                    KPIType.MAINTAINABILITY: 0.8
                }
            )
        ]
    
    async def evaluate_quality_gate(
        self,
        gate_id: str,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None,
        use_tenant_config: bool = True
    ) -> GateResult:
        """
        评估质量门控
        
        Args:
            gate_id: 门控ID
            tenant_id: 租户ID（用于租户特定配置）
            project_id: 项目ID
            use_tenant_config: 是否使用租户配置
        
        Returns:
            GateResult: 门控结果
        """
        # 获取门控配置
        gate = self._get_gate(gate_id, tenant_id, use_tenant_config)
        if not gate:
            raise ValueError(f"Quality gate not found: {gate_id}")
        
        if not gate.enabled:
            return GateResult(
                gate_id=gate_id,
                status=GateStatus.PENDING,
                passed_checks=[],
                failed_checks=[],
                warnings=["门控已禁用"],
                kpi_values={},
                timestamp=datetime.now(timezone.utc),
                tenant_id=tenant_id
            )
        
        # 检查每个KPI
        passed_checks: List[KPIType] = []
        failed_checks: List[KPIType] = []
        warnings: List[str] = []
        kpi_values: Dict[KPIType, float] = {}
        
        for kpi_type in gate.kpi_checks:
            threshold = gate.thresholds.get(kpi_type)
            if threshold is None:
                warnings.append(f"KPI {kpi_type.value} 未设置阈值")
                continue
            
            # 获取当前值
            current_value = await self.kpi_tracker.get_current_value(
                kpi_type, tenant_id, project_id
            )
            
            if current_value is None:
                warnings.append(f"无法获取 {kpi_type.value} 的当前值")
                continue
            
            kpi_values[kpi_type] = current_value
            
            # 比较阈值（根据KPI类型判断是越高越好还是越低越好）
            if self._check_kpi_passes(kpi_type, current_value, threshold):
                passed_checks.append(kpi_type)
            else:
                failed_checks.append(kpi_type)
        
        # 确定状态
        if failed_checks:
            status = GateStatus.FAIL
        elif warnings:
            status = GateStatus.WARNING
        else:
            status = GateStatus.PASS
        
        # 记录审计
        audit_trail_id = None
        if self.audit_logger:
            audit_trail_id = await self._log_gate_evaluation(
                gate_id, status, tenant_id, project_id, kpi_values
            )
        
        return GateResult(
            gate_id=gate_id,
            status=status,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            warnings=warnings,
            kpi_values=kpi_values,
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            audit_trail_id=audit_trail_id
        )
    
    def _get_gate(
        self,
        gate_id: str,
        tenant_id: Optional[str],
        use_tenant_config: bool
    ) -> Optional[QualityGate]:
        """获取门控配置"""
        # 优先使用租户配置
        if use_tenant_config and tenant_id and tenant_id in self.tenant_gates:
            for gate in self.tenant_gates[tenant_id]:
                if gate.gate_id == gate_id:
                    return gate
        
        # 使用默认配置
        for gate in self.default_gates:
            if gate.gate_id == gate_id:
                return gate
        
        return None
    
    def _check_kpi_passes(
        self,
        kpi_type: KPIType,
        current_value: float,
        threshold: float
    ) -> bool:
        """
        检查KPI是否通过
        
        根据KPI类型判断是越高越好还是越低越好
        """
        # 复杂度：越低越好
        if kpi_type == KPIType.COMPLEXITY:
            return current_value <= threshold
        
        # 覆盖率、安全态势、可维护性、文档：越高越好
        elif kpi_type in [
            KPIType.COVERAGE,
            KPIType.SECURITY_POSTURE,
            KPIType.MAINTAINABILITY,
            KPIType.DOCUMENTATION
        ]:
            return current_value >= threshold
        
        # 性能：根据具体指标判断（这里简化处理）
        elif kpi_type == KPIType.PERFORMANCE:
            return current_value >= threshold
        
        return False
    
    async def _log_gate_evaluation(
        self,
        gate_id: str,
        status: GateStatus,
        tenant_id: Optional[str],
        project_id: Optional[str],
        kpi_values: Dict[KPIType, float]
    ) -> str:
        """记录门控评估到审计日志"""
        if not self.audit_logger:
            return ""
        
        entry = AuditEntry(
            entry_id=f"gate_{gate_id}_{datetime.now(timezone.utc).isoformat()}",
            action="quality_gate_evaluation",
            entity_type="quality_gate",
            entity_id=gate_id,
            tenant_id=tenant_id,
            changes={
                "status": status.value,
                "kpi_values": {k.value: v for k, v in kpi_values.items()},
                "project_id": project_id
            }
        )
        
        # TODO: 实际记录到审计日志
        return entry.entry_id
    
    async def track_kpi(
        self,
        kpi_type: KPIType,
        value: float,
        tenant_id: Optional[str] = None,
        project_id: Optional[str] = None
    ) -> None:
        """跟踪KPI指标"""
        metric = KPIMetric(
            kpi_type=kpi_type,
            value=value,
            threshold=0.0,  # 阈值在门控中定义
            unit=self._get_kpi_unit(kpi_type),
            timestamp=datetime.now(timezone.utc),
            tenant_id=tenant_id,
            project_id=project_id
        )
        
        await self.kpi_tracker.track_metric(metric)
    
    def _get_kpi_unit(self, kpi_type: KPIType) -> str:
        """获取KPI单位"""
        units = {
            KPIType.COMPLEXITY: "cyclomatic_complexity",
            KPIType.COVERAGE: "percentage",
            KPIType.SECURITY_POSTURE: "score",
            KPIType.PERFORMANCE: "score",
            KPIType.MAINTAINABILITY: "score",
            KPIType.DOCUMENTATION: "score"
        }
        return units.get(kpi_type, "unit")
    
    async def configure_tenant_gate(
        self,
        tenant_id: str,
        gate: QualityGate,
        user_id: Optional[str] = None
    ) -> None:
        """
        配置租户特定的质量门控
        
        Args:
            tenant_id: 租户ID
            gate: 质量门控配置
            user_id: 操作用户ID（用于审计）
        """
        if tenant_id not in self.tenant_gates:
            self.tenant_gates[tenant_id] = []
        
        # 检查是否已存在
        existing_index = None
        for i, existing_gate in enumerate(self.tenant_gates[tenant_id]):
            if existing_gate.gate_id == gate.gate_id:
                existing_index = i
                break
        
        if existing_index is not None:
            self.tenant_gates[tenant_id][existing_index] = gate
        else:
            self.tenant_gates[tenant_id].append(gate)
        
        # 记录审计
        if self.audit_logger:
            await self._log_tenant_gate_config(tenant_id, gate, user_id)
    
    async def _log_tenant_gate_config(
        self,
        tenant_id: str,
        gate: QualityGate,
        user_id: Optional[str]
    ) -> None:
        """记录租户门控配置到审计日志"""
        # TODO: 实现审计日志记录
        pass
    
    async def get_tenant_kpi_dashboard(
        self,
        tenant_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        获取租户KPI仪表板数据
        
        Args:
            tenant_id: 租户ID
            days: 查询天数
        
        Returns:
            仪表板数据
        """
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days)
        
        dashboard = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "kpis": {}
        }
        
        # 获取每个KPI的历史数据
        for kpi_type in KPIType:
            historical = await self.kpi_tracker.get_historical_values(
                kpi_type, start_date, end_date, tenant_id
            )
            
            if historical:
                values = [m.value for m in historical]
                dashboard["kpis"][kpi_type.value] = {
                    "current": values[-1] if values else None,
                    "average": sum(values) / len(values) if values else None,
                    "min": min(values) if values else None,
                    "max": max(values) if values else None,
                    "trend": self._calculate_trend(values),
                    "data_points": len(values)
                }
        
        return dashboard
    
    def _calculate_trend(self, values: List[float]) -> str:
        """计算趋势（上升/下降/稳定）"""
        if len(values) < 2:
            return "insufficient_data"
        
        # 简单线性回归斜率
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"

