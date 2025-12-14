"""
监控和回滚模块 (Monitoring and Rollback Module)

实现实时监控和分钟级回滚机制：
- 实时监控每个版本的性能指标、错误率、用户反馈
- 建立分钟级回滚计划，确保V2异常时立即恢复
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """健康状态"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class PerformanceMetrics:
    """性能指标"""
    version: str
    timestamp: datetime
    request_count: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    cpu_usage_pct: float = 0.0
    memory_usage_pct: float = 0.0
    
    @property
    def error_rate(self) -> float:
        """错误率"""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.request_count == 0:
            return 0.0
        return self.success_count / self.request_count


@dataclass
class UserFeedback:
    """用户反馈"""
    feedback_id: str
    version: str
    user_id: str
    rating: int  # 1-5
    comment: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    issue_type: Optional[str] = None  # error, performance, feature, other


class RealTimeMetricsCollector:
    """
    实时指标收集器
    
    实时收集每个版本的性能指标、错误率等。
    """
    
    def __init__(self, window_size: int = 60):
        """
        初始化
        
        Args:
            window_size: 滑动窗口大小（秒）
        """
        self.window_size = window_size
        self.metrics_history: Dict[str, deque] = {}  # version -> metrics queue
        self.feedback_history: Dict[str, List[UserFeedback]] = {}  # version -> feedback list
    
    def collect_metrics(self, metrics: PerformanceMetrics):
        """收集性能指标"""
        version = metrics.version
        if version not in self.metrics_history:
            self.metrics_history[version] = deque(maxlen=self.window_size)
        
        self.metrics_history[version].append(metrics)
    
    def collect_feedback(self, feedback: UserFeedback):
        """收集用户反馈"""
        version = feedback.version
        if version not in self.feedback_history:
            self.feedback_history[version] = []
        
        self.feedback_history[version].append(feedback)
        
        # 只保留最近24小时的反馈
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
        self.feedback_history[version] = [
            f for f in self.feedback_history[version]
            if f.timestamp > cutoff_time
        ]
    
    def get_recent_metrics(
        self,
        version: str,
        minutes: int = 5
    ) -> List[PerformanceMetrics]:
        """获取最近的指标"""
        if version not in self.metrics_history:
            return []
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        return [
            m for m in self.metrics_history[version]
            if m.timestamp > cutoff_time
        ]
    
    def get_average_metrics(
        self,
        version: str,
        minutes: int = 5
    ) -> Optional[PerformanceMetrics]:
        """获取平均指标"""
        recent_metrics = self.get_recent_metrics(version, minutes)
        if not recent_metrics:
            return None
        
        total_requests = sum(m.request_count for m in recent_metrics)
        total_success = sum(m.success_count for m in recent_metrics)
        total_errors = sum(m.error_count for m in recent_metrics)
        total_latency = sum(m.avg_latency_ms * m.success_count for m in recent_metrics)
        total_success_count = sum(m.success_count for m in recent_metrics)
        
        avg_latency = total_latency / total_success_count if total_success_count > 0 else 0.0
        
        return PerformanceMetrics(
            version=version,
            timestamp=datetime.now(timezone.utc),
            request_count=total_requests,
            success_count=total_success,
            error_count=total_errors,
            avg_latency_ms=avg_latency
        )
    
    def get_user_feedback_summary(
        self,
        version: str,
        hours: int = 24
    ) -> Dict[str, Any]:
        """获取用户反馈摘要"""
        if version not in self.feedback_history:
            return {
                "total_feedback": 0,
                "avg_rating": 0.0,
                "issue_types": {}
            }
        
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_feedback = [
            f for f in self.feedback_history[version]
            if f.timestamp > cutoff_time
        ]
        
        if not recent_feedback:
            return {
                "total_feedback": 0,
                "avg_rating": 0.0,
                "issue_types": {}
            }
        
        avg_rating = sum(f.rating for f in recent_feedback) / len(recent_feedback)
        issue_types = {}
        for feedback in recent_feedback:
            issue_type = feedback.issue_type or "other"
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
        
        return {
            "total_feedback": len(recent_feedback),
            "avg_rating": avg_rating,
            "issue_types": issue_types
        }


@dataclass
class HealthCheckResult:
    """健康检查结果"""
    version: str
    status: HealthStatus
    timestamp: datetime
    metrics: Optional[PerformanceMetrics] = None
    issues: List[str] = field(default_factory=list)
    score: float = 0.0  # 0-100


class HealthCheckSystem:
    """
    健康检查系统
    
    实时检查每个版本的健康状态。
    """
    
    def __init__(
        self,
        metrics_collector: RealTimeMetricsCollector,
        error_rate_threshold: float = 0.02,  # 2%
        latency_threshold_ms: float = 3000,  # 3秒
        min_rating_threshold: float = 3.0  # 最低评分
    ):
        self.metrics_collector = metrics_collector
        self.error_rate_threshold = error_rate_threshold
        self.latency_threshold_ms = latency_threshold_ms
        self.min_rating_threshold = min_rating_threshold
        self.health_history: Dict[str, List[HealthCheckResult]] = {}
    
    def check_health(self, version: str) -> HealthCheckResult:
        """检查版本健康状态"""
        # 获取最近的指标
        metrics = self.metrics_collector.get_average_metrics(version, minutes=5)
        
        if not metrics:
            return HealthCheckResult(
                version=version,
                status=HealthStatus.UNKNOWN,
                timestamp=datetime.now(timezone.utc),
                issues=["No metrics available"]
            )
        
        # 获取用户反馈
        feedback_summary = self.metrics_collector.get_user_feedback_summary(version)
        
        # 评估健康状态
        issues = []
        score = 100.0
        
        # 检查错误率
        if metrics.error_rate > self.error_rate_threshold:
            issues.append(f"High error rate: {metrics.error_rate:.2%}")
            score -= 30
        
        # 检查延迟
        if metrics.avg_latency_ms > self.latency_threshold_ms:
            issues.append(f"High latency: {metrics.avg_latency_ms:.0f}ms")
            score -= 20
        
        # 检查用户反馈
        if feedback_summary["total_feedback"] > 0:
            avg_rating = feedback_summary["avg_rating"]
            if avg_rating < self.min_rating_threshold:
                issues.append(f"Low user rating: {avg_rating:.1f}")
                score -= 25
        
        # 确定状态
        if score >= 80:
            status = HealthStatus.HEALTHY
        elif score >= 60:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        result = HealthCheckResult(
            version=version,
            status=status,
            timestamp=datetime.now(timezone.utc),
            metrics=metrics,
            issues=issues,
            score=score
        )
        
        # 保存历史
        if version not in self.health_history:
            self.health_history[version] = []
        self.health_history[version].append(result)
        
        # 只保留最近100条记录
        if len(self.health_history[version]) > 100:
            self.health_history[version] = self.health_history[version][-100:]
        
        return result


@dataclass
class RollbackPlan:
    """回滚计划"""
    plan_id: str
    version: str
    target_version: str  # 回滚到的版本
    steps: List[Dict[str, Any]]
    estimated_duration_seconds: int
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class RollbackExecution:
    """回滚执行记录"""
    execution_id: str
    plan_id: str
    version: str
    status: str  # pending, in_progress, completed, failed
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None


class RollbackManager:
    """
    回滚管理器
    
    管理版本回滚操作，确保分钟级回滚能力。
    """
    
    def __init__(self, rollback_timeout_seconds: int = 60):
        """
        初始化
        
        Args:
            rollback_timeout_seconds: 回滚超时时间（秒），默认60秒（分钟级）
        """
        self.rollback_timeout_seconds = rollback_timeout_seconds
        self.rollback_plans: Dict[str, RollbackPlan] = {}
        self.rollback_executions: Dict[str, RollbackExecution] = {}
        self.version_snapshots: Dict[str, Dict[str, Any]] = {}  # version -> snapshot
    
    def create_rollback_plan(
        self,
        version: str,
        target_version: str
    ) -> RollbackPlan:
        """创建回滚计划"""
        plan_id = f"rollback_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 定义回滚步骤
        steps = [
            {
                "step": 1,
                "action": "stop_current_version",
                "description": f"停止当前版本 {version}",
                "timeout_seconds": 10
            },
            {
                "step": 2,
                "action": "restore_snapshot",
                "description": f"恢复目标版本 {target_version} 的快照",
                "timeout_seconds": 20
            },
            {
                "step": 3,
                "action": "start_target_version",
                "description": f"启动目标版本 {target_version}",
                "timeout_seconds": 15
            },
            {
                "step": 4,
                "action": "verify_health",
                "description": "验证目标版本健康状态",
                "timeout_seconds": 10
            },
            {
                "step": 5,
                "action": "update_routing",
                "description": "更新流量路由配置",
                "timeout_seconds": 5
            }
        ]
        
        estimated_duration = sum(step["timeout_seconds"] for step in steps)
        
        plan = RollbackPlan(
            plan_id=plan_id,
            version=version,
            target_version=target_version,
            steps=steps,
            estimated_duration_seconds=estimated_duration
        )
        
        self.rollback_plans[plan_id] = plan
        
        logger.info(f"Created rollback plan {plan_id} for version {version}")
        
        return plan
    
    async def execute_rollback(
        self,
        plan_id: str,
        health_check_callback: Optional[Callable[[str], HealthCheckResult]] = None
    ) -> RollbackExecution:
        """
        执行回滚
        
        Args:
            plan_id: 回滚计划ID
            health_check_callback: 健康检查回调函数
        """
        plan = self.rollback_plans.get(plan_id)
        if not plan:
            raise ValueError(f"Rollback plan not found: {plan_id}")
        
        execution = RollbackExecution(
            execution_id=f"exec_{plan_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            plan_id=plan_id,
            version=plan.version,
            status="in_progress",
            started_at=datetime.now(timezone.utc)
        )
        
        self.rollback_executions[execution.execution_id] = execution
        
        logger.info(f"Executing rollback plan {plan_id}")
        
        try:
            # 执行回滚步骤
            for step in plan.steps:
                logger.info(f"Executing step {step['step']}: {step['action']}")
                
                # 这里应该实现实际的回滚操作
                # 例如：停止服务、恢复快照、启动服务等
                
                await asyncio.sleep(0.1)  # 模拟操作时间
                
                # 检查超时
                elapsed = (datetime.now(timezone.utc) - execution.started_at).total_seconds()
                if elapsed > self.rollback_timeout_seconds:
                    raise TimeoutError(f"Rollback timeout after {elapsed} seconds")
            
            # 验证健康状态
            if health_check_callback:
                health_result = health_check_callback(plan.target_version)
                if health_result.status == HealthStatus.CRITICAL:
                    raise Exception("Target version health check failed")
            
            execution.status = "completed"
            execution.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Rollback completed successfully: {plan_id}")
            
        except Exception as e:
            execution.status = "failed"
            execution.error = str(e)
            execution.completed_at = datetime.now(timezone.utc)
            logger.error(f"Rollback failed: {e}")
        
        return execution
    
    def create_snapshot(self, version: str, snapshot_data: Dict[str, Any]):
        """创建版本快照"""
        self.version_snapshots[version] = snapshot_data
        logger.info(f"Created snapshot for version {version}")
    
    def get_snapshot(self, version: str) -> Optional[Dict[str, Any]]:
        """获取版本快照"""
        return self.version_snapshots.get(version)


class VersionMonitoringSystem:
    """
    版本监控系统
    
    综合监控系统，整合指标收集、健康检查和回滚管理。
    """
    
    def __init__(
        self,
        metrics_collector: RealTimeMetricsCollector,
        health_check_system: HealthCheckSystem,
        rollback_manager: RollbackManager
    ):
        self.metrics_collector = metrics_collector
        self.health_check_system = health_check_system
        self.rollback_manager = rollback_manager
        self.monitoring_enabled = True
        self.auto_rollback_enabled = True
        self.critical_thresholds = {
            "error_rate": 0.05,  # 5%
            "latency_ms": 10000,  # 10秒
            "health_score": 50.0
        }
    
    async def start_monitoring(self, versions: List[str], interval_seconds: int = 30):
        """
        启动监控
        
        Args:
            versions: 要监控的版本列表
            interval_seconds: 监控间隔（秒）
        """
        logger.info(f"Starting monitoring for versions: {versions}")
        
        while self.monitoring_enabled:
            for version in versions:
                try:
                    # 健康检查
                    health_result = self.health_check_system.check_health(version)
                    
                    logger.debug(
                        f"Version {version} health: {health_result.status} "
                        f"(score: {health_result.score:.1f})"
                    )
                    
                    # 检查是否需要自动回滚（仅对V2）
                    if (
                        version == "v2" and
                        self.auto_rollback_enabled and
                        health_result.status == HealthStatus.CRITICAL
                    ):
                        await self._trigger_auto_rollback(version, health_result)
                
                except Exception as e:
                    logger.error(f"Error monitoring version {version}: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def _trigger_auto_rollback(
        self,
        version: str,
        health_result: HealthCheckResult
    ):
        """触发自动回滚"""
        logger.warning(
            f"Critical health detected for {version}, triggering auto-rollback"
        )
        
        # 查找上一个稳定版本
        target_version = "v2_previous"  # 这里应该实现版本历史查找逻辑
        
        # 创建回滚计划
        plan = self.rollback_manager.create_rollback_plan(version, target_version)
        
        # 执行回滚
        execution = await self.rollback_manager.execute_rollback(
            plan.plan_id,
            health_check_callback=self.health_check_system.check_health
        )
        
        if execution.status == "completed":
            logger.info(f"Auto-rollback completed for {version}")
        else:
            logger.error(f"Auto-rollback failed for {version}: {execution.error}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_enabled = False
        logger.info("Monitoring stopped")


class MinuteLevelRollbackPlan:
    """
    分钟级回滚计划
    
    确保V2异常时能够在60秒内完成回滚。
    """
    
    def __init__(self, rollback_manager: RollbackManager):
        self.rollback_manager = rollback_manager
    
    def prepare_rollback_plan(self, version: str) -> RollbackPlan:
        """
        准备回滚计划
        
        预先创建回滚计划，确保在需要时能够快速执行。
        """
        # 假设回滚到上一个稳定版本
        target_version = f"{version}_stable_backup"
        
        plan = self.rollback_manager.create_rollback_plan(version, target_version)
        
        # 验证计划可以在60秒内完成
        if plan.estimated_duration_seconds > 60:
            logger.warning(
                f"Rollback plan estimated duration ({plan.estimated_duration_seconds}s) "
                f"exceeds 60 seconds"
            )
        
        return plan
    
    async def emergency_rollback(
        self,
        version: str,
        reason: str
    ) -> RollbackExecution:
        """
        紧急回滚
        
        在检测到严重问题时立即执行回滚。
        """
        logger.critical(f"Emergency rollback triggered for {version}: {reason}")
        
        # 创建并立即执行回滚计划
        plan = self.prepare_rollback_plan(version)
        
        execution = await self.rollback_manager.execute_rollback(plan.plan_id)
        
        return execution

