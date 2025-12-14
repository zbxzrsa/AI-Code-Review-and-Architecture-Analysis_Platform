"""
资源调度模块 (Resource Scheduler Module)

实现动态资源调度机制：
- V3作为备份计算资源，参与V1的测试和技术分析
- 优先保证V1的测试需求、技术集成、问题解决
- 优先保证V2的稳定运行
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class ResourceType(str, Enum):
    """资源类型"""
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"
    GPU = "gpu"


class TaskPriority(str, Enum):
    """任务优先级"""
    CRITICAL = "critical"  # V2稳定运行
    HIGH = "high"          # V1测试、技术集成、问题解决
    MEDIUM = "medium"      # V1常规实验
    LOW = "low"            # V3技术分析、参数对比


@dataclass
class ResourceAllocation:
    """资源分配"""
    version: str
    resource_type: ResourceType
    allocated: float
    requested: float
    priority: TaskPriority
    allocated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ComputeResource:
    """计算资源"""
    resource_id: str
    resource_type: ResourceType
    total_capacity: float
    available_capacity: float
    allocated_capacity: float = 0.0
    
    @property
    def utilization_rate(self) -> float:
        """利用率"""
        if self.total_capacity == 0:
            return 0.0
        return self.allocated_capacity / self.total_capacity


class ComputeResourcePool:
    """
    计算资源池
    
    管理所有可用的计算资源。
    """
    
    def __init__(self):
        self.resources: Dict[str, ComputeResource] = {}
        self.allocations: Dict[str, List[ResourceAllocation]] = {}  # version -> allocations
    
    def add_resource(self, resource: ComputeResource):
        """添加资源"""
        self.resources[resource.resource_id] = resource
        logger.info(
            f"Added resource {resource.resource_id}: "
            f"{resource.resource_type} ({resource.total_capacity})"
        )
    
    def allocate_resource(
        self,
        version: str,
        resource_type: ResourceType,
        amount: float,
        priority: TaskPriority
    ) -> bool:
        """
        分配资源
        
        Returns:
            是否分配成功
        """
        # 查找可用资源
        available_resources = [
            r for r in self.resources.values()
            if r.resource_type == resource_type and
            (r.available_capacity - r.allocated_capacity) >= amount
        ]
        
        if not available_resources:
            logger.warning(
                f"No available {resource_type} resources for version {version}"
            )
            return False
        
        # 选择资源（简单策略：选择可用容量最大的）
        resource = max(
            available_resources,
            key=lambda r: r.available_capacity - r.allocated_capacity
        )
        
        # 分配资源
        resource.allocated_capacity += amount
        
        allocation = ResourceAllocation(
            version=version,
            resource_type=resource_type,
            allocated=amount,
            requested=amount,
            priority=priority
        )
        
        if version not in self.allocations:
            self.allocations[version] = []
        self.allocations[version].append(allocation)
        
        logger.info(
            f"Allocated {amount} {resource_type} to version {version} "
            f"(priority: {priority})"
        )
        
        return True
    
    def release_resource(
        self,
        version: str,
        resource_type: ResourceType,
        amount: float
    ):
        """释放资源"""
        if version not in self.allocations:
            return
        
        # 查找对应的资源并释放
        for resource in self.resources.values():
            if resource.resource_type == resource_type:
                resource.allocated_capacity = max(0, resource.allocated_capacity - amount)
                break
        
        # 从分配记录中移除
        self.allocations[version] = [
            a for a in self.allocations[version]
            if not (a.resource_type == resource_type and a.allocated == amount)
        ]
        
        logger.info(f"Released {amount} {resource_type} from version {version}")
    
    def get_available_capacity(self, resource_type: ResourceType) -> float:
        """获取可用容量"""
        total_available = 0.0
        for resource in self.resources.values():
            if resource.resource_type == resource_type:
                total_available += (
                    resource.available_capacity - resource.allocated_capacity
                )
        return total_available


@dataclass
class ResourceAllocationPolicy:
    """
    资源分配策略
    
    定义不同版本的资源分配优先级和规则。
    """
    # V2稳定版优先级最高
    v2_min_guaranteed: Dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.CPU: 4.0,
        ResourceType.MEMORY: 8.0,  # 8Gi
        ResourceType.STORAGE: 50.0,  # 50Gi
    })
    
    # V1开发版优先级中等，但保证测试需求
    v1_min_guaranteed: Dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.CPU: 2.0,
        ResourceType.MEMORY: 4.0,  # 4Gi
        ResourceType.STORAGE: 20.0,  # 20Gi
    })
    
    # V3基准版优先级最低，作为备份
    v3_min_guaranteed: Dict[ResourceType, float] = field(default_factory=lambda: {
        ResourceType.CPU: 1.0,
        ResourceType.MEMORY: 2.0,  # 2Gi
        ResourceType.STORAGE: 10.0,  # 10Gi
    })
    
    # 动态调整规则
    v1_test_boost_multiplier: float = 1.5  # V1测试时资源提升50%
    v3_backup_utilization_max: float = 0.7  # V3作为备份时最大利用率70%


class PriorityBasedScheduler:
    """
    基于优先级的调度器
    
    根据任务优先级分配资源。
    """
    
    def __init__(
        self,
        resource_pool: ComputeResourcePool,
        allocation_policy: ResourceAllocationPolicy
    ):
        self.resource_pool = resource_pool
        self.policy = allocation_policy
        self.pending_requests: List[Dict[str, Any]] = []
    
    def request_resources(
        self,
        version: str,
        resource_type: ResourceType,
        amount: float,
        priority: TaskPriority,
        task_id: Optional[str] = None
    ) -> bool:
        """
        请求资源
        
        Returns:
            是否分配成功
        """
        # 根据版本和优先级确定实际优先级
        actual_priority = self._determine_actual_priority(version, priority)
        
        # 检查是否有足够资源
        available = self.resource_pool.get_available_capacity(resource_type)
        
        # 如果资源不足，检查是否可以抢占低优先级任务
        if available < amount:
            if actual_priority in [TaskPriority.CRITICAL, TaskPriority.HIGH]:
                # 尝试抢占低优先级资源
                if self._preempt_low_priority_resources(resource_type, amount):
                    available = self.resource_pool.get_available_capacity(resource_type)
        
        if available >= amount:
            return self.resource_pool.allocate_resource(
                version, resource_type, amount, actual_priority
            )
        else:
            # 加入待处理队列
            self.pending_requests.append({
                "version": version,
                "resource_type": resource_type,
                "amount": amount,
                "priority": actual_priority,
                "task_id": task_id,
                "requested_at": datetime.now(timezone.utc)
            })
            logger.warning(
                f"Resource request queued for {version}: "
                f"{amount} {resource_type} (priority: {actual_priority})"
            )
            return False
    
    def _determine_actual_priority(
        self,
        version: str,
        priority: TaskPriority
    ) -> TaskPriority:
        """确定实际优先级"""
        # V2的任务始终是CRITICAL
        if version == "v2":
            return TaskPriority.CRITICAL
        
        # V1的测试、技术集成、问题解决是HIGH
        if version == "v1" and priority in [
            TaskPriority.HIGH,
            TaskPriority.CRITICAL
        ]:
            return TaskPriority.HIGH
        
        # V3的任务通常是LOW
        if version == "v3":
            return TaskPriority.LOW
        
        return priority
    
    def _preempt_low_priority_resources(
        self,
        resource_type: ResourceType,
        required_amount: float
    ) -> bool:
        """抢占低优先级资源"""
        # 查找低优先级的分配
        low_priority_versions = ["v3"]
        
        for version in low_priority_versions:
            allocations = self.resource_pool.allocations.get(version, [])
            for allocation in allocations:
                if (
                    allocation.resource_type == resource_type and
                    allocation.priority == TaskPriority.LOW
                ):
                    # 释放资源
                    self.resource_pool.release_resource(
                        version,
                        resource_type,
                        allocation.allocated
                    )
                    
                    if self.resource_pool.get_available_capacity(resource_type) >= required_amount:
                        return True
        
        return False
    
    async def process_pending_requests(self):
        """处理待处理的资源请求"""
        while self.pending_requests:
            request = self.pending_requests[0]
            
            available = self.resource_pool.get_available_capacity(
                request["resource_type"]
            )
            
            if available >= request["amount"]:
                success = self.resource_pool.allocate_resource(
                    request["version"],
                    request["resource_type"],
                    request["amount"],
                    request["priority"]
                )
                
                if success:
                    self.pending_requests.pop(0)
                    logger.info(
                        f"Processed pending request for {request['version']}"
                    )
                else:
                    # 如果仍然失败，等待一段时间后重试
                    await asyncio.sleep(5)
            else:
                # 资源仍然不足，等待
                await asyncio.sleep(10)


class DynamicResourceScheduler:
    """
    动态资源调度器
    
    根据实际需求动态调整资源分配。
    """
    
    def __init__(
        self,
        resource_pool: ComputeResourcePool,
        allocation_policy: ResourceAllocationPolicy,
        scheduler: PriorityBasedScheduler
    ):
        self.resource_pool = resource_pool
        self.policy = allocation_policy
        self.scheduler = scheduler
        self.scheduling_enabled = True
    
    async def start_scheduling(self, interval_seconds: int = 60):
        """启动调度"""
        logger.info("Starting dynamic resource scheduling")
        
        while self.scheduling_enabled:
            try:
                # 处理待处理的请求
                await self.scheduler.process_pending_requests()
                
                # 动态调整资源分配
                await self._adjust_allocations()
                
            except Exception as e:
                logger.error(f"Error in resource scheduling: {e}")
            
            await asyncio.sleep(interval_seconds)
    
    async def _adjust_allocations(self):
        """调整资源分配"""
        # 检查V2的资源是否充足
        v2_allocations = self.resource_pool.allocations.get("v2", [])
        for resource_type, min_required in self.policy.v2_min_guaranteed.items():
            allocated = sum(
                a.allocated for a in v2_allocations
                if a.resource_type == resource_type
            )
            
            if allocated < min_required:
                # 确保V2有足够的资源
                self.scheduler.request_resources(
                    "v2",
                    resource_type,
                    min_required - allocated,
                    TaskPriority.CRITICAL
                )
        
        # 检查V1的资源（测试时提升）
        v1_allocations = self.resource_pool.allocations.get("v1", [])
        # 这里可以根据V1的实际任务类型动态调整
        
        # V3作为备份，限制其资源使用
        v3_allocations = self.resource_pool.allocations.get("v3", [])
        for allocation in v3_allocations:
            resource = next(
                (r for r in self.resource_pool.resources.values()
                 if r.resource_type == allocation.resource_type),
                None
            )
            if resource and resource.utilization_rate > self.policy.v3_backup_utilization_max:
                # 释放部分V3资源
                excess = (
                    resource.allocated_capacity -
                    resource.total_capacity * self.policy.v3_backup_utilization_max
                )
                if excess > 0:
                    self.resource_pool.release_resource(
                        "v3",
                        allocation.resource_type,
                        excess
                    )
    
    def stop_scheduling(self):
        """停止调度"""
        self.scheduling_enabled = False
        logger.info("Resource scheduling stopped")
    
    def get_resource_utilization_report(self) -> Dict[str, Any]:
        """获取资源利用率报告"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "versions": {}
        }
        
        for version in ["v1", "v2", "v3"]:
            allocations = self.resource_pool.allocations.get(version, [])
            version_report = {
                "total_allocated": {},
                "by_priority": {}
            }
            
            for allocation in allocations:
                resource_type = allocation.resource_type.value
                if resource_type not in version_report["total_allocated"]:
                    version_report["total_allocated"][resource_type] = 0.0
                version_report["total_allocated"][resource_type] += allocation.allocated
                
                priority = allocation.priority.value
                if priority not in version_report["by_priority"]:
                    version_report["by_priority"][priority] = {}
                if resource_type not in version_report["by_priority"][priority]:
                    version_report["by_priority"][priority][resource_type] = 0.0
                version_report["by_priority"][priority][resource_type] += allocation.allocated
            
            report["versions"][version] = version_report
        
        return report

