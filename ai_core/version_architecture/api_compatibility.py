"""
API兼容性模块 (API Compatibility Module)

确保所有版本保持API兼容性，实现无缝的用户过渡。
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from enum import Enum

logger = logging.getLogger(__name__)


class APIVersion(str, Enum):
    """API版本"""
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"


@dataclass
class APIEndpoint:
    """API端点定义"""
    path: str
    method: str  # GET, POST, PUT, DELETE
    version: APIVersion
    parameters: Dict[str, Any] = field(default_factory=dict)
    response_schema: Dict[str, Any] = field(default_factory=dict)
    deprecated: bool = False
    deprecated_since: Optional[str] = None
    replacement: Optional[str] = None


@dataclass
class APIContract:
    """API契约"""
    endpoint: APIEndpoint
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]
    error_schema: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0"


class CompatibilityLayer:
    """
    兼容性层
    
    在不同版本之间提供API兼容性转换。
    """
    
    def __init__(self):
        self.adapters: Dict[str, Callable] = {}  # version_pair -> adapter function
    
    def register_adapter(
        self,
        source_version: str,
        target_version: str,
        adapter: Callable[[Dict[str, Any]], Dict[str, Any]]
    ):
        """注册适配器"""
        key = f"{source_version}_to_{target_version}"
        self.adapters[key] = adapter
        logger.info(f"Registered adapter: {key}")
    
    def adapt_request(
        self,
        source_version: str,
        target_version: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """适配请求"""
        if source_version == target_version:
            return request_data
        
        key = f"{source_version}_to_{target_version}"
        adapter = self.adapters.get(key)
        
        if adapter:
            return adapter(request_data)
        else:
            logger.warning(f"No adapter found for {key}, using original request")
            return request_data
    
    def adapt_response(
        self,
        source_version: str,
        target_version: str,
        response_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """适配响应"""
        if source_version == target_version:
            return response_data
        
        key = f"{target_version}_to_{source_version}"
        adapter = self.adapters.get(key)
        
        if adapter:
            return adapter(response_data)
        else:
            logger.warning(f"No adapter found for {key}, using original response")
            return response_data


@dataclass
class BackwardCompatibilityCheck:
    """向后兼容性检查结果"""
    endpoint: str
    version: str
    compatible: bool
    issues: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BackwardCompatibilityValidator:
    """
    向后兼容性验证器
    
    验证新版本API是否向后兼容旧版本。
    """
    
    def __init__(self):
        self.api_registry: Dict[str, List[APIEndpoint]] = {}  # version -> endpoints
        self.compatibility_history: List[BackwardCompatibilityCheck] = []
    
    def register_endpoint(self, endpoint: APIEndpoint):
        """注册API端点"""
        version = endpoint.version.value
        if version not in self.api_registry:
            self.api_registry[version] = []
        self.api_registry[version].append(endpoint)
        logger.info(f"Registered endpoint: {endpoint.path} ({version})")
    
    def validate_backward_compatibility(
        self,
        new_version: str,
        old_version: str
    ) -> List[BackwardCompatibilityCheck]:
        """
        验证向后兼容性
        
        Args:
            new_version: 新版本
            old_version: 旧版本
            
        Returns:
            兼容性检查结果列表
        """
        new_endpoints = self.api_registry.get(new_version, [])
        old_endpoints = self.api_registry.get(old_version, [])
        
        # 创建旧版本端点映射
        old_endpoint_map = {
            (ep.path, ep.method): ep
            for ep in old_endpoints
        }
        
        results = []
        
        # 检查每个新端点是否与旧版本兼容
        for new_ep in new_endpoints:
            key = (new_ep.path, new_ep.method)
            old_ep = old_endpoint_map.get(key)
            
            if old_ep:
                # 端点存在，检查参数和响应是否兼容
                compatible, issues = self._check_endpoint_compatibility(new_ep, old_ep)
            else:
                # 新端点，检查是否标记为deprecated
                compatible = True
                issues = []
                if not new_ep.deprecated:
                    issues.append(f"New endpoint {new_ep.path} not found in {old_version}")
            
            result = BackwardCompatibilityCheck(
                endpoint=new_ep.path,
                version=new_version,
                compatible=compatible,
                issues=issues
            )
            results.append(result)
        
        # 检查是否有旧端点被移除
        new_endpoint_map = {
            (ep.path, ep.method): ep
            for ep in new_endpoints
        }
        
        for old_ep in old_endpoints:
            key = (old_ep.path, old_ep.method)
            if key not in new_endpoint_map:
                result = BackwardCompatibilityCheck(
                    endpoint=old_ep.path,
                    version=new_version,
                    compatible=False,
                    issues=[f"Endpoint removed in {new_version}"]
                )
                results.append(result)
        
        self.compatibility_history.extend(results)
        return results
    
    def _check_endpoint_compatibility(
        self,
        new_ep: APIEndpoint,
        old_ep: APIEndpoint
    ) -> tuple:
        """检查端点兼容性"""
        compatible = True
        issues = []
        
        # 检查参数兼容性
        new_params = set(new_ep.parameters.keys())
        old_params = set(old_ep.parameters.keys())
        
        # 新版本不能移除必需参数
        removed_params = old_params - new_params
        if removed_params:
            compatible = False
            issues.append(f"Removed parameters: {removed_params}")
        
        # 检查响应兼容性（简化检查）
        # 实际应该进行更详细的schema比较
        
        return compatible, issues


@dataclass
class VersionTransition:
    """版本转换记录"""
    transition_id: str
    from_version: str
    to_version: str
    transition_type: str  # upgrade, rollback, migration
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    api_changes: List[Dict[str, Any]] = field(default_factory=list)
    compatibility_issues: List[str] = field(default_factory=list)


class VersionTransitionManager:
    """
    版本转换管理器
    
    管理版本之间的转换，确保API兼容性。
    """
    
    def __init__(
        self,
        compatibility_validator: BackwardCompatibilityValidator,
        compatibility_layer: CompatibilityLayer
    ):
        self.validator = compatibility_validator
        self.compatibility_layer = compatibility_layer
        self.transitions: Dict[str, VersionTransition] = {}
    
    def plan_transition(
        self,
        from_version: str,
        to_version: str,
        transition_type: str = "upgrade"
    ) -> VersionTransition:
        """规划版本转换"""
        transition_id = f"transition_{from_version}_to_{to_version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # 验证向后兼容性
        compatibility_results = self.validator.validate_backward_compatibility(
            to_version, from_version
        )
        
        incompatible_endpoints = [
            r for r in compatibility_results if not r.compatible
        ]
        
        transition = VersionTransition(
            transition_id=transition_id,
            from_version=from_version,
            to_version=to_version,
            transition_type=transition_type,
            started_at=datetime.now(timezone.utc),
            status="pending",
            compatibility_issues=[
                issue for r in incompatible_endpoints
                for issue in r.issues
            ]
        )
        
        self.transitions[transition_id] = transition
        
        logger.info(
            f"Planned transition {transition_id}: "
            f"{from_version} -> {to_version} "
            f"({len(incompatible_endpoints)} incompatible endpoints)"
        )
        
        return transition
    
    async def execute_transition(
        self,
        transition_id: str
    ) -> VersionTransition:
        """执行版本转换"""
        transition = self.transitions.get(transition_id)
        if not transition:
            raise ValueError(f"Transition not found: {transition_id}")
        
        transition.status = "in_progress"
        
        try:
            # 检查兼容性问题
            if transition.compatibility_issues:
                logger.warning(
                    f"Transition {transition_id} has compatibility issues: "
                    f"{transition.compatibility_issues}"
                )
                # 在实际实现中，这里应该决定是否继续或中止
            
            # 执行转换步骤
            # 1. 备份当前版本
            # 2. 部署新版本
            # 3. 验证新版本
            # 4. 切换流量
            # 5. 监控新版本
            
            # 模拟转换过程
            await asyncio.sleep(1)  # 模拟转换时间
            
            transition.status = "completed"
            transition.completed_at = datetime.now(timezone.utc)
            
            logger.info(f"Transition {transition_id} completed successfully")
            
        except Exception as e:
            transition.status = "failed"
            transition.completed_at = datetime.now(timezone.utc)
            logger.error(f"Transition {transition_id} failed: {e}")
        
        return transition


class APIVersionManager:
    """
    API版本管理器
    
    管理所有版本的API定义和兼容性。
    """
    
    def __init__(
        self,
        compatibility_validator: BackwardCompatibilityValidator,
        compatibility_layer: CompatibilityLayer,
        transition_manager: VersionTransitionManager
    ):
        self.validator = compatibility_validator
        self.compatibility_layer = compatibility_layer
        self.transition_manager = transition_manager
        self.current_version = "v2"
    
    def register_api(
        self,
        endpoint: APIEndpoint,
        contract: APIContract
    ):
        """注册API"""
        self.validator.register_endpoint(endpoint)
        logger.info(f"Registered API: {endpoint.path} ({endpoint.version.value})")
    
    def handle_request(
        self,
        endpoint_path: str,
        method: str,
        request_data: Dict[str, Any],
        client_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理API请求
        
        Args:
            endpoint_path: 端点路径
            method: HTTP方法
            request_data: 请求数据
            client_version: 客户端版本（如果指定）
            
        Returns:
            响应数据
        """
        # 确定目标版本（通常是当前版本）
        target_version = self.current_version
        
        # 如果客户端指定了版本，适配请求
        if client_version and client_version != target_version:
            request_data = self.compatibility_layer.adapt_request(
                client_version,
                target_version,
                request_data
            )
        
        # 处理请求（这里应该调用实际的API处理逻辑）
        response_data = {
            "status": "success",
            "data": request_data
        }
        
        # 如果客户端指定了版本，适配响应
        if client_version and client_version != target_version:
            response_data = self.compatibility_layer.adapt_response(
                client_version,
                target_version,
                response_data
            )
        
        return response_data
    
    async def upgrade_version(
        self,
        new_version: str
    ) -> VersionTransition:
        """升级版本"""
        transition = self.transition_manager.plan_transition(
            self.current_version,
            new_version,
            transition_type="upgrade"
        )
        
        # 执行转换
        transition = await self.transition_manager.execute_transition(
            transition.transition_id
        )
        
        if transition.status == "completed":
            self.current_version = new_version
            logger.info(f"Version upgraded to {new_version}")
        
        return transition
    
    def get_compatibility_report(
        self,
        from_version: str,
        to_version: str
    ) -> Dict[str, Any]:
        """获取兼容性报告"""
        results = self.validator.validate_backward_compatibility(
            to_version, from_version
        )
        
        return {
            "from_version": from_version,
            "to_version": to_version,
            "total_endpoints": len(results),
            "compatible_endpoints": len([r for r in results if r.compatible]),
            "incompatible_endpoints": len([r for r in results if not r.compatible]),
            "details": [r.__dict__ for r in results]
        }

