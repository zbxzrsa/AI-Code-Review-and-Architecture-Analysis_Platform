"""
安全通信通道模块 (Secure Communication Channel)

建立AI系统间的安全通信通道，确保数据隔离的同时允许必要的信息共享
"""

import asyncio
import logging
import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import hashlib
import hmac

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型"""
    TECH_COMPARISON = "tech_comparison"      # 技术对比数据
    ERROR_REPORT = "error_report"            # 错误报告
    FIX_SUGGESTION = "fix_suggestion"        # 修复建议
    PERFORMANCE_DATA = "performance_data"    # 性能数据
    STATUS_UPDATE = "status_update"          # 状态更新
    COLLABORATION_REQUEST = "collaboration_request"  # 协作请求


class SecurityLevel(str, Enum):
    """安全级别"""
    PUBLIC = "public"           # 公开：所有版本可访问
    RESTRICTED = "restricted"   # 受限：特定版本可访问
    CONFIDENTIAL = "confidential"  # 机密：仅授权版本可访问
    SECRET = "secret"           # 秘密：最高安全级别


@dataclass
class Message:
    """通信消息"""
    message_id: str
    message_type: MessageType
    source_version: str
    target_version: str
    security_level: SecurityLevel
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    signature: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CommunicationChannel:
    """通信通道配置"""
    channel_id: str
    source_version: str
    target_version: str
    allowed_message_types: List[MessageType]
    security_level: SecurityLevel
    encryption_enabled: bool = True
    authentication_required: bool = True
    rate_limit: Optional[int] = None  # 消息/秒
    enabled: bool = True


class SecureChannel:
    """
    安全通信通道
    
    管理AI系统间的安全通信，确保：
    - 数据隔离
    - 必要信息共享
    - 安全认证
    - 访问控制
    """
    
    def __init__(self, secret_key: Optional[str] = None):
        """
        初始化安全通道
        
        Args:
            secret_key: 用于签名的密钥
        """
        self.secret_key = secret_key or self._generate_secret_key()
        self.channels: Dict[str, CommunicationChannel] = {}
        self.message_queue: Dict[str, List[Message]] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = {}
        self.access_control: Dict[str, List[str]] = {}  # version -> allowed_versions
    
    def _generate_secret_key(self) -> str:
        """生成密钥"""
        return hashlib.sha256(
            f"secure_channel_{datetime.now().isoformat()}".encode()
        ).hexdigest()
    
    def create_channel(
        self,
        source_version: str,
        target_version: str,
        allowed_message_types: List[MessageType],
        security_level: SecurityLevel = SecurityLevel.RESTRICTED
    ) -> CommunicationChannel:
        """
        创建通信通道
        
        Args:
            source_version: 源版本
            target_version: 目标版本
            allowed_message_types: 允许的消息类型
            security_level: 安全级别
            
        Returns:
            CommunicationChannel: 创建的通道
        """
        channel_id = f"{source_version}->{target_version}"
        
        channel = CommunicationChannel(
            channel_id=channel_id,
            source_version=source_version,
            target_version=target_version,
            allowed_message_types=allowed_message_types,
            security_level=security_level,
        )
        
        self.channels[channel_id] = channel
        
        # 初始化消息队列
        if channel_id not in self.message_queue:
            self.message_queue[channel_id] = []
        
        logger.info(f"创建通信通道: {channel_id} (安全级别: {security_level.value})")
        
        return channel
    
    def _sign_message(self, message: Message) -> str:
        """为消息生成签名"""
        message_str = json.dumps({
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "source_version": message.source_version,
            "target_version": message.target_version,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
        }, sort_keys=True)
        
        signature = hmac.new(
            self.secret_key.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _verify_message(self, message: Message) -> bool:
        """验证消息签名"""
        if not message.signature:
            return False
        
        expected_signature = self._sign_message(message)
        return hmac.compare_digest(message.signature, expected_signature)
    
    def _check_access_control(
        self,
        source_version: str,
        target_version: str,
        security_level: SecurityLevel
    ) -> bool:
        """检查访问控制"""
        # PUBLIC: 所有版本可访问
        if security_level == SecurityLevel.PUBLIC:
            return True
        
        # RESTRICTED: 检查通道配置
        channel_id = f"{source_version}->{target_version}"
        if channel_id in self.channels:
            return True
        
        # CONFIDENTIAL/SECRET: 需要显式授权
        if security_level in [SecurityLevel.CONFIDENTIAL, SecurityLevel.SECRET]:
            allowed = self.access_control.get(source_version, [])
            return target_version in allowed
        
        return False
    
    async def send_message(
        self,
        message_type: MessageType,
        source_version: str,
        target_version: str,
        payload: Dict[str, Any],
        security_level: SecurityLevel = SecurityLevel.RESTRICTED
    ) -> bool:
        """
        发送消息
        
        Args:
            message_type: 消息类型
            source_version: 源版本
            target_version: 目标版本
            payload: 消息负载
            security_level: 安全级别
            
        Returns:
            bool: 是否发送成功
        """
        # 检查访问控制
        if not self._check_access_control(source_version, target_version, security_level):
            logger.warning(
                f"访问被拒绝: {source_version} -> {target_version} "
                f"(安全级别: {security_level.value})"
            )
            return False
        
        # 检查通道是否存在
        channel_id = f"{source_version}->{target_version}"
        if channel_id not in self.channels:
            logger.warning(f"通道不存在: {channel_id}")
            return False
        
        channel = self.channels[channel_id]
        
        # 检查消息类型是否允许
        if message_type not in channel.allowed_message_types:
            logger.warning(
                f"消息类型不允许: {message_type.value} "
                f"(通道: {channel_id})"
            )
            return False
        
        # 创建消息
        message = Message(
            message_id=f"msg_{datetime.now().timestamp()}",
            message_type=message_type,
            source_version=source_version,
            target_version=target_version,
            security_level=security_level,
            payload=payload,
        )
        
        # 生成签名
        message.signature = self._sign_message(message)
        
        # 添加到队列
        self.message_queue[channel_id].append(message)
        
        logger.info(
            f"消息已发送: {message_type.value} "
            f"({source_version} -> {target_version})"
        )
        
        # 触发消息处理器
        await self._handle_message(message)
        
        return True
    
    async def receive_message(
        self,
        target_version: str,
        message_type: Optional[MessageType] = None
    ) -> Optional[Message]:
        """
        接收消息
        
        Args:
            target_version: 目标版本
            message_type: 消息类型（可选过滤）
            
        Returns:
            Message: 接收到的消息，如果没有则返回None
        """
        # 查找所有指向目标版本的通道
        for channel_id, channel in self.channels.items():
            if channel.target_version == target_version:
                queue = self.message_queue.get(channel_id, [])
                
                for message in queue:
                    # 验证签名
                    if not self._verify_message(message):
                        logger.warning(f"消息签名验证失败: {message.message_id}")
                        continue
                    
                    # 类型过滤
                    if message_type and message.message_type != message_type:
                        continue
                    
                    # 移除已处理的消息
                    queue.remove(message)
                    
                    return message
        
        return None
    
    async def _handle_message(self, message: Message) -> None:
        """处理消息"""
        handlers = self.message_handlers.get(message.message_type, [])
        
        for handler in handlers:
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"消息处理失败: {e}", exc_info=True)
    
    def register_handler(
        self,
        message_type: MessageType,
        handler: Callable
    ) -> None:
        """注册消息处理器"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        
        self.message_handlers[message_type].append(handler)
        logger.info(f"注册消息处理器: {message_type.value}")
    
    def grant_access(
        self,
        source_version: str,
        target_version: str
    ) -> None:
        """授予访问权限"""
        if source_version not in self.access_control:
            self.access_control[source_version] = []
        
        if target_version not in self.access_control[source_version]:
            self.access_control[source_version].append(target_version)
            logger.info(f"授予访问权限: {source_version} -> {target_version}")
    
    def revoke_access(
        self,
        source_version: str,
        target_version: str
    ) -> None:
        """撤销访问权限"""
        if source_version in self.access_control:
            if target_version in self.access_control[source_version]:
                self.access_control[source_version].remove(target_version)
                logger.info(f"撤销访问权限: {source_version} -> {target_version}")
    
    def get_channel_status(self) -> Dict[str, Any]:
        """获取通道状态"""
        return {
            "channels": {
                channel_id: {
                    "source": channel.source_version,
                    "target": channel.target_version,
                    "security_level": channel.security_level.value,
                    "enabled": channel.enabled,
                    "queue_size": len(self.message_queue.get(channel_id, [])),
                }
                for channel_id, channel in self.channels.items()
            },
            "access_control": self.access_control,
            "total_messages": sum(
                len(queue) for queue in self.message_queue.values()
            ),
        }


def create_secure_channel(
    secret_key: Optional[str] = None
) -> SecureChannel:
    """
    创建安全通信通道
    
    Args:
        secret_key: 密钥（可选）
        
    Returns:
        SecureChannel: 配置好的安全通道
    """
    channel = SecureChannel(secret_key=secret_key)
    
    # 创建默认通道：v3 -> v1 (技术对比数据)
    channel.create_channel(
        source_version="v3",
        target_version="v1",
        allowed_message_types=[
            MessageType.TECH_COMPARISON,
            MessageType.PERFORMANCE_DATA,
        ],
        security_level=SecurityLevel.RESTRICTED,
    )
    
    # 创建默认通道：v1 -> v2 (错误报告)
    channel.create_channel(
        source_version="v1",
        target_version="v2",
        allowed_message_types=[
            MessageType.ERROR_REPORT,
            MessageType.FIX_SUGGESTION,
        ],
        security_level=SecurityLevel.RESTRICTED,
    )
    
    # 创建默认通道：v2 -> v3 (性能数据)
    channel.create_channel(
        source_version="v2",
        target_version="v3",
        allowed_message_types=[
            MessageType.PERFORMANCE_DATA,
            MessageType.STATUS_UPDATE,
        ],
        security_level=SecurityLevel.RESTRICTED,
    )
    
    # 创建协作通道：所有版本之间的协作请求
    for source in ["v1", "v2", "v3"]:
        for target in ["v1", "v2", "v3"]:
            if source != target:
                channel.create_channel(
                    source_version=source,
                    target_version=target,
                    allowed_message_types=[MessageType.COLLABORATION_REQUEST],
                    security_level=SecurityLevel.CONFIDENTIAL,
                )
    
    # 授予必要的访问权限
    channel.grant_access("v3", "v1")
    channel.grant_access("v1", "v2")
    channel.grant_access("v2", "v3")
    
    return channel

