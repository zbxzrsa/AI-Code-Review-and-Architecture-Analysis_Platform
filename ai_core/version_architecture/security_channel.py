"""
安全通信通道

建立AI系统间的安全通信通道，确保数据隔离的同时允许必要的信息共享
"""

import asyncio
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set

try:
    from cryptography.fernet import Fernet
except ImportError:
    # 如果cryptography未安装，使用简单的base64编码作为降级方案
    import base64
    import os
    
    class Fernet:
        """简单的Fernet替代实现（仅用于开发）"""
        def __init__(self, key: bytes):
            self.key = key[:32]  # 使用前32字节作为密钥
        
        def encrypt(self, data: bytes) -> bytes:
            return base64.b64encode(data)
        
        def decrypt(self, token: bytes) -> bytes:
            return base64.b64decode(token)
    
    def generate_key() -> bytes:
        return os.urandom(32)
from .version_config import VersionRole, SecurityChannelConfig

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """消息类型"""
    TECH_COMPARISON = "tech_comparison"  # 技术对比数据
    ERROR_REPORT = "error_report"  # 错误报告
    FIX_SUGGESTION = "fix_suggestion"  # 修复建议
    PERFORMANCE_METRICS = "performance_metrics"  # 性能指标
    PROMOTION_REQUEST = "promotion_request"  # 升级请求
    COLLABORATION_REQUEST = "collaboration_request"  # 协作请求


@dataclass
class SecureMessage:
    """安全消息"""
    message_id: str
    message_type: MessageType
    from_version: VersionRole
    to_version: VersionRole
    payload: Dict[str, Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    signature: Optional[str] = None
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "from_version": self.from_version.value,
            "to_version": self.to_version.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "signature": self.signature,
            "encrypted": self.encrypted,
        }


class SecurityChannel:
    """
    安全通信通道
    
    功能：
    - 消息加密和签名
    - 消息验证
    - 访问控制
    - 审计日志
    """
    
    def __init__(self, config: SecurityChannelConfig):
        """
        初始化安全通信通道
        
        Args:
            config: 安全通道配置
        """
        self.config = config
        
        # 生成加密密钥（实际应用中应从安全存储获取）
        if config.encryption_enabled:
            try:
                self.encryption_key = Fernet.generate_key()
            except AttributeError:
                # 如果使用降级方案
                import os
                self.encryption_key = os.urandom(32)
            self.cipher = Fernet(self.encryption_key)
        else:
            self.encryption_key = None
            self.cipher = None
        
        # 消息队列
        self.message_queue: Dict[VersionRole, List[SecureMessage]] = {
            VersionRole.V1_DEVELOPMENT: [],
            VersionRole.V2_STABLE: [],
            VersionRole.V3_BENCHMARK: [],
        }
        
        # 审计日志
        self.audit_log: List[Dict[str, Any]] = []
        
        # 消息统计
        self.message_stats: Dict[str, int] = {}
        
        logger.info("安全通信通道初始化完成")
    
    async def send_message(
        self,
        message: SecureMessage,
    ) -> bool:
        """
        发送安全消息
        
        Args:
            message: 要发送的消息
        
        Returns:
            是否发送成功
        """
        # 验证消息类型是否允许
        if message.message_type.value not in self.config.allowed_message_types:
            logger.warning(f"消息类型不允许: {message.message_type.value}")
            return False
        
        # 加密消息（如果需要）
        if self.config.encryption_enabled and self.cipher:
            try:
                payload_str = json.dumps(message.payload)
                encrypted_payload = self.cipher.encrypt(payload_str.encode())
                message.payload = {"encrypted_data": encrypted_payload.decode()}
                message.encrypted = True
            except Exception as e:
                logger.error(f"消息加密失败: {e}")
                return False
        
        # 签名消息（如果需要）
        if self.config.message_validation:
            message.signature = self._generate_signature(message)
        
        # 添加到目标版本的消息队列
        self.message_queue[message.to_version].append(message)
        
        # 记录审计日志
        if self.config.audit_logging:
            self._log_message(message, "sent")
        
        # 更新统计
        self.message_stats[message.message_type.value] = \
            self.message_stats.get(message.message_type.value, 0) + 1
        
        logger.info(
            f"消息已发送: {message.message_id} "
            f"({message.from_version.value} -> {message.to_version.value})"
        )
        
        return True
    
    async def receive_message(
        self,
        version_role: VersionRole,
        message_type: Optional[MessageType] = None,
    ) -> Optional[SecureMessage]:
        """
        接收消息
        
        Args:
            version_role: 版本角色
            message_type: 消息类型（可选，用于过滤）
        
        Returns:
            消息或None
        """
        queue = self.message_queue.get(version_role, [])
        
        if not queue:
            return None
        
        # 查找匹配的消息
        for i, message in enumerate(queue):
            if message_type is None or message.message_type == message_type:
                # 移除已处理的消息
                queue.pop(i)
                
                # 验证消息
                if self.config.message_validation:
                    if not self._verify_message(message):
                        logger.warning(f"消息验证失败: {message.message_id}")
                        continue
                
                # 解密消息（如果需要）
                if message.encrypted and self.cipher:
                    try:
                        encrypted_data = message.payload.get("encrypted_data", "")
                        decrypted_payload = self.cipher.decrypt(encrypted_data.encode())
                        message.payload = json.loads(decrypted_payload.decode())
                        message.encrypted = False
                    except Exception as e:
                        logger.error(f"消息解密失败: {e}")
                        continue
                
                # 记录审计日志
                if self.config.audit_logging:
                    self._log_message(message, "received")
                
                return message
        
        return None
    
    async def receive_all_messages(
        self,
        version_role: VersionRole,
    ) -> List[SecureMessage]:
        """
        接收所有消息
        
        Args:
            version_role: 版本角色
        
        Returns:
            消息列表
        """
        messages = []
        while True:
            message = await self.receive_message(version_role)
            if message is None:
                break
            messages.append(message)
        return messages
    
    def _generate_signature(self, message: SecureMessage) -> str:
        """
        生成消息签名
        
        Args:
            message: 消息
        
        Returns:
            签名
        """
        # 使用HMAC生成签名
        secret = self.encryption_key or b"default_secret"
        message_str = json.dumps({
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "from_version": message.from_version.value,
            "to_version": message.to_version.value,
            "payload": message.payload,
            "timestamp": message.timestamp.isoformat(),
        }, sort_keys=True)
        
        signature = hmac.new(
            secret,
            message_str.encode(),
            hashlib.sha256,
        ).hexdigest()
        
        return signature
    
    def _verify_message(self, message: SecureMessage) -> bool:
        """
        验证消息
        
        Args:
            message: 消息
        
        Returns:
            是否验证通过
        """
        if not message.signature:
            return False
        
        expected_signature = self._generate_signature(message)
        return hmac.compare_digest(message.signature, expected_signature)
    
    def _log_message(self, message: SecureMessage, action: str):
        """
        记录审计日志
        
        Args:
            message: 消息
            action: 操作（sent/received）
        """
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "message_id": message.message_id,
            "message_type": message.message_type.value,
            "from_version": message.from_version.value,
            "to_version": message.to_version.value,
        }
        
        self.audit_log.append(log_entry)
        
        # 限制日志大小（保留最近N条）
        max_log_size = 10000
        if len(self.audit_log) > max_log_size:
            self.audit_log = self.audit_log[-max_log_size:]
    
    def get_audit_log(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        version_role: Optional[VersionRole] = None,
    ) -> List[Dict[str, Any]]:
        """
        获取审计日志
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            version_role: 版本角色（可选，用于过滤）
        
        Returns:
            日志条目列表
        """
        logs = self.audit_log
        
        if start_time:
            logs = [log for log in logs if datetime.fromisoformat(log["timestamp"]) >= start_time]
        
        if end_time:
            logs = [log for log in logs if datetime.fromisoformat(log["timestamp"]) <= end_time]
        
        if version_role:
            logs = [
                log for log in logs
                if log.get("from_version") == version_role.value
                or log.get("to_version") == version_role.value
            ]
        
        return logs
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return {
            "total_messages": sum(self.message_stats.values()),
            "by_type": self.message_stats.copy(),
            "queue_sizes": {
                role.value: len(queue)
                for role, queue in self.message_queue.items()
            },
            "audit_log_size": len(self.audit_log),
        }


def create_security_channel(config: SecurityChannelConfig) -> SecurityChannel:
    """
    创建安全通信通道
    
    Args:
        config: 安全通道配置
    
    Returns:
        安全通信通道实例
    """
    return SecurityChannel(config)

