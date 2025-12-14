"""
文档系统

维护完整的文档：
- 技术更新日志
- 评估参数和决策过程记录
- 版本切换操作手册
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """文档类型"""
    TECH_UPDATE_LOG = "tech_update_log"
    EVALUATION_RECORD = "evaluation_record"
    VERSION_SWITCH_MANUAL = "version_switch_manual"
    PERFORMANCE_REPORT = "performance_report"
    DECISION_LOG = "decision_log"


@dataclass
class TechUpdateLogEntry:
    """技术更新日志条目"""
    entry_id: str
    timestamp: datetime
    technology_id: str
    technology_name: str
    version_from: str
    version_to: str
    update_type: str  # "promotion", "rollback", "patch"
    description: str
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    issues_resolved: List[str] = field(default_factory=list)
    approved_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRecord:
    """评估记录"""
    record_id: str
    timestamp: datetime
    technology_id: str
    evaluation_type: str  # "tech_comparison", "experiment", "validation"
    
    # 评估参数
    evaluation_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 评估结果
    evaluation_results: Dict[str, Any] = field(default_factory=dict)
    
    # 决策过程
    decision_process: List[Dict[str, Any]] = field(default_factory=list)
    
    # 最终决策
    final_decision: str = ""
    decision_reason: str = ""
    decision_maker: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VersionSwitchStep:
    """版本切换步骤"""
    step_number: int
    action: str
    description: str
    commands: List[str] = field(default_factory=list)
    verification: str = ""
    rollback_instructions: str = ""


@dataclass
class VersionSwitchManual:
    """版本切换操作手册"""
    manual_id: str
    version_from: str
    version_to: str
    created_at: datetime
    updated_at: datetime
    
    # 前置条件
    prerequisites: List[str] = field(default_factory=list)
    
    # 切换步骤
    steps: List[VersionSwitchStep] = field(default_factory=list)
    
    # 验证步骤
    verification_steps: List[str] = field(default_factory=list)
    
    # 回滚步骤
    rollback_steps: List[VersionSwitchStep] = field(default_factory=list)
    
    # 注意事项
    notes: List[str] = field(default_factory=list)
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)


class DocumentationManager:
    """
    文档管理器
    
    管理所有文档的创建、更新和查询
    """
    
    def __init__(self, docs_directory: str = "docs/version_architecture"):
        """
        初始化文档管理器
        
        Args:
            docs_directory: 文档目录
        """
        self.docs_directory = Path(docs_directory)
        self.docs_directory.mkdir(parents=True, exist_ok=True)
        
        # 文档存储
        self.tech_update_logs: List[TechUpdateLogEntry] = []
        self.evaluation_records: List[EvaluationRecord] = []
        self.version_switch_manuals: Dict[str, VersionSwitchManual] = {}
        
        # 加载现有文档
        self._load_documents()
        
        logger.info(f"文档管理器初始化完成: {self.docs_directory}")
    
    def _load_documents(self):
        """加载现有文档"""
        # 加载技术更新日志
        log_file = self.docs_directory / "tech_update_log.json"
        if log_file.exists():
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tech_update_logs = [
                        TechUpdateLogEntry(**entry) for entry in data
                    ]
            except Exception as e:
                logger.error(f"加载技术更新日志失败: {e}")
        
        # 加载评估记录
        record_file = self.docs_directory / "evaluation_records.json"
        if record_file.exists():
            try:
                with open(record_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.evaluation_records = [
                        EvaluationRecord(**record) for record in data
                    ]
            except Exception as e:
                logger.error(f"加载评估记录失败: {e}")
    
    def _save_documents(self):
        """保存文档"""
        # 保存技术更新日志
        log_file = self.docs_directory / "tech_update_log.json"
        try:
            with open(log_file, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(entry) for entry in self.tech_update_logs],
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.error(f"保存技术更新日志失败: {e}")
        
        # 保存评估记录
        record_file = self.docs_directory / "evaluation_records.json"
        try:
            with open(record_file, "w", encoding="utf-8") as f:
                json.dump(
                    [asdict(record) for record in self.evaluation_records],
                    f,
                    indent=2,
                    default=str,
                )
        except Exception as e:
            logger.error(f"保存评估记录失败: {e}")
    
    def add_tech_update_log(
        self,
        technology_id: str,
        technology_name: str,
        version_from: str,
        version_to: str,
        update_type: str,
        description: str,
        performance_metrics: Optional[Dict[str, Any]] = None,
        issues_resolved: Optional[List[str]] = None,
        approved_by: Optional[str] = None,
    ) -> TechUpdateLogEntry:
        """
        添加技术更新日志
        
        Args:
            technology_id: 技术ID
            technology_name: 技术名称
            version_from: 源版本
            version_to: 目标版本
            update_type: 更新类型
            description: 描述
            performance_metrics: 性能指标
            issues_resolved: 已解决问题列表
            approved_by: 批准人
        
        Returns:
            日志条目
        """
        entry = TechUpdateLogEntry(
            entry_id=f"log_{datetime.now().timestamp()}",
            timestamp=datetime.now(timezone.utc),
            technology_id=technology_id,
            technology_name=technology_name,
            version_from=version_from,
            version_to=version_to,
            update_type=update_type,
            description=description,
            performance_metrics=performance_metrics or {},
            issues_resolved=issues_resolved or [],
            approved_by=approved_by,
        )
        
        self.tech_update_logs.append(entry)
        self._save_documents()
        
        logger.info(f"技术更新日志已添加: {entry.entry_id}")
        return entry
    
    def add_evaluation_record(
        self,
        technology_id: str,
        evaluation_type: str,
        evaluation_parameters: Dict[str, Any],
        evaluation_results: Dict[str, Any],
        decision_process: List[Dict[str, Any]],
        final_decision: str,
        decision_reason: str,
        decision_maker: Optional[str] = None,
    ) -> EvaluationRecord:
        """
        添加评估记录
        
        Args:
            technology_id: 技术ID
            evaluation_type: 评估类型
            evaluation_parameters: 评估参数
            evaluation_results: 评估结果
            decision_process: 决策过程
            final_decision: 最终决策
            decision_reason: 决策原因
            decision_maker: 决策者
        
        Returns:
            评估记录
        """
        record = EvaluationRecord(
            record_id=f"eval_{datetime.now().timestamp()}",
            timestamp=datetime.now(timezone.utc),
            technology_id=technology_id,
            evaluation_type=evaluation_type,
            evaluation_parameters=evaluation_parameters,
            evaluation_results=evaluation_results,
            decision_process=decision_process,
            final_decision=final_decision,
            decision_reason=decision_reason,
            decision_maker=decision_maker,
        )
        
        self.evaluation_records.append(record)
        self._save_documents()
        
        logger.info(f"评估记录已添加: {record.record_id}")
        return record
    
    def create_version_switch_manual(
        self,
        version_from: str,
        version_to: str,
        steps: List[VersionSwitchStep],
        prerequisites: Optional[List[str]] = None,
        verification_steps: Optional[List[str]] = None,
        rollback_steps: Optional[List[VersionSwitchStep]] = None,
        notes: Optional[List[str]] = None,
    ) -> VersionSwitchManual:
        """
        创建版本切换操作手册
        
        Args:
            version_from: 源版本
            version_to: 目标版本
            steps: 切换步骤
            prerequisites: 前置条件
            verification_steps: 验证步骤
            rollback_steps: 回滚步骤
            notes: 注意事项
        
        Returns:
            操作手册
        """
        manual_id = f"manual_{version_from}_to_{version_to}_{datetime.now().timestamp()}"
        
        manual = VersionSwitchManual(
            manual_id=manual_id,
            version_from=version_from,
            version_to=version_to,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            prerequisites=prerequisites or [],
            steps=steps,
            verification_steps=verification_steps or [],
            rollback_steps=rollback_steps or [],
            notes=notes or [],
        )
        
        self.version_switch_manuals[manual_id] = manual
        
        # 保存为Markdown文件
        self._save_manual_as_markdown(manual)
        
        logger.info(f"版本切换操作手册已创建: {manual_id}")
        return manual
    
    def _save_manual_as_markdown(self, manual: VersionSwitchManual):
        """保存操作手册为Markdown文件"""
        manual_file = self.docs_directory / f"{manual.manual_id}.md"
        
        content = f"""# 版本切换操作手册

## 基本信息

- **手册ID**: {manual.manual_id}
- **源版本**: {manual.version_from}
- **目标版本**: {manual.version_to}
- **创建时间**: {manual.created_at.isoformat()}
- **更新时间**: {manual.updated_at.isoformat()}

## 前置条件

"""
        for i, prerequisite in enumerate(manual.prerequisites, 1):
            content += f"{i}. {prerequisite}\n"
        
        content += "\n## 切换步骤\n\n"
        for step in manual.steps:
            content += f"### 步骤 {step.step_number}: {step.action}\n\n"
            content += f"{step.description}\n\n"
            if step.commands:
                content += "**执行命令**:\n\n```bash\n"
                for cmd in step.commands:
                    content += f"{cmd}\n"
                content += "```\n\n"
            if step.verification:
                content += f"**验证**: {step.verification}\n\n"
            if step.rollback_instructions:
                content += f"**回滚说明**: {step.rollback_instructions}\n\n"
        
        if manual.verification_steps:
            content += "## 验证步骤\n\n"
            for i, verification in enumerate(manual.verification_steps, 1):
                content += f"{i}. {verification}\n"
            content += "\n"
        
        if manual.rollback_steps:
            content += "## 回滚步骤\n\n"
            for step in manual.rollback_steps:
                content += f"### 步骤 {step.step_number}: {step.action}\n\n"
                content += f"{step.description}\n\n"
                if step.commands:
                    content += "**执行命令**:\n\n```bash\n"
                    for cmd in step.commands:
                        content += f"{cmd}\n"
                    content += "```\n\n"
        
        if manual.notes:
            content += "## 注意事项\n\n"
            for note in manual.notes:
                content += f"- {note}\n"
            content += "\n"
        
        try:
            with open(manual_file, "w", encoding="utf-8") as f:
                f.write(content)
        except Exception as e:
            logger.error(f"保存操作手册失败: {e}")
    
    def get_tech_update_logs(
        self,
        technology_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[TechUpdateLogEntry]:
        """
        获取技术更新日志
        
        Args:
            technology_id: 技术ID（可选，用于过滤）
            limit: 限制数量
        
        Returns:
            日志条目列表
        """
        logs = self.tech_update_logs
        
        if technology_id:
            logs = [log for log in logs if log.technology_id == technology_id]
        
        return logs[-limit:]
    
    def get_evaluation_records(
        self,
        technology_id: Optional[str] = None,
        evaluation_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[EvaluationRecord]:
        """
        获取评估记录
        
        Args:
            technology_id: 技术ID（可选）
            evaluation_type: 评估类型（可选）
            limit: 限制数量
        
        Returns:
            评估记录列表
        """
        records = self.evaluation_records
        
        if technology_id:
            records = [r for r in records if r.technology_id == technology_id]
        
        if evaluation_type:
            records = [r for r in records if r.evaluation_type == evaluation_type]
        
        return records[-limit:]
    
    def get_version_switch_manual(
        self,
        version_from: str,
        version_to: str,
    ) -> Optional[VersionSwitchManual]:
        """
        获取版本切换操作手册
        
        Args:
            version_from: 源版本
            version_to: 目标版本
        
        Returns:
            操作手册或None
        """
        for manual in self.version_switch_manuals.values():
            if manual.version_from == version_from and manual.version_to == version_to:
                return manual
        return None


def create_documentation_manager(docs_directory: str = "docs/version_architecture") -> DocumentationManager:
    """
    创建文档管理器
    
    Args:
        docs_directory: 文档目录
    
    Returns:
        文档管理器实例
    """
    return DocumentationManager(docs_directory)

