"""
文档系统模块 (Documentation System Module)

维护完整的文档系统：
- 技术更新日志
- 评估参数和决策过程记录
- 版本切换操作手册
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentType(str, Enum):
    """文档类型"""
    TECH_UPDATE_LOG = "tech_update_log"
    EVALUATION_RECORD = "evaluation_record"
    VERSION_SWITCH_MANUAL = "version_switch_manual"
    PERFORMANCE_REPORT = "performance_report"
    DECISION_LOG = "decision_log"


@dataclass
class TechnologyUpdateLog:
    """技术更新日志"""
    log_id: str
    technology_id: str
    technology_name: str
    version: str
    update_type: str  # new, upgrade, fix, optimization
    description: str
    changes: List[Dict[str, Any]] = field(default_factory=list)
    performance_impact: Dict[str, float] = field(default_factory=dict)
    issues_resolved: List[str] = field(default_factory=list)
    updated_by: str = "system"
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    approved_by: Optional[str] = None
    approved_at: Optional[datetime] = None


@dataclass
class EvaluationParameter:
    """评估参数"""
    parameter_name: str
    value: Any
    unit: Optional[str] = None
    measurement_method: Optional[str] = None
    confidence_level: Optional[float] = None


@dataclass
class DecisionRecord:
    """决策记录"""
    decision_id: str
    technology_id: str
    decision_type: str  # promote, reject, defer
    decision: str
    rationale: str
    evaluation_parameters: List[EvaluationParameter] = field(default_factory=list)
    decision_makers: List[str] = field(default_factory=list)
    decision_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    outcome: Optional[str] = None
    outcome_evaluated_at: Optional[datetime] = None


@dataclass
class VersionSwitchStep:
    """版本切换步骤"""
    step_number: int
    step_name: str
    description: str
    commands: List[str] = field(default_factory=list)
    expected_output: Optional[str] = None
    rollback_commands: List[str] = field(default_factory=list)
    estimated_duration_seconds: int = 60
    critical: bool = False  # 关键步骤，失败时需要回滚


@dataclass
class VersionSwitchManual:
    """版本切换操作手册"""
    manual_id: str
    from_version: str
    to_version: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    prerequisites: List[str] = field(default_factory=list)
    steps: List[VersionSwitchStep] = field(default_factory=list)
    verification_steps: List[str] = field(default_factory=list)
    rollback_procedure: List[str] = field(default_factory=list)
    estimated_total_duration_seconds: int = 300
    risk_level: str = "medium"  # low, medium, high
    tested: bool = False
    tested_at: Optional[datetime] = None


class DocumentationSystem:
    """
    文档系统
    
    管理所有技术文档的生成、存储和检索。
    """
    
    def __init__(self, storage_path: str = "./docs/version_architecture"):
        """
        初始化
        
        Args:
            storage_path: 文档存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        (self.storage_path / "tech_updates").mkdir(exist_ok=True)
        (self.storage_path / "evaluations").mkdir(exist_ok=True)
        (self.storage_path / "manuals").mkdir(exist_ok=True)
        (self.storage_path / "decisions").mkdir(exist_ok=True)
    
    def log_technology_update(self, log: TechnologyUpdateLog):
        """记录技术更新"""
        log_file = (
            self.storage_path / "tech_updates" /
            f"{log.log_id}.json"
        )
        
        log_dict = asdict(log)
        # 转换datetime为字符串
        log_dict["updated_at"] = log.updated_at.isoformat()
        if log.approved_at:
            log_dict["approved_at"] = log.approved_at.isoformat()
        
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Logged technology update: {log.log_id}")
    
    def record_evaluation(
        self,
        technology_id: str,
        parameters: List[EvaluationParameter],
        decision: DecisionRecord
    ):
        """记录评估参数和决策过程"""
        # 保存评估参数
        eval_file = (
            self.storage_path / "evaluations" /
            f"{technology_id}_{decision.decision_id}.json"
        )
        
        eval_data = {
            "technology_id": technology_id,
            "evaluation_parameters": [
                asdict(p) for p in parameters
            ],
            "decision": asdict(decision)
        }
        
        # 转换datetime
        eval_data["decision"]["decision_at"] = decision.decision_at.isoformat()
        if decision.outcome_evaluated_at:
            eval_data["decision"]["outcome_evaluated_at"] = (
                decision.outcome_evaluated_at.isoformat()
            )
        
        with open(eval_file, "w", encoding="utf-8") as f:
            json.dump(eval_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Recorded evaluation for technology: {technology_id}")
    
    def create_version_switch_manual(
        self,
        from_version: str,
        to_version: str,
        steps: List[VersionSwitchStep]
    ) -> VersionSwitchManual:
        """创建版本切换操作手册"""
        manual_id = f"manual_{from_version}_to_{to_version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        total_duration = sum(step.estimated_duration_seconds for step in steps)
        
        manual = VersionSwitchManual(
            manual_id=manual_id,
            from_version=from_version,
            to_version=to_version,
            steps=steps,
            estimated_total_duration_seconds=total_duration,
            prerequisites=[
                f"Backup of {from_version}",
                f"Health check of {from_version}",
                f"Verification of {to_version} availability"
            ],
            verification_steps=[
                f"Verify {to_version} is running",
                f"Check API endpoints",
                f"Monitor error rates",
                f"Validate performance metrics"
            ],
            rollback_procedure=[
                f"Stop {to_version}",
                f"Restore {from_version}",
                f"Verify {from_version} health",
                f"Update routing to {from_version}"
            ]
        )
        
        # 保存手册
        manual_file = (
            self.storage_path / "manuals" /
            f"{manual_id}.json"
        )
        
        manual_dict = asdict(manual)
        manual_dict["created_at"] = manual.created_at.isoformat()
        manual_dict["updated_at"] = manual.updated_at.isoformat()
        if manual.tested_at:
            manual_dict["tested_at"] = manual.tested_at.isoformat()
        
        # 转换步骤中的datetime（如果有）
        for step in manual_dict["steps"]:
            if "estimated_duration_seconds" not in step:
                step["estimated_duration_seconds"] = 60
        
        with open(manual_file, "w", encoding="utf-8") as f:
            json.dump(manual_dict, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created version switch manual: {manual_id}")
        
        return manual
    
    def get_technology_update_history(
        self,
        technology_id: Optional[str] = None,
        limit: int = 100
    ) -> List[TechnologyUpdateLog]:
        """获取技术更新历史"""
        updates_dir = self.storage_path / "tech_updates"
        
        if not updates_dir.exists():
            return []
        
        logs = []
        for log_file in sorted(updates_dir.glob("*.json"), reverse=True):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    log_dict = json.load(f)
                
                if technology_id and log_dict.get("technology_id") != technology_id:
                    continue
                
                # 转换字符串回datetime
                log_dict["updated_at"] = datetime.fromisoformat(log_dict["updated_at"])
                if log_dict.get("approved_at"):
                    log_dict["approved_at"] = datetime.fromisoformat(
                        log_dict["approved_at"]
                    )
                
                log = TechnologyUpdateLog(**log_dict)
                logs.append(log)
                
                if len(logs) >= limit:
                    break
            except Exception as e:
                logger.error(f"Error reading log file {log_file}: {e}")
        
        return logs
    
    def get_evaluation_history(
        self,
        technology_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取评估历史"""
        eval_dir = self.storage_path / "evaluations"
        
        if not eval_dir.exists():
            return []
        
        evaluations = []
        for eval_file in eval_dir.glob("*.json"):
            try:
                with open(eval_file, "r", encoding="utf-8") as f:
                    eval_data = json.load(f)
                
                if technology_id and eval_data.get("technology_id") != technology_id:
                    continue
                
                evaluations.append(eval_data)
            except Exception as e:
                logger.error(f"Error reading evaluation file {eval_file}: {e}")
        
        return evaluations
    
    def get_version_switch_manual(
        self,
        from_version: str,
        to_version: str
    ) -> Optional[VersionSwitchManual]:
        """获取版本切换操作手册"""
        manuals_dir = self.storage_path / "manuals"
        
        if not manuals_dir.exists():
            return None
        
        # 查找匹配的手册
        for manual_file in manuals_dir.glob("*.json"):
            try:
                with open(manual_file, "r", encoding="utf-8") as f:
                    manual_dict = json.load(f)
                
                if (
                    manual_dict.get("from_version") == from_version and
                    manual_dict.get("to_version") == to_version
                ):
                    # 转换datetime
                    manual_dict["created_at"] = datetime.fromisoformat(
                        manual_dict["created_at"]
                    )
                    manual_dict["updated_at"] = datetime.fromisoformat(
                        manual_dict["updated_at"]
                    )
                    if manual_dict.get("tested_at"):
                        manual_dict["tested_at"] = datetime.fromisoformat(
                            manual_dict["tested_at"]
                        )
                    
                    return VersionSwitchManual(**manual_dict)
            except Exception as e:
                logger.error(f"Error reading manual file {manual_file}: {e}")
        
        return None
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """生成摘要报告"""
        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "statistics": {
                "total_tech_updates": len(list((self.storage_path / "tech_updates").glob("*.json"))),
                "total_evaluations": len(list((self.storage_path / "evaluations").glob("*.json"))),
                "total_manuals": len(list((self.storage_path / "manuals").glob("*.json"))),
            },
            "recent_updates": [],
            "recent_decisions": []
        }
        
        # 获取最近的更新
        recent_updates = self.get_technology_update_history(limit=10)
        report["recent_updates"] = [
            {
                "log_id": u.log_id,
                "technology_name": u.technology_name,
                "update_type": u.update_type,
                "updated_at": u.updated_at.isoformat()
            }
            for u in recent_updates
        ]
        
        return report

