"""
Human-in-the-Loop AI系统

功能：
- 人工审核工作流
- 反馈循环
- 人工验证和确认
- 学习人类反馈
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import uuid

logger = logging.getLogger(__name__)


class ReviewStatus(str, Enum):
    """审核状态"""
    PENDING = "pending"
    IN_REVIEW = "in_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


class FeedbackType(str, Enum):
    """反馈类型"""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    MISSING = "missing"
    FALSE_POSITIVE = "false_positive"
    FALSE_NEGATIVE = "false_negative"


@dataclass
class HumanReviewTask:
    """人工审核任务"""
    task_id: str
    review_id: str
    finding_id: str
    issue: str
    ai_suggestion: str
    confidence: float
    status: ReviewStatus
    assigned_to: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    reviewed_at: Optional[datetime] = None
    reviewer_notes: Optional[str] = None
    priority: str = "medium"  # high, medium, low


@dataclass
class HumanFeedback:
    """人工反馈"""
    feedback_id: str
    task_id: str
    finding_id: str
    feedback_type: FeedbackType
    reviewer_id: str
    comments: Optional[str] = None
    corrected_suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReviewWorkflow:
    """审核工作流"""
    workflow_id: str
    name: str
    description: str
    steps: List[Dict[str, Any]]
    auto_approve_threshold: float = 0.95  # 置信度超过此值可自动通过
    require_human_review: bool = True
    tenant_id: Optional[str] = None


class HumanInTheLoopService:
    """
    Human-in-the-Loop AI服务
    
    功能：
    1. 创建人工审核任务
    2. 管理工作流
    3. 收集和存储反馈
    4. 学习反馈以改进AI
    """
    
    def __init__(self, db_connection=None, notification_service=None):
        """
        初始化服务
        
        Args:
            db_connection: 数据库连接
            notification_service: 通知服务
        """
        self.db = db_connection
        self.notification_service = notification_service
        
        # 待审核任务
        self.pending_tasks: Dict[str, HumanReviewTask] = {}
        
        # 工作流配置
        self.workflows: Dict[str, ReviewWorkflow] = {}
    
    async def create_review_tasks(
        self,
        findings: List[Dict[str, Any]],
        review_id: str,
        workflow_id: Optional[str] = None,
        auto_approve_threshold: float = 0.95
    ) -> List[HumanReviewTask]:
        """
        创建人工审核任务
        
        Args:
            findings: AI发现的列表
            review_id: 审查ID
            workflow_id: 工作流ID（可选）
            auto_approve_threshold: 自动批准阈值
        
        Returns:
            创建的任务列表
        """
        tasks = []
        
        for finding in findings:
            finding_id = finding.get("id", str(uuid.uuid4()))
            confidence = finding.get("confidence", 0.5)
            
            # 高置信度且超过阈值，可以自动通过
            if confidence >= auto_approve_threshold:
                task = HumanReviewTask(
                    task_id=str(uuid.uuid4()),
                    review_id=review_id,
                    finding_id=finding_id,
                    issue=finding.get("issue", ""),
                    ai_suggestion=finding.get("suggestion", ""),
                    confidence=confidence,
                    status=ReviewStatus.APPROVED,
                    priority=self._determine_priority(finding)
                )
                task.reviewed_at = datetime.now(timezone.utc)
                task.reviewer_notes = "自动批准：高置信度"
            else:
                # 需要人工审核
                task = HumanReviewTask(
                    task_id=str(uuid.uuid4()),
                    review_id=review_id,
                    finding_id=finding_id,
                    issue=finding.get("issue", ""),
                    ai_suggestion=finding.get("suggestion", ""),
                    confidence=confidence,
                    status=ReviewStatus.PENDING,
                    priority=self._determine_priority(finding)
                )
            
            tasks.append(task)
            self.pending_tasks[task.task_id] = task
        
        # 发送通知
        pending_count = sum(1 for t in tasks if t.status == ReviewStatus.PENDING)
        if pending_count > 0 and self.notification_service:
            await self.notification_service.notify_reviewers(
                f"有 {pending_count} 个新任务需要审核",
                review_id
            )
        
        return tasks
    
    def _determine_priority(self, finding: Dict[str, Any]) -> str:
        """确定任务优先级"""
        severity = finding.get("severity", "medium")
        confidence = finding.get("confidence", 0.5)
        
        if severity == "critical" or (severity == "high" and confidence > 0.8):
            return "high"
        elif severity == "high" or confidence > 0.7:
            return "medium"
        else:
            return "low"
    
    async def submit_feedback(
        self,
        task_id: str,
        reviewer_id: str,
        feedback_type: FeedbackType,
        comments: Optional[str] = None,
        corrected_suggestion: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> HumanFeedback:
        """
        提交人工反馈
        
        Args:
            task_id: 任务ID
            reviewer_id: 审核者ID
            feedback_type: 反馈类型
            comments: 评论
            corrected_suggestion: 修正后的建议
            metadata: 额外元数据
        
        Returns:
            HumanFeedback: 反馈对象
        """
        if task_id not in self.pending_tasks:
            raise ValueError(f"Task not found: {task_id}")
        
        task = self.pending_tasks[task_id]
        
        # 创建反馈
        feedback = HumanFeedback(
            feedback_id=str(uuid.uuid4()),
            task_id=task_id,
            finding_id=task.finding_id,
            feedback_type=feedback_type,
            reviewer_id=reviewer_id,
            comments=comments,
            corrected_suggestion=corrected_suggestion,
            metadata=metadata or {}
        )
        
        # 更新任务状态
        if feedback_type == FeedbackType.CORRECT:
            task.status = ReviewStatus.APPROVED
        elif feedback_type == FeedbackType.INCORRECT:
            task.status = ReviewStatus.REJECTED
        elif feedback_type == FeedbackType.PARTIAL:
            task.status = ReviewStatus.NEEDS_REVISION
        else:
            task.status = ReviewStatus.REJECTED
        
        task.reviewed_at = datetime.now(timezone.utc)
        task.reviewer_notes = comments
        task.assigned_to = reviewer_id
        
        # 存储反馈（用于学习）
        await self._store_feedback(feedback)
        
        # 触发学习流程
        await self._learn_from_feedback(feedback, task)
        
        logger.info(
            f"Feedback submitted for task {task_id}",
            extra={
                "task_id": task_id,
                "reviewer_id": reviewer_id,
                "feedback_type": feedback_type.value
            }
        )
        
        return feedback
    
    async def _store_feedback(self, feedback: HumanFeedback) -> None:
        """存储反馈到数据库"""
        # TODO: 实现数据库存储
        pass
    
    async def _learn_from_feedback(
        self,
        feedback: HumanFeedback,
        task: HumanReviewTask
    ) -> None:
        """
        从反馈中学习
        
        更新AI模型的置信度、规则权重等
        """
        # TODO: 实现学习逻辑
        # 1. 分析反馈模式
        # 2. 调整置信度计算
        # 3. 更新规则权重
        # 4. 记录到学习数据库
        
        logger.info(
            f"Learning from feedback {feedback.feedback_id}",
            extra={
                "feedback_type": feedback.feedback_type.value,
                "finding_id": feedback.finding_id
            }
        )
    
    async def get_pending_tasks(
        self,
        reviewer_id: Optional[str] = None,
        priority: Optional[str] = None,
        limit: int = 50
    ) -> List[HumanReviewTask]:
        """
        获取待审核任务
        
        Args:
            reviewer_id: 审核者ID（可选，用于过滤）
            priority: 优先级过滤
            limit: 返回数量限制
        
        Returns:
            任务列表
        """
        tasks = [
            t for t in self.pending_tasks.values()
            if t.status == ReviewStatus.PENDING
        ]
        
        if reviewer_id:
            tasks = [t for t in tasks if t.assigned_to == reviewer_id]
        
        if priority:
            tasks = [t for t in tasks if t.priority == priority]
        
        # 按优先级和创建时间排序
        priority_order = {"high": 3, "medium": 2, "low": 1}
        tasks.sort(
            key=lambda t: (
                priority_order.get(t.priority, 0),
                t.created_at
            ),
            reverse=True
        )
        
        return tasks[:limit]
    
    async def get_feedback_statistics(
        self,
        review_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        获取反馈统计
        
        Args:
            review_id: 审查ID（可选）
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            统计数据
        """
        # TODO: 从数据库查询反馈统计
        return {
            "total_feedback": 0,
            "correct_count": 0,
            "incorrect_count": 0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "average_confidence_when_correct": 0.0,
            "average_confidence_when_incorrect": 0.0
        }
    
    def configure_workflow(
        self,
        workflow: ReviewWorkflow
    ) -> None:
        """
        配置审核工作流
        
        Args:
            workflow: 工作流配置
        """
        self.workflows[workflow.workflow_id] = workflow
        logger.info(f"Workflow configured: {workflow.workflow_id}")
    
    async def apply_workflow(
        self,
        workflow_id: str,
        findings: List[Dict[str, Any]],
        review_id: str
    ) -> List[HumanReviewTask]:
        """
        应用工作流
        
        Args:
            workflow_id: 工作流ID
            findings: 发现列表
            review_id: 审查ID
        
        Returns:
            创建的任务列表
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        workflow = self.workflows[workflow_id]
        
        # 使用工作流的自动批准阈值
        tasks = await self.create_review_tasks(
            findings,
            review_id,
            workflow_id,
            workflow.auto_approve_threshold
        )
        
        return tasks

