"""
Performance Monitoring Dashboard

Features:
- Real-time metrics collection
- Learning effect evaluation
- Feedback optimization closed-loop
- SLA compliance tracking
"""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import statistics

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class MetricDefinition:
    """Definition of a metric"""
    name: str
    metric_type: MetricType
    description: str
    unit: str
    labels: List[str] = field(default_factory=list)
    
    # SLA thresholds
    warning_threshold: Optional[float] = None
    error_threshold: Optional[float] = None
    target_value: Optional[float] = None


@dataclass
class MetricValue:
    """A metric value with timestamp"""
    metric_name: str
    value: float
    timestamp: str
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """An alert triggered by metrics"""
    alert_id: str
    metric_name: str
    severity: AlertSeverity
    message: str
    timestamp: str
    value: float
    threshold: float
    resolved: bool = False
    resolved_at: Optional[str] = None


@dataclass
class LearningEffectMetrics:
    """Metrics for learning effect evaluation"""
    # Knowledge metrics
    knowledge_items_learned: int = 0
    knowledge_retention_rate: float = 0.0
    knowledge_application_rate: float = 0.0
    
    # Performance impact
    accuracy_improvement: float = 0.0
    latency_improvement: float = 0.0
    error_reduction: float = 0.0
    
    # Learning efficiency
    learning_throughput: float = 0.0  # items/hour
    learning_delay_seconds: float = 0.0
    
    # Quality metrics
    false_positive_rate: float = 0.0
    false_negative_rate: float = 0.0


@dataclass
class FeedbackRecord:
    """User feedback record"""
    feedback_id: str
    timestamp: str
    user_id: str
    
    # Feedback data
    rating: int  # 1-5
    category: str  # 'accuracy', 'speed', 'relevance', etc.
    comment: Optional[str] = None
    
    # Context
    version: str = ""
    feature: str = ""
    
    # Processing
    processed: bool = False
    action_taken: Optional[str] = None


class PerformanceMonitor:
    """
    Performance Monitoring Dashboard
    
    Tracks and visualizes system performance with:
    - Real-time metrics
    - Historical trends
    - SLA compliance
    - Alert management
    """
    
    # Predefined metric definitions
    METRIC_DEFINITIONS = {
        "learning_delay_seconds": MetricDefinition(
            name="learning_delay_seconds",
            metric_type=MetricType.GAUGE,
            description="Time from data occurrence to model update",
            unit="seconds",
            warning_threshold=240,
            error_threshold=300,
            target_value=300  # < 5 minutes
        ),
        "iteration_cycle_hours": MetricDefinition(
            name="iteration_cycle_hours",
            metric_type=MetricType.GAUGE,
            description="Time for complete version iteration",
            unit="hours",
            warning_threshold=20,
            error_threshold=24,
            target_value=24  # ≤ 24 hours
        ),
        "system_availability": MetricDefinition(
            name="system_availability",
            metric_type=MetricType.GAUGE,
            description="System availability percentage",
            unit="percent",
            warning_threshold=0.995,
            error_threshold=0.999,
            target_value=0.999  # > 99.9%
        ),
        "merge_success_rate": MetricDefinition(
            name="merge_success_rate",
            metric_type=MetricType.GAUGE,
            description="Automatic merge success rate",
            unit="percent",
            warning_threshold=0.92,
            error_threshold=0.95,
            target_value=0.95  # > 95%
        ),
        "accuracy": MetricDefinition(
            name="accuracy",
            metric_type=MetricType.GAUGE,
            description="Model accuracy",
            unit="percent",
            warning_threshold=0.82,
            error_threshold=0.85,
            target_value=0.85
        ),
        "latency_p95_ms": MetricDefinition(
            name="latency_p95_ms",
            metric_type=MetricType.HISTOGRAM,
            description="P95 response latency",
            unit="milliseconds",
            warning_threshold=180,
            error_threshold=200,
            target_value=200
        ),
        "error_rate": MetricDefinition(
            name="error_rate",
            metric_type=MetricType.GAUGE,
            description="Request error rate",
            unit="percent",
            warning_threshold=0.03,
            error_threshold=0.05,
            target_value=0.02
        ),
        "throughput_rps": MetricDefinition(
            name="throughput_rps",
            metric_type=MetricType.GAUGE,
            description="Requests per second",
            unit="rps",
            warning_threshold=80,
            error_threshold=50,
            target_value=100
        )
    }
    
    def __init__(self):
        self.metric_definitions = dict(self.METRIC_DEFINITIONS)
        self.metric_history: Dict[str, List[MetricValue]] = {}
        self.alerts: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        
        self.max_history_per_metric = 1000
        
        # Callbacks
        self.on_alert: Optional[Callable] = None
        self.on_sla_violation: Optional[Callable] = None
    
    def register_metric(self, definition: MetricDefinition) -> None:
        """Register a new metric"""
        self.metric_definitions[definition.name] = definition
        logger.info(f"Registered metric: {definition.name}")
    
    def record_metric(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ) -> None:
        """Record a metric value"""
        if name not in self.metric_history:
            self.metric_history[name] = []
        
        metric_value = MetricValue(
            metric_name=name,
            value=value,
            timestamp=datetime.now().isoformat(),
            labels=labels or {}
        )
        
        self.metric_history[name].append(metric_value)
        
        # Cleanup old values
        if len(self.metric_history[name]) > self.max_history_per_metric:
            self.metric_history[name] = self.metric_history[name][-self.max_history_per_metric:]
        
        # Check thresholds
        self._check_thresholds(name, value)
    
    def _check_thresholds(self, name: str, value: float) -> None:
        """Check if value exceeds thresholds"""
        if name not in self.metric_definitions:
            return
        
        definition = self.metric_definitions[name]
        
        # Determine if threshold is exceeded
        # For some metrics, lower is better (latency, error_rate)
        # For others, higher is better (availability, accuracy)
        lower_is_better = name in ["latency_p95_ms", "error_rate", "learning_delay_seconds", "iteration_cycle_hours"]
        
        if lower_is_better:
            error_exceeded = definition.error_threshold and value >= definition.error_threshold
            warning_exceeded = definition.warning_threshold and value >= definition.warning_threshold
        else:
            error_exceeded = definition.error_threshold and value <= definition.error_threshold
            warning_exceeded = definition.warning_threshold and value <= definition.warning_threshold
        
        if error_exceeded:
            self._trigger_alert(name, value, AlertSeverity.ERROR, definition.error_threshold)
        elif warning_exceeded:
            self._trigger_alert(name, value, AlertSeverity.WARNING, definition.warning_threshold)
        elif name in self.active_alerts:
            # Resolve active alert
            self._resolve_alert(name)
    
    def _trigger_alert(
        self,
        metric_name: str,
        value: float,
        severity: AlertSeverity,
        threshold: float
    ) -> None:
        """Trigger an alert"""
        if metric_name in self.active_alerts:
            # Alert already active
            return
        
        alert = Alert(
            alert_id=f"alert_{metric_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            metric_name=metric_name,
            severity=severity,
            message=f"{metric_name} is {value:.4f}, threshold: {threshold:.4f}",
            timestamp=datetime.now().isoformat(),
            value=value,
            threshold=threshold
        )
        
        self.alerts.append(alert)
        self.active_alerts[metric_name] = alert
        
        logger.warning(f"Alert triggered: {alert.message}")
        
        if self.on_alert:
            asyncio.create_task(self._async_callback(self.on_alert, alert))
    
    def _resolve_alert(self, metric_name: str) -> None:
        """Resolve an active alert"""
        if metric_name in self.active_alerts:
            alert = self.active_alerts[metric_name]
            alert.resolved = True
            alert.resolved_at = datetime.now().isoformat()
            del self.active_alerts[metric_name]
            
            logger.info(f"Alert resolved: {metric_name}")
    
    async def _async_callback(self, callback: Callable, *args) -> None:
        """Run callback asynchronously"""
        if asyncio.iscoroutinefunction(callback):
            await callback(*args)
        else:
            callback(*args)
    
    def get_metric_history(
        self,
        name: str,
        duration_minutes: int = 60
    ) -> List[MetricValue]:
        """Get metric history for a duration"""
        if name not in self.metric_history:
            return []
        
        cutoff = datetime.now() - timedelta(minutes=duration_minutes)
        
        return [
            v for v in self.metric_history[name]
            if datetime.fromisoformat(v.timestamp) >= cutoff
        ]
    
    def get_metric_statistics(
        self,
        name: str,
        duration_minutes: int = 60
    ) -> Dict[str, float]:
        """Get statistics for a metric"""
        history = self.get_metric_history(name, duration_minutes)
        
        if not history:
            return {}
        
        values = [v.value for v in history]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "stddev": statistics.stdev(values) if len(values) > 1 else 0,
            "latest": values[-1]
        }
    
    def get_sla_compliance(self) -> Dict[str, Any]:
        """Get SLA compliance status"""
        compliance = {}
        
        for name, definition in self.metric_definitions.items():
            if definition.target_value is None:
                continue
            
            stats = self.get_metric_statistics(name, duration_minutes=60)
            
            if not stats:
                compliance[name] = {
                    "status": "no_data",
                    "target": definition.target_value
                }
                continue
            
            current = stats.get("mean", stats.get("latest", 0))
            
            # Determine if meeting target
            lower_is_better = name in ["latency_p95_ms", "error_rate", "learning_delay_seconds", "iteration_cycle_hours"]
            
            if lower_is_better:
                meeting_target = current <= definition.target_value
            else:
                meeting_target = current >= definition.target_value
            
            compliance[name] = {
                "status": "compliant" if meeting_target else "violation",
                "current": current,
                "target": definition.target_value,
                "meeting_target": meeting_target,
                "unit": definition.unit
            }
        
        # Overall compliance
        total = len([c for c in compliance.values() if "current" in c])
        compliant = len([c for c in compliance.values() if c.get("meeting_target", False)])
        
        return {
            "metrics": compliance,
            "overall_compliance": compliant / total if total > 0 else 1.0,
            "compliant_count": compliant,
            "total_count": total
        }
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display"""
        return {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                name: self.get_metric_statistics(name)
                for name in self.metric_definitions
            },
            "sla_compliance": self.get_sla_compliance(),
            "active_alerts": [
                {
                    "alert_id": a.alert_id,
                    "metric": a.metric_name,
                    "severity": a.severity.value,
                    "message": a.message,
                    "timestamp": a.timestamp
                }
                for a in self.active_alerts.values()
            ],
            "system_health": self._calculate_system_health()
        }
    
    def _calculate_system_health(self) -> Dict[str, Any]:
        """Calculate overall system health score"""
        compliance = self.get_sla_compliance()
        
        # Health score based on SLA compliance
        health_score = compliance.get("overall_compliance", 1.0)
        
        # Adjust for active alerts
        for alert in self.active_alerts.values():
            if alert.severity == AlertSeverity.CRITICAL:
                health_score -= 0.3
            elif alert.severity == AlertSeverity.ERROR:
                health_score -= 0.15
            elif alert.severity == AlertSeverity.WARNING:
                health_score -= 0.05
        
        health_score = max(0, min(1, health_score))
        
        if health_score >= 0.95:
            status = "healthy"
        elif health_score >= 0.80:
            status = "degraded"
        else:
            status = "unhealthy"
        
        return {
            "score": health_score,
            "status": status,
            "active_alerts_count": len(self.active_alerts)
        }


class LearningMetrics:
    """
    Learning Effect Evaluation System
    
    Tracks and evaluates the effectiveness of continuous learning.
    """
    
    def __init__(self):
        self.metrics_history: List[LearningEffectMetrics] = []
        self.current_metrics = LearningEffectMetrics()
        
        # Baseline for comparison
        self.baseline_accuracy: float = 0.85
        self.baseline_latency: float = 200.0
        self.baseline_error_rate: float = 0.02
    
    def record_learning_event(
        self,
        items_learned: int,
        learning_delay: float
    ) -> None:
        """Record a learning event"""
        self.current_metrics.knowledge_items_learned += items_learned
        self.current_metrics.learning_delay_seconds = learning_delay
        
        # Calculate throughput (items per hour)
        # Simplified: based on recent delay
        if learning_delay > 0:
            self.current_metrics.learning_throughput = (
                items_learned / (learning_delay / 3600)
            )
    
    def record_performance_metrics(
        self,
        accuracy: float,
        latency_ms: float,
        error_rate: float
    ) -> None:
        """Record performance after learning"""
        # Calculate improvements over baseline
        self.current_metrics.accuracy_improvement = (
            (accuracy - self.baseline_accuracy) / self.baseline_accuracy
        )
        self.current_metrics.latency_improvement = (
            (self.baseline_latency - latency_ms) / self.baseline_latency
        )
        self.current_metrics.error_reduction = (
            (self.baseline_error_rate - error_rate) / self.baseline_error_rate
        )
    
    def record_quality_metrics(
        self,
        false_positives: int,
        false_negatives: int,
        total_predictions: int
    ) -> None:
        """Record quality metrics"""
        if total_predictions > 0:
            self.current_metrics.false_positive_rate = false_positives / total_predictions
            self.current_metrics.false_negative_rate = false_negatives / total_predictions
    
    def snapshot_metrics(self) -> None:
        """Take a snapshot of current metrics"""
        import copy
        self.metrics_history.append(copy.deepcopy(self.current_metrics))
        
        # Keep only last 100 snapshots
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
    
    def get_learning_effectiveness_score(self) -> float:
        """Calculate overall learning effectiveness score"""
        m = self.current_metrics
        
        # Weighted score
        score = 0.0
        
        # Positive factors
        if m.accuracy_improvement > 0:
            score += 0.3 * min(1, m.accuracy_improvement / 0.1)  # Cap at 10% improvement
        
        if m.latency_improvement > 0:
            score += 0.2 * min(1, m.latency_improvement / 0.2)  # Cap at 20% improvement
        
        if m.error_reduction > 0:
            score += 0.2 * min(1, m.error_reduction / 0.5)  # Cap at 50% reduction
        
        # Learning efficiency
        if m.learning_delay_seconds > 0 and m.learning_delay_seconds <= 300:
            score += 0.15 * (1 - m.learning_delay_seconds / 300)
        
        # Quality
        quality_score = 1 - (m.false_positive_rate + m.false_negative_rate)
        score += 0.15 * max(0, quality_score)
        
        return min(1.0, max(0, score))
    
    def get_trend(self) -> Dict[str, str]:
        """Get learning trend over recent history"""
        if len(self.metrics_history) < 2:
            return {"status": "insufficient_data"}
        
        recent = self.metrics_history[-5:]
        
        accuracy_trend = recent[-1].accuracy_improvement - recent[0].accuracy_improvement
        latency_trend = recent[-1].latency_improvement - recent[0].latency_improvement
        
        return {
            "accuracy": "improving" if accuracy_trend > 0 else "declining",
            "latency": "improving" if latency_trend > 0 else "declining",
            "overall": "positive" if accuracy_trend + latency_trend > 0 else "needs_attention"
        }
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive learning metrics report"""
        return {
            "current_metrics": {
                "knowledge_items_learned": self.current_metrics.knowledge_items_learned,
                "learning_throughput": self.current_metrics.learning_throughput,
                "learning_delay_seconds": self.current_metrics.learning_delay_seconds,
                "accuracy_improvement": f"{self.current_metrics.accuracy_improvement:.2%}",
                "latency_improvement": f"{self.current_metrics.latency_improvement:.2%}",
                "error_reduction": f"{self.current_metrics.error_reduction:.2%}",
                "false_positive_rate": f"{self.current_metrics.false_positive_rate:.3%}",
                "false_negative_rate": f"{self.current_metrics.false_negative_rate:.3%}"
            },
            "effectiveness_score": self.get_learning_effectiveness_score(),
            "trend": self.get_trend(),
            "history_length": len(self.metrics_history)
        }


class FeedbackOptimizer:
    """
    Feedback Optimization Closed-Loop System
    
    Collects and processes user feedback to drive improvements.
    """
    
    def __init__(self):
        self.feedback_queue: List[FeedbackRecord] = []
        self.processed_feedback: List[FeedbackRecord] = []
        self.feedback_summary: Dict[str, Any] = {}
        
        # Optimization actions
        self.pending_actions: List[Dict[str, Any]] = []
        self.completed_actions: List[Dict[str, Any]] = []
    
    def record_feedback(
        self,
        user_id: str,
        rating: int,
        category: str,
        comment: Optional[str] = None,
        version: str = "",
        feature: str = ""
    ) -> FeedbackRecord:
        """Record user feedback"""
        feedback = FeedbackRecord(
            feedback_id=f"fb_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id[:8]}",
            timestamp=datetime.now().isoformat(),
            user_id=user_id,
            rating=rating,
            category=category,
            comment=comment,
            version=version,
            feature=feature
        )
        
        self.feedback_queue.append(feedback)
        return feedback
    
    def process_feedback_batch(self) -> Dict[str, Any]:
        """Process pending feedback and generate optimization actions"""
        if not self.feedback_queue:
            return {"status": "no_feedback"}
        
        # Analyze feedback
        by_category: Dict[str, List[int]] = {}
        by_version: Dict[str, List[int]] = {}
        comments: List[str] = []
        
        for fb in self.feedback_queue:
            if fb.category not in by_category:
                by_category[fb.category] = []
            by_category[fb.category].append(fb.rating)
            
            if fb.version:
                if fb.version not in by_version:
                    by_version[fb.version] = []
                by_version[fb.version].append(fb.rating)
            
            if fb.comment:
                comments.append(fb.comment)
            
            fb.processed = True
        
        # Generate summary
        category_scores = {
            cat: statistics.mean(ratings)
            for cat, ratings in by_category.items()
        }
        
        version_scores = {
            ver: statistics.mean(ratings)
            for ver, ratings in by_version.items()
        }
        
        # Identify issues (categories with low ratings)
        issues = [
            cat for cat, score in category_scores.items()
            if score < 3.5
        ]
        
        # Generate optimization actions
        for issue in issues:
            action = {
                "action_id": f"act_{issue}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "type": "improve",
                "category": issue,
                "priority": 5 - int(category_scores.get(issue, 3)),
                "description": f"Improve {issue} based on user feedback",
                "created_at": datetime.now().isoformat()
            }
            self.pending_actions.append(action)
        
        # Move to processed
        self.processed_feedback.extend(self.feedback_queue)
        self.feedback_queue = []
        
        # Update summary
        self.feedback_summary = {
            "total_feedback": len(self.processed_feedback),
            "average_rating": statistics.mean([f.rating for f in self.processed_feedback]),
            "category_scores": category_scores,
            "version_scores": version_scores,
            "issues_identified": issues,
            "pending_actions": len(self.pending_actions)
        }
        
        return self.feedback_summary
    
    def get_pending_actions(self) -> List[Dict[str, Any]]:
        """Get pending optimization actions"""
        return sorted(self.pending_actions, key=lambda a: a.get("priority", 0), reverse=True)
    
    def complete_action(self, action_id: str, result: str) -> None:
        """Mark an action as completed"""
        for action in self.pending_actions:
            if action["action_id"] == action_id:
                action["completed_at"] = datetime.now().isoformat()
                action["result"] = result
                self.completed_actions.append(action)
                self.pending_actions.remove(action)
                break
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get optimization report"""
        return {
            "feedback_summary": self.feedback_summary,
            "pending_actions": len(self.pending_actions),
            "completed_actions": len(self.completed_actions),
            "action_completion_rate": (
                len(self.completed_actions) /
                (len(self.pending_actions) + len(self.completed_actions))
                if (len(self.pending_actions) + len(self.completed_actions)) > 0
                else 0
            ),
            "top_priority_actions": self.get_pending_actions()[:5]
        }
