"""
Performance Tracker - Model Performance Monitoring and Evaluation
Tracks model performance across versions with automated analysis
"""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import numpy as np
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single performance measurement"""
    timestamp: str
    version_id: str
    model_name: str
    metrics: Dict[str, float]
    dataset_info: Dict[str, Any]
    inference_time_ms: float
    memory_usage_mb: float
    batch_size: int
    hardware_info: Dict[str, str]


@dataclass
class PerformanceTrend:
    """Performance trend analysis"""
    metric_name: str
    values: List[float]
    timestamps: List[str]
    trend_direction: str  # 'improving', 'declining', 'stable'
    slope: float
    r_squared: float
    forecast_next: float


@dataclass
class PerformanceAlert:
    """Performance degradation alert"""
    alert_id: str
    timestamp: str
    version_id: str
    metric_name: str
    current_value: float
    baseline_value: float
    degradation_percent: float
    severity: str  # 'warning', 'critical'
    message: str


class PerformanceTracker:
    """
    Model Performance Tracking System
    
    Features:
    - Real-time performance monitoring
    - Historical trend analysis
    - Automated degradation detection
    - A/B testing support
    - Performance forecasting
    """
    
    def __init__(
        self,
        storage_path: str,
        alert_threshold_percent: float = 5.0,
        baseline_window_size: int = 10
    ):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.snapshots_file = self.storage_path / "snapshots.json"
        self.alerts_file = self.storage_path / "alerts.json"
        self.baselines_file = self.storage_path / "baselines.json"
        
        self.alert_threshold = alert_threshold_percent
        self.baseline_window = baseline_window_size
        
        self.snapshots: List[PerformanceSnapshot] = []
        self.alerts: List[PerformanceAlert] = []
        self.baselines: Dict[str, Dict[str, float]] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load tracking data from disk"""
        if self.snapshots_file.exists():
            with open(self.snapshots_file, 'r') as f:
                data = json.load(f)
                self.snapshots = [PerformanceSnapshot(**s) for s in data]
        
        if self.alerts_file.exists():
            with open(self.alerts_file, 'r') as f:
                data = json.load(f)
                self.alerts = [PerformanceAlert(**a) for a in data]
        
        if self.baselines_file.exists():
            with open(self.baselines_file, 'r') as f:
                self.baselines = json.load(f)
    
    def _save_data(self) -> None:
        """Save tracking data to disk"""
        with open(self.snapshots_file, 'w') as f:
            json.dump([asdict(s) for s in self.snapshots], f, indent=2)
        
        with open(self.alerts_file, 'w') as f:
            json.dump([asdict(a) for a in self.alerts], f, indent=2)
        
        with open(self.baselines_file, 'w') as f:
            json.dump(self.baselines, f, indent=2)
    
    def record_performance(
        self,
        version_id: str,
        model_name: str,
        metrics: Dict[str, float],
        dataset_info: Dict[str, Any],
        inference_time_ms: float,
        memory_usage_mb: float,
        batch_size: int = 1,
        hardware_info: Optional[Dict[str, str]] = None
    ) -> PerformanceSnapshot:
        """
        Record a performance snapshot
        
        Args:
            version_id: Model version ID
            model_name: Model name
            metrics: Performance metrics
            dataset_info: Information about test dataset
            inference_time_ms: Inference time in milliseconds
            memory_usage_mb: Memory usage in MB
            batch_size: Batch size used
            hardware_info: Hardware information
            
        Returns:
            Created PerformanceSnapshot
        """
        snapshot = PerformanceSnapshot(
            timestamp=datetime.now().isoformat(),
            version_id=version_id,
            model_name=model_name,
            metrics=metrics,
            dataset_info=dataset_info,
            inference_time_ms=inference_time_ms,
            memory_usage_mb=memory_usage_mb,
            batch_size=batch_size,
            hardware_info=hardware_info or {}
        )
        
        self.snapshots.append(snapshot)
        
        # Check for degradation
        alerts = self._check_degradation(snapshot)
        self.alerts.extend(alerts)
        
        # Update baseline if needed
        self._update_baseline(model_name, metrics)
        
        self._save_data()
        
        return snapshot
    
    def _check_degradation(
        self,
        snapshot: PerformanceSnapshot
    ) -> List[PerformanceAlert]:
        """Check for performance degradation against baseline"""
        alerts = []
        model_name = snapshot.model_name
        
        if model_name not in self.baselines:
            return alerts
        
        baseline = self.baselines[model_name]
        
        for metric_name, current_value in snapshot.metrics.items():
            if metric_name not in baseline:
                continue
            
            baseline_value = baseline[metric_name]
            
            # Calculate degradation (assuming higher is better)
            if baseline_value > 0:
                degradation = (baseline_value - current_value) / baseline_value * 100
            else:
                degradation = 0
            
            if degradation > self.alert_threshold:
                severity = 'critical' if degradation > self.alert_threshold * 2 else 'warning'
                
                alert = PerformanceAlert(
                    alert_id=f"alert_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    timestamp=datetime.now().isoformat(),
                    version_id=snapshot.version_id,
                    metric_name=metric_name,
                    current_value=current_value,
                    baseline_value=baseline_value,
                    degradation_percent=degradation,
                    severity=severity,
                    message=f"Performance degradation detected: {metric_name} dropped by {degradation:.2f}%"
                )
                alerts.append(alert)
                logger.warning(alert.message)
        
        return alerts
    
    def _update_baseline(
        self,
        model_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """Update baseline using moving average"""
        if model_name not in self.baselines:
            self.baselines[model_name] = metrics.copy()
            return
        
        # Get recent snapshots for this model
        recent = [
            s for s in self.snapshots[-self.baseline_window:]
            if s.model_name == model_name
        ]
        
        if len(recent) >= self.baseline_window:
            # Calculate moving average
            for metric_name in metrics:
                values = [s.metrics.get(metric_name, 0) for s in recent]
                self.baselines[model_name][metric_name] = np.mean(values)
    
    def get_performance_history(
        self,
        model_name: str,
        version_id: Optional[str] = None,
        metric_name: Optional[str] = None,  # noqa: ARG002 - reserved for metric filtering
        limit: Optional[int] = None
    ) -> List[PerformanceSnapshot]:
        """Get performance history for a model"""
        snapshots = [
            s for s in self.snapshots
            if s.model_name == model_name
        ]
        
        if version_id:
            snapshots = [s for s in snapshots if s.version_id == version_id]
        
        snapshots.sort(key=lambda x: x.timestamp, reverse=True)
        
        if limit:
            snapshots = snapshots[:limit]
        
        return snapshots
    
    def analyze_trend(
        self,
        model_name: str,
        metric_name: str,
        window_size: int = 20
    ) -> PerformanceTrend:
        """
        Analyze performance trend for a metric
        
        Args:
            model_name: Model name
            metric_name: Metric to analyze
            window_size: Number of recent snapshots to analyze
            
        Returns:
            PerformanceTrend analysis
        """
        snapshots = self.get_performance_history(model_name, limit=window_size)
        snapshots.reverse()  # Oldest first
        
        if len(snapshots) < 2:
            return PerformanceTrend(
                metric_name=metric_name,
                values=[],
                timestamps=[],
                trend_direction='stable',
                slope=0,
                r_squared=0,
                forecast_next=0
            )
        
        values = [s.metrics.get(metric_name, 0) for s in snapshots]
        timestamps = [s.timestamp for s in snapshots]
        
        # Linear regression for trend
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope and R-squared
        if len(x) > 1:
            slope, intercept = np.polyfit(x, y, 1)
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            slope = 0
            r_squared = 0
            intercept = values[0] if values else 0
        
        # Determine trend direction
        if abs(slope) < 0.001:
            direction = 'stable'
        elif slope > 0:
            direction = 'improving'
        else:
            direction = 'declining'
        
        # Forecast next value
        forecast = slope * len(values) + intercept
        
        return PerformanceTrend(
            metric_name=metric_name,
            values=values,
            timestamps=timestamps,
            trend_direction=direction,
            slope=float(slope),
            r_squared=float(r_squared),
            forecast_next=float(forecast)
        )
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Compare performance between two versions
        
        Args:
            version_a: First version ID
            version_b: Second version ID
            model_name: Model name
            
        Returns:
            Comparison results
        """
        snapshots_a = [
            s for s in self.snapshots
            if s.version_id == version_a and s.model_name == model_name
        ]
        snapshots_b = [
            s for s in self.snapshots
            if s.version_id == version_b and s.model_name == model_name
        ]
        
        if not snapshots_a or not snapshots_b:
            raise ValueError("Insufficient data for comparison")
        
        # Calculate average metrics for each version
        def avg_metrics(snapshots: List[PerformanceSnapshot]) -> Dict[str, float]:
            all_metrics = defaultdict(list)
            for s in snapshots:
                for k, v in s.metrics.items():
                    all_metrics[k].append(v)
            return {k: np.mean(v) for k, v in all_metrics.items()}
        
        avg_a = avg_metrics(snapshots_a)
        avg_b = avg_metrics(snapshots_b)
        
        # Calculate differences
        comparison = {}
        all_metrics = set(avg_a.keys()) | set(avg_b.keys())
        
        for metric in all_metrics:
            val_a = avg_a.get(metric, 0)
            val_b = avg_b.get(metric, 0)
            diff = val_b - val_a
            pct_change = (diff / val_a * 100) if val_a != 0 else 0
            
            comparison[metric] = {
                'version_a': val_a,
                'version_b': val_b,
                'difference': diff,
                'percent_change': pct_change,
                'improved': diff > 0
            }
        
        # Overall winner
        improvements = sum(1 for m in comparison.values() if m['improved'])
        winner = version_b if improvements > len(comparison) / 2 else version_a
        
        return {
            'version_a': version_a,
            'version_b': version_b,
            'metrics_comparison': comparison,
            'winner': winner,
            'confidence': improvements / len(comparison) if comparison else 0
        }
    
    def get_alerts(
        self,
        model_name: Optional[str] = None,
        severity: Optional[str] = None,
        since: Optional[str] = None
    ) -> List[PerformanceAlert]:
        """Get performance alerts"""
        alerts = self.alerts.copy()
        
        if model_name:
            alerts = [
                a for a in alerts
                if any(
                    s.model_name == model_name and s.version_id == a.version_id
                    for s in self.snapshots
                )
            ]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def generate_report(
        self,
        model_name: str,
        include_trends: bool = True,
        include_alerts: bool = True
    ) -> Dict[str, Any]:
        """
        Generate comprehensive performance report
        
        Args:
            model_name: Model name
            include_trends: Include trend analysis
            include_alerts: Include alerts
            
        Returns:
            Performance report dictionary
        """
        snapshots = self.get_performance_history(model_name)
        
        if not snapshots:
            return {'error': 'No performance data found'}
        
        latest = snapshots[0]
        
        report = {
            'model_name': model_name,
            'generated_at': datetime.now().isoformat(),
            'total_snapshots': len(snapshots),
            'latest_performance': {
                'version': latest.version_id,
                'timestamp': latest.timestamp,
                'metrics': latest.metrics,
                'inference_time_ms': latest.inference_time_ms,
                'memory_usage_mb': latest.memory_usage_mb
            },
            'baseline': self.baselines.get(model_name, {})
        }
        
        if include_trends:
            trends = {}
            for metric in latest.metrics:
                trend = self.analyze_trend(model_name, metric)
                trends[metric] = {
                    'direction': trend.trend_direction,
                    'slope': trend.slope,
                    'r_squared': trend.r_squared,
                    'forecast': trend.forecast_next
                }
            report['trends'] = trends
        
        if include_alerts:
            alerts = self.get_alerts(model_name)
            report['recent_alerts'] = [
                {
                    'timestamp': a.timestamp,
                    'metric': a.metric_name,
                    'severity': a.severity,
                    'message': a.message
                }
                for a in alerts[:10]
            ]
        
        return report
    
    def set_baseline(
        self,
        model_name: str,
        metrics: Dict[str, float]
    ) -> None:
        """Manually set baseline metrics"""
        self.baselines[model_name] = metrics.copy()
        self._save_data()
        logger.info(f"Set baseline for {model_name}: {metrics}")
    
    def clear_alerts(
        self,
        model_name: Optional[str] = None,
        before: Optional[str] = None
    ) -> int:
        """Clear alerts"""
        original_count = len(self.alerts)
        
        if model_name:
            version_ids = {
                s.version_id for s in self.snapshots
                if s.model_name == model_name
            }
            self.alerts = [
                a for a in self.alerts
                if a.version_id not in version_ids
            ]
        elif before:
            self.alerts = [a for a in self.alerts if a.timestamp >= before]
        else:
            self.alerts = []
        
        cleared = original_count - len(self.alerts)
        self._save_data()
        
        return cleared
