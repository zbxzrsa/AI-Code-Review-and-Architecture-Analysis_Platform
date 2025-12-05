"""
Version Evolution Tracker for V1 VC-AI

Tracks model and architecture evolution:
- Model parameter history
- Architecture modifications
- Training metrics over time
- Performance deltas
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
import json
import hashlib


class ModificationType(str, Enum):
    """Types of model modifications"""
    ARCHITECTURE_CHANGE = "architecture_change"
    HYPERPARAMETER_TUNE = "hyperparameter_tune"
    DATA_UPDATE = "data_update"
    QUANTIZATION = "quantization"
    PRUNING = "pruning"
    LORA_ADAPTATION = "lora_adaptation"
    FINE_TUNING = "fine_tuning"
    CHECKPOINT_RESTORE = "checkpoint_restore"


@dataclass
class ModelVersion:
    """A snapshot of model state"""
    version_id: str
    parent_version_id: Optional[str]
    timestamp: datetime
    
    # Model configuration
    model_name: str
    architecture_config: Dict[str, Any]
    training_config: Dict[str, Any]
    
    # Metrics at this version
    metrics: Dict[str, float]
    
    # Modification details
    modification_type: ModificationType
    modification_description: str
    modification_rationale: str
    
    # Storage
    checkpoint_path: Optional[str] = None
    config_hash: str = ""
    
    def __post_init__(self):
        if not self.config_hash:
            self.config_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash of configuration for deduplication"""
        config_str = json.dumps({
            "architecture": self.architecture_config,
            "training": self.training_config,
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version_id": self.version_id,
            "parent_version_id": self.parent_version_id,
            "timestamp": self.timestamp.isoformat(),
            "model_name": self.model_name,
            "architecture_config": self.architecture_config,
            "training_config": self.training_config,
            "metrics": self.metrics,
            "modification_type": self.modification_type.value,
            "modification_description": self.modification_description,
            "modification_rationale": self.modification_rationale,
            "checkpoint_path": self.checkpoint_path,
            "config_hash": self.config_hash,
        }


@dataclass
class ExperimentRecord:
    """Record of an experiment run"""
    experiment_id: str
    version_id: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    
    # Configuration
    experiment_name: str = ""
    hypothesis: str = ""
    variables_tested: List[str] = field(default_factory=list)
    
    # Results
    status: str = "running"  # running, completed, failed, cancelled
    final_metrics: Dict[str, float] = field(default_factory=dict)
    metric_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Analysis
    key_findings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Promotion
    promoted_to_v2: bool = False
    promotion_date: Optional[datetime] = None


class EvolutionTracker:
    """
    Tracks model evolution over time.
    
    Maintains history of:
    - Model versions and modifications
    - Training metrics progression
    - Performance deltas between versions
    - Experiment records
    """
    
    def __init__(self):
        self.versions: Dict[str, ModelVersion] = {}
        self.experiments: Dict[str, ExperimentRecord] = {}
        self.current_version_id: Optional[str] = None
        self.version_lineage: Dict[str, List[str]] = {}  # parent -> children
    
    def record_initial_state(
        self,
        version_id: str,
        model_name: str,
        architecture_config: Dict[str, Any],
        training_config: Dict[str, Any],
        initial_metrics: Dict[str, float],
    ) -> ModelVersion:
        """
        Record the initial model state.
        
        Args:
            version_id: Unique version identifier
            model_name: Name of the base model
            architecture_config: Architecture configuration
            training_config: Training configuration
            initial_metrics: Initial evaluation metrics
            
        Returns:
            Created ModelVersion
        """
        version = ModelVersion(
            version_id=version_id,
            parent_version_id=None,
            timestamp=datetime.now(timezone.utc),
            model_name=model_name,
            architecture_config=architecture_config,
            training_config=training_config,
            metrics=initial_metrics,
            modification_type=ModificationType.CHECKPOINT_RESTORE,
            modification_description="Initial model state",
            modification_rationale="Starting point for experimentation",
        )
        
        self.versions[version_id] = version
        self.current_version_id = version_id
        self.version_lineage[version_id] = []
        
        return version
    
    def log_modification(
        self,
        version_id: str,
        modification_type: ModificationType,
        description: str,
        rationale: str,
        new_architecture_config: Optional[Dict[str, Any]] = None,
        new_training_config: Optional[Dict[str, Any]] = None,
        new_metrics: Optional[Dict[str, float]] = None,
        checkpoint_path: Optional[str] = None,
    ) -> ModelVersion:
        """
        Log a modification to the model.
        
        Args:
            version_id: New version identifier
            modification_type: Type of modification
            description: Description of the change
            rationale: Why this change was made
            new_architecture_config: Updated architecture (optional)
            new_training_config: Updated training config (optional)
            new_metrics: New metrics after modification
            checkpoint_path: Path to saved checkpoint
            
        Returns:
            Created ModelVersion
        """
        parent_version = self.versions.get(self.current_version_id)
        if not parent_version:
            raise ValueError("No current version to modify")
        
        # Inherit configs from parent if not provided
        architecture_config = new_architecture_config or parent_version.architecture_config
        training_config = new_training_config or parent_version.training_config
        metrics = new_metrics or parent_version.metrics
        
        version = ModelVersion(
            version_id=version_id,
            parent_version_id=self.current_version_id,
            timestamp=datetime.now(timezone.utc),
            model_name=parent_version.model_name,
            architecture_config=architecture_config,
            training_config=training_config,
            metrics=metrics,
            modification_type=modification_type,
            modification_description=description,
            modification_rationale=rationale,
            checkpoint_path=checkpoint_path,
        )
        
        self.versions[version_id] = version
        
        # Update lineage
        if self.current_version_id in self.version_lineage:
            self.version_lineage[self.current_version_id].append(version_id)
        self.version_lineage[version_id] = []
        
        self.current_version_id = version_id
        
        return version
    
    def capture_training_metrics(
        self,
        experiment_id: str,
        step: int,
        metrics: Dict[str, float],
    ):
        """Capture training metrics for an experiment"""
        if experiment_id not in self.experiments:
            return
        
        self.experiments[experiment_id].metric_history.append({
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **metrics,
        })
    
    def start_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        hypothesis: str,
        variables_tested: List[str],
    ) -> ExperimentRecord:
        """
        Start tracking a new experiment.
        
        Args:
            experiment_id: Unique experiment identifier
            experiment_name: Human-readable name
            hypothesis: What we're testing
            variables_tested: Variables being experimented with
            
        Returns:
            Created ExperimentRecord
        """
        if not self.current_version_id:
            raise ValueError("No model version to experiment with")
        
        record = ExperimentRecord(
            experiment_id=experiment_id,
            version_id=self.current_version_id,
            started_at=datetime.now(timezone.utc),
            experiment_name=experiment_name,
            hypothesis=hypothesis,
            variables_tested=variables_tested,
        )
        
        self.experiments[experiment_id] = record
        return record
    
    def complete_experiment(
        self,
        experiment_id: str,
        final_metrics: Dict[str, float],
        key_findings: List[str],
        recommendations: List[str],
        status: str = "completed",
    ) -> ExperimentRecord:
        """
        Complete an experiment with results.
        
        Args:
            experiment_id: Experiment to complete
            final_metrics: Final evaluation metrics
            key_findings: Key insights from the experiment
            recommendations: Recommended next steps
            status: Final status (completed, failed, cancelled)
            
        Returns:
            Updated ExperimentRecord
        """
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        record = self.experiments[experiment_id]
        record.completed_at = datetime.now(timezone.utc)
        record.status = status
        record.final_metrics = final_metrics
        record.key_findings = key_findings
        record.recommendations = recommendations
        
        return record
    
    def get_version_history(self, version_id: Optional[str] = None) -> List[ModelVersion]:
        """Get the full history leading to a version"""
        target_id = version_id or self.current_version_id
        history = []
        
        current_id = target_id
        while current_id:
            version = self.versions.get(current_id)
            if not version:
                break
            history.append(version)
            current_id = version.parent_version_id
        
        return list(reversed(history))
    
    def compute_performance_delta(
        self,
        version_id_a: str,
        version_id_b: str,
    ) -> Dict[str, float]:
        """
        Compute performance difference between two versions.
        
        Returns:
            Dict of metric_name -> delta (b - a)
        """
        version_a = self.versions.get(version_id_a)
        version_b = self.versions.get(version_id_b)
        
        if not version_a or not version_b:
            raise ValueError("Version not found")
        
        delta = {}
        all_metrics = set(version_a.metrics.keys()) | set(version_b.metrics.keys())
        
        for metric in all_metrics:
            val_a = version_a.metrics.get(metric, 0)
            val_b = version_b.metrics.get(metric, 0)
            delta[metric] = val_b - val_a
        
        return delta
    
    def find_best_version(self, metric: str, higher_is_better: bool = True) -> Optional[ModelVersion]:
        """Find the version with the best value for a metric"""
        if not self.versions:
            return None
        
        best_version = None
        best_value = None
        
        for version in self.versions.values():
            value = version.metrics.get(metric)
            if value is None:
                continue
            
            if best_value is None:
                best_value = value
                best_version = version
            elif higher_is_better and value > best_value:
                best_value = value
                best_version = version
            elif not higher_is_better and value < best_value:
                best_value = value
                best_version = version
        
        return best_version
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get a summary of an experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        record = self.experiments[experiment_id]
        
        # Compute metric trends
        trends = {}
        if record.metric_history:
            for metric in record.metric_history[0].keys():
                if metric in ["step", "timestamp"]:
                    continue
                values = [m.get(metric) for m in record.metric_history if m.get(metric) is not None]
                if values:
                    trends[metric] = {
                        "start": values[0],
                        "end": values[-1],
                        "min": min(values),
                        "max": max(values),
                        "delta": values[-1] - values[0],
                    }
        
        duration = None
        if record.completed_at:
            duration = (record.completed_at - record.started_at).total_seconds()
        
        return {
            "experiment_id": experiment_id,
            "experiment_name": record.experiment_name,
            "hypothesis": record.hypothesis,
            "variables_tested": record.variables_tested,
            "status": record.status,
            "duration_seconds": duration,
            "final_metrics": record.final_metrics,
            "metric_trends": trends,
            "key_findings": record.key_findings,
            "recommendations": record.recommendations,
            "promoted": record.promoted_to_v2,
        }
    
    def export_evolution_report(self) -> Dict[str, Any]:
        """Export complete evolution report"""
        return {
            "total_versions": len(self.versions),
            "current_version": self.current_version_id,
            "total_experiments": len(self.experiments),
            "completed_experiments": sum(1 for e in self.experiments.values() if e.status == "completed"),
            "promoted_experiments": sum(1 for e in self.experiments.values() if e.promoted_to_v2),
            "version_history": [v.to_dict() for v in self.get_version_history()],
            "experiment_summaries": [
                self.get_experiment_summary(eid)
                for eid in self.experiments.keys()
            ],
        }
