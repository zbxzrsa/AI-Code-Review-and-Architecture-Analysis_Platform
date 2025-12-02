"""
Version Tracker - Detailed Version Lineage and Dependency Tracking
Tracks model evolution, experiments, and dependencies
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PROMOTED = "promoted"
    QUARANTINED = "quarantined"


@dataclass
class Experiment:
    """Training experiment record"""
    experiment_id: str
    name: str
    model_name: str
    parent_version: Optional[str]
    status: ExperimentStatus
    created_at: str
    started_at: Optional[str]
    completed_at: Optional[str]
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, float]
    dataset_id: str
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    resulting_version: Optional[str] = None
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Experiment':
        data['status'] = ExperimentStatus(data['status'])
        return cls(**data)


@dataclass
class VersionNode:
    """Node in version lineage graph"""
    version_id: str
    model_name: str
    parent_id: Optional[str]
    children_ids: List[str]
    experiment_id: Optional[str]
    created_at: str
    metrics: Dict[str, float]
    is_production: bool = False
    is_best: bool = False
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VersionNode':
        return cls(**data)


@dataclass
class ModelDependency:
    """Model dependency information"""
    model_name: str
    version_id: str
    dependency_type: str  # 'teacher', 'ensemble_member', 'feature_extractor'
    weight: float = 1.0


class VersionTracker:
    """
    Version Lineage and Experiment Tracker
    
    Features:
    - Version lineage graph
    - Experiment lifecycle management
    - Dependency tracking
    - Best version identification
    - Auto-evolution support
    """
    
    def __init__(self, storage_path: str):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.experiments_file = self.storage_path / "experiments.json"
        self.lineage_file = self.storage_path / "lineage.json"
        self.dependencies_file = self.storage_path / "dependencies.json"
        
        self.experiments: Dict[str, Experiment] = {}
        self.lineage: Dict[str, VersionNode] = {}
        self.dependencies: Dict[str, List[ModelDependency]] = {}
        
        self._load_data()
    
    def _load_data(self) -> None:
        """Load tracker data"""
        if self.experiments_file.exists():
            with open(self.experiments_file, 'r') as f:
                data = json.load(f)
                self.experiments = {
                    k: Experiment.from_dict(v) for k, v in data.items()
                }
        
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                data = json.load(f)
                self.lineage = {
                    k: VersionNode.from_dict(v) for k, v in data.items()
                }
        
        if self.dependencies_file.exists():
            with open(self.dependencies_file, 'r') as f:
                data = json.load(f)
                self.dependencies = {
                    k: [ModelDependency(**d) for d in v]
                    for k, v in data.items()
                }
    
    def _save_data(self) -> None:
        """Save tracker data"""
        with open(self.experiments_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.experiments.items()},
                f, indent=2
            )
        
        with open(self.lineage_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.lineage.items()},
                f, indent=2
            )
        
        with open(self.dependencies_file, 'w') as f:
            json.dump(
                {k: [asdict(d) for d in v] for k, v in self.dependencies.items()},
                f, indent=2
            )
    
    def _generate_experiment_id(self, name: str) -> str:
        """Generate unique experiment ID"""
        content = f"{name}:{datetime.now().isoformat()}"
        return f"exp_{hashlib.sha256(content.encode()).hexdigest()[:12]}"
    
    # Experiment Management
    
    def create_experiment(
        self,
        name: str,
        model_name: str,
        hyperparameters: Dict[str, Any],
        dataset_id: str,
        parent_version: Optional[str] = None,
        notes: str = "",
        tags: Optional[List[str]] = None
    ) -> Experiment:
        """Create a new experiment"""
        exp_id = self._generate_experiment_id(name)
        
        experiment = Experiment(
            experiment_id=exp_id,
            name=name,
            model_name=model_name,
            parent_version=parent_version,
            status=ExperimentStatus.PENDING,
            created_at=datetime.now().isoformat(),
            started_at=None,
            completed_at=None,
            hyperparameters=hyperparameters,
            metrics={},
            dataset_id=dataset_id,
            notes=notes,
            tags=tags or []
        )
        
        self.experiments[exp_id] = experiment
        self._save_data()
        
        logger.info(f"Created experiment: {exp_id} - {name}")
        return experiment
    
    def start_experiment(self, experiment_id: str) -> Experiment:
        """Mark experiment as started"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now().isoformat()
        
        self._save_data()
        return exp
    
    def complete_experiment(
        self,
        experiment_id: str,
        metrics: Dict[str, float],
        resulting_version: Optional[str] = None
    ) -> Experiment:
        """Mark experiment as completed"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.COMPLETED
        exp.completed_at = datetime.now().isoformat()
        exp.metrics = metrics
        exp.resulting_version = resulting_version
        
        self._save_data()
        logger.info(f"Completed experiment: {experiment_id}")
        return exp
    
    def fail_experiment(
        self,
        experiment_id: str,
        reason: str
    ) -> Experiment:
        """Mark experiment as failed"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.FAILED
        exp.completed_at = datetime.now().isoformat()
        exp.notes = f"{exp.notes}\nFailed: {reason}"
        
        self._save_data()
        logger.warning(f"Experiment failed: {experiment_id} - {reason}")
        return exp
    
    def promote_experiment(
        self,
        experiment_id: str,
        version_id: str
    ) -> Experiment:
        """Promote experiment result to production"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.PROMOTED
        exp.resulting_version = version_id
        
        # Update lineage
        if version_id in self.lineage:
            self.lineage[version_id].is_production = True
        
        self._save_data()
        logger.info(f"Promoted experiment: {experiment_id}")
        return exp
    
    def quarantine_experiment(
        self,
        experiment_id: str,
        reason: str
    ) -> Experiment:
        """Quarantine a problematic experiment"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        exp = self.experiments[experiment_id]
        exp.status = ExperimentStatus.QUARANTINED
        exp.notes = f"{exp.notes}\nQuarantined: {reason}"
        
        self._save_data()
        logger.warning(f"Quarantined experiment: {experiment_id}")
        return exp
    
    # Version Lineage
    
    def register_version(
        self,
        version_id: str,
        model_name: str,
        metrics: Dict[str, float],
        parent_id: Optional[str] = None,
        experiment_id: Optional[str] = None
    ) -> VersionNode:
        """Register a version in the lineage graph"""
        node = VersionNode(
            version_id=version_id,
            model_name=model_name,
            parent_id=parent_id,
            children_ids=[],
            experiment_id=experiment_id,
            created_at=datetime.now().isoformat(),
            metrics=metrics
        )
        
        # Update parent's children list
        if parent_id and parent_id in self.lineage:
            self.lineage[parent_id].children_ids.append(version_id)
        
        self.lineage[version_id] = node
        
        # Update best version
        self._update_best_version(model_name)
        
        self._save_data()
        return node
    
    def _update_best_version(self, model_name: str) -> None:
        """Update the best version flag for a model"""
        model_versions = [
            v for v in self.lineage.values()
            if v.model_name == model_name
        ]
        
        if not model_versions:
            return
        
        # Find best by primary metric (assuming 'accuracy')
        best = max(
            model_versions,
            key=lambda v: v.metrics.get('accuracy', 0)
        )
        
        # Update flags
        for v in model_versions:
            v.is_best = (v.version_id == best.version_id)
    
    def get_lineage(
        self,
        version_id: str,
        depth: int = -1
    ) -> Dict[str, Any]:
        """
        Get version lineage tree
        
        Args:
            version_id: Starting version
            depth: Maximum depth (-1 for unlimited)
            
        Returns:
            Lineage tree structure
        """
        if version_id not in self.lineage:
            return {}
        
        def build_tree(vid: str, current_depth: int) -> Dict:
            if vid not in self.lineage:
                return {}
            
            node = self.lineage[vid]
            tree = {
                'version_id': vid,
                'model_name': node.model_name,
                'metrics': node.metrics,
                'is_production': node.is_production,
                'is_best': node.is_best,
                'created_at': node.created_at
            }
            
            if depth == -1 or current_depth < depth:
                tree['children'] = [
                    build_tree(cid, current_depth + 1)
                    for cid in node.children_ids
                ]
            
            return tree
        
        # Find root
        current = version_id
        while self.lineage[current].parent_id:
            current = self.lineage[current].parent_id
        
        return build_tree(current, 0)
    
    def get_ancestors(self, version_id: str) -> List[str]:
        """Get all ancestor versions"""
        ancestors = []
        current = version_id
        
        while current in self.lineage:
            parent = self.lineage[current].parent_id
            if parent:
                ancestors.append(parent)
                current = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, version_id: str) -> List[str]:
        """Get all descendant versions"""
        if version_id not in self.lineage:
            return []
        
        descendants = []
        queue = list(self.lineage[version_id].children_ids)
        
        while queue:
            current = queue.pop(0)
            descendants.append(current)
            if current in self.lineage:
                queue.extend(self.lineage[current].children_ids)
        
        return descendants
    
    # Dependencies
    
    def add_dependency(
        self,
        version_id: str,
        dependency: ModelDependency
    ) -> None:
        """Add a model dependency"""
        if version_id not in self.dependencies:
            self.dependencies[version_id] = []
        
        self.dependencies[version_id].append(dependency)
        self._save_data()
    
    def get_dependencies(
        self,
        version_id: str,
        dependency_type: Optional[str] = None
    ) -> List[ModelDependency]:
        """Get dependencies for a version"""
        deps = self.dependencies.get(version_id, [])
        
        if dependency_type:
            deps = [d for d in deps if d.dependency_type == dependency_type]
        
        return deps
    
    def get_dependents(self, version_id: str) -> List[str]:
        """Get versions that depend on this version"""
        dependents = []
        
        for vid, deps in self.dependencies.items():
            for dep in deps:
                if dep.version_id == version_id:
                    dependents.append(vid)
                    break
        
        return dependents
    
    # Query Methods
    
    def get_experiments(
        self,
        model_name: Optional[str] = None,
        status: Optional[ExperimentStatus] = None,
        limit: Optional[int] = None
    ) -> List[Experiment]:
        """Query experiments"""
        experiments = list(self.experiments.values())
        
        if model_name:
            experiments = [e for e in experiments if e.model_name == model_name]
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        experiments.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            experiments = experiments[:limit]
        
        return experiments
    
    def get_best_version(
        self,
        model_name: str,
        metric: str = 'accuracy'
    ) -> Optional[VersionNode]:
        """Get the best version for a model"""
        versions = [
            v for v in self.lineage.values()
            if v.model_name == model_name
        ]
        
        if not versions:
            return None
        
        return max(versions, key=lambda v: v.metrics.get(metric, 0))
    
    def get_production_version(
        self,
        model_name: str
    ) -> Optional[VersionNode]:
        """Get the current production version"""
        for v in self.lineage.values():
            if v.model_name == model_name and v.is_production:
                return v
        return None
    
    def suggest_next_experiment(
        self,
        model_name: str
    ) -> Dict[str, Any]:
        """
        Suggest next experiment based on history
        
        Returns:
            Suggested experiment configuration
        """
        # Get completed experiments
        completed = [
            e for e in self.experiments.values()
            if e.model_name == model_name and 
               e.status == ExperimentStatus.COMPLETED
        ]
        
        if not completed:
            return {
                'suggestion': 'baseline',
                'message': 'No completed experiments. Start with baseline.',
                'hyperparameters': {}
            }
        
        # Find best experiment
        best = max(completed, key=lambda e: e.metrics.get('accuracy', 0))
        
        # Suggest variations
        suggestions = []
        hp = best.hyperparameters.copy()
        
        # Learning rate variations
        if 'learning_rate' in hp:
            suggestions.append({
                'name': 'Lower learning rate',
                'hyperparameters': {**hp, 'learning_rate': hp['learning_rate'] * 0.5}
            })
            suggestions.append({
                'name': 'Higher learning rate',
                'hyperparameters': {**hp, 'learning_rate': hp['learning_rate'] * 2}
            })
        
        # Batch size variations
        if 'batch_size' in hp:
            suggestions.append({
                'name': 'Larger batch',
                'hyperparameters': {**hp, 'batch_size': hp['batch_size'] * 2}
            })
        
        return {
            'suggestion': 'variation',
            'message': f'Based on best experiment: {best.experiment_id}',
            'best_metrics': best.metrics,
            'variations': suggestions
        }
    
    def get_evolution_stats(self, model_name: str) -> Dict[str, Any]:
        """Get evolution statistics for a model"""
        versions = [
            v for v in self.lineage.values()
            if v.model_name == model_name
        ]
        
        experiments = [
            e for e in self.experiments.values()
            if e.model_name == model_name
        ]
        
        if not versions:
            return {'error': 'No versions found'}
        
        # Calculate stats
        metrics_over_time = []
        for v in sorted(versions, key=lambda x: x.created_at):
            metrics_over_time.append({
                'version': v.version_id,
                'timestamp': v.created_at,
                'metrics': v.metrics
            })
        
        # Success rate
        completed = [e for e in experiments if e.status == ExperimentStatus.COMPLETED]
        failed = [e for e in experiments if e.status == ExperimentStatus.FAILED]
        success_rate = len(completed) / (len(completed) + len(failed)) if experiments else 0
        
        return {
            'model_name': model_name,
            'total_versions': len(versions),
            'total_experiments': len(experiments),
            'success_rate': success_rate,
            'best_version': self.get_best_version(model_name).version_id if versions else None,
            'production_version': self.get_production_version(model_name).version_id if self.get_production_version(model_name) else None,
            'metrics_evolution': metrics_over_time
        }
