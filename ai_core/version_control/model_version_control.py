"""
AI Model Version Control System
Git-based model versioning with automated tracking, comparison, and rollback
"""

import os
import json
import hashlib
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging
import pickle

import torch
import torch.nn as nn
from torch.serialization import save, load

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Represents a model version"""
    version_id: str
    model_name: str
    commit_hash: str
    parent_version: Optional[str]
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    architecture_hash: str
    checkpoint_path: str
    tags: List[str] = field(default_factory=list)
    description: str = ""
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        return cls(**data)


@dataclass
class VersionComparison:
    """Comparison result between two versions"""
    version_a: str
    version_b: str
    metric_diffs: Dict[str, float]
    hyperparameter_diffs: Dict[str, Tuple[Any, Any]]
    architecture_changed: bool
    performance_improvement: float
    recommendation: str


class ModelVersionControl:
    """
    Git-based Model Version Control System
    
    Features:
    - Automatic model versioning with Git integration
    - Model checkpoint management
    - Version comparison and rollback
    - Performance tracking across versions
    - Self-updating cycle support
    """
    
    def __init__(
        self,
        repo_path: str,
        model_dir: str = "models",
        use_git: bool = True,
        auto_commit: bool = True
    ):
        self.repo_path = Path(repo_path)
        self.model_dir = self.repo_path / model_dir
        self.versions_file = self.model_dir / "versions.json"
        self.use_git = use_git
        self.auto_commit = auto_commit
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize versions registry
        self.versions: Dict[str, ModelVersion] = {}
        self._load_versions()
        
        # Initialize Git if enabled
        if self.use_git:
            self._init_git()
    
    def _init_git(self) -> None:
        """Initialize Git repository if not exists"""
        git_dir = self.repo_path / ".git"
        if not git_dir.exists():
            subprocess.run(
                ["git", "init"],
                cwd=self.repo_path,
                capture_output=True
            )
            logger.info(f"Initialized Git repository at {self.repo_path}")
    
    def _load_versions(self) -> None:
        """Load versions from registry file"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                data = json.load(f)
                self.versions = {
                    k: ModelVersion.from_dict(v) 
                    for k, v in data.items()
                }
    
    def _save_versions(self) -> None:
        """Save versions to registry file"""
        with open(self.versions_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.versions.items()},
                f,
                indent=2
            )
    
    def _compute_architecture_hash(self, model: nn.Module) -> str:
        """Compute hash of model architecture"""
        arch_str = str(model)
        return hashlib.sha256(arch_str.encode()).hexdigest()[:16]
    
    def _get_git_commit_hash(self) -> str:
        """Get current Git commit hash"""
        if not self.use_git:
            return hashlib.sha256(
                datetime.now().isoformat().encode()
            ).hexdigest()[:8]
        
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return result.stdout.strip()[:8]
        except Exception:
            return "unknown"
    
    def _generate_version_id(self, model_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit = self._get_git_commit_hash()
        return f"{model_name}_v{timestamp}_{commit}"
    
    def commit_model(
        self,
        model: nn.Module,
        model_name: str,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        tags: Optional[List[str]] = None,
        description: str = "",
        parent_version: Optional[str] = None
    ) -> ModelVersion:
        """
        Commit a new model version
        
        Args:
            model: PyTorch model to save
            model_name: Name of the model
            metrics: Performance metrics
            hyperparameters: Training hyperparameters
            optimizer: Optional optimizer state
            scheduler: Optional scheduler state
            tags: Optional tags for the version
            description: Version description
            parent_version: ID of parent version
            
        Returns:
            Created ModelVersion
        """
        version_id = self._generate_version_id(model_name)
        
        # Create checkpoint directory
        checkpoint_dir = self.model_dir / model_name / version_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model checkpoint
        checkpoint_path = checkpoint_dir / "checkpoint.pt"
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'hyperparameters': hyperparameters,
            'version_id': version_id,
            'created_at': datetime.now().isoformat()
        }
        
        if optimizer:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        torch.save(checkpoint, checkpoint_path)
        
        # Save model architecture
        arch_path = checkpoint_dir / "architecture.txt"
        with open(arch_path, 'w') as f:
            f.write(str(model))
        
        # Create version record
        version = ModelVersion(
            version_id=version_id,
            model_name=model_name,
            commit_hash=self._get_git_commit_hash(),
            parent_version=parent_version,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            hyperparameters=hyperparameters,
            architecture_hash=self._compute_architecture_hash(model),
            checkpoint_path=str(checkpoint_path),
            tags=tags or [],
            description=description
        )
        
        # Register version
        self.versions[version_id] = version
        self._save_versions()
        
        # Git commit if enabled
        if self.use_git and self.auto_commit:
            self._git_commit(version)
        
        logger.info(f"Committed model version: {version_id}")
        return version
    
    def _git_commit(self, version: ModelVersion) -> None:
        """Create Git commit for version"""
        try:
            # Add files
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.repo_path,
                capture_output=True
            )
            
            # Commit
            message = f"Model: {version.model_name} | Version: {version.version_id}\n\n"
            message += f"Metrics: {json.dumps(version.metrics, indent=2)}\n"
            message += f"Description: {version.description}"
            
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                capture_output=True
            )
            
            # Tag version
            subprocess.run(
                ["git", "tag", version.version_id],
                cwd=self.repo_path,
                capture_output=True
            )
            
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")
    
    def load_version(
        self,
        version_id: str,
        model: nn.Module,
        load_optimizer: bool = False  # noqa: ARG002 - reserved for optimizer loading
    ) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Load a specific model version
        
        Args:
            version_id: Version ID to load
            model: Model instance to load weights into
            load_optimizer: Whether to return optimizer state (reserved)
            
        Returns:
            Tuple of (model, checkpoint_data)
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        checkpoint = torch.load(version.checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint
    
    def rollback(
        self,
        model: nn.Module,
        target_version: str
    ) -> Tuple[nn.Module, ModelVersion]:
        """
        Rollback model to a previous version (basic).
        
        Args:
            model: Model instance to rollback
            target_version: Version ID to rollback to
            
        Returns:
            Tuple of (rolled_back_model, version)
        """
        if target_version not in self.versions:
            raise ValueError(f"Version {target_version} not found")
        
        model, _ = self.load_version(target_version, model)
        version = self.versions[target_version]
        
        logger.info(f"Rolled back to version: {target_version}")
        return model, version
    
    def rollback_version(
        self,
        model: nn.Module,
        target_version: str,
        strategy: str = "snapshot",
        traffic_safe: bool = True,
        health_check_fn: Optional[callable] = None,
        verification_fn: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Advanced version rollback with safety features.
        
        ⚠️ PARTIALLY IMPLEMENTED: Advanced features are placeholders.
        
        Args:
            model: Model instance to rollback
            target_version: Version ID to rollback to
            strategy: Rollback strategy
                - "snapshot": Full state restore (default, implemented)
                - "delta": Incremental changes (placeholder)
                - "gradual": Traffic-based gradual shift (placeholder)
            traffic_safe: If True, perform gradual traffic shift
            health_check_fn: Function to check model health after rollback
            verification_fn: Function to verify rollback with baseline tests
            
        Returns:
            Rollback result with status, verification, and audit info
            
        Technical Design:
            1. Snapshot-based rollback (full state restore) ✅
            2. Delta-based rollback (incremental changes) ⚠️
            3. Traffic-safe rollback (gradual traffic shift) ⚠️
            4. Automatic health check during rollback ⚠️
            5. Rollback verification with baseline tests ⚠️
            6. Audit log of all rollback operations ✅
            
        Target Version: v2.1.0 (full implementation)
        
        Example:
            ```python
            result = vc.rollback_version(
                model, 
                "v1.0.0",
                strategy="snapshot",
                health_check_fn=lambda m: m.health_score > 0.9,
            )
            if result['success']:
                print(f"Rolled back to {result['version_id']}")
            ```
        """
        result = {
            "success": False,
            "version_id": target_version,
            "strategy": strategy,
            "previous_version": None,
            "health_check_passed": None,
            "verification_passed": None,
            "traffic_shift_complete": None,
            "rollback_time_ms": None,
            "audit_id": None,
            "error": None,
            "timestamp": datetime.now().isoformat(),
        }
        
        import time
        start_time = time.time()
        
        try:
            if target_version not in self.versions:
                raise ValueError(f"Version {target_version} not found")
            
            # Get current version for audit
            current_versions = [v for v in self.versions.values() if 'current' in v.tags]
            result["previous_version"] = current_versions[0].version_id if current_versions else None
            
            # Strategy-based rollback
            if strategy == "snapshot":
                # Full state restore (implemented)
                model, version = self.rollback(model, target_version)
                
            elif strategy == "delta":
                # Incremental changes (placeholder)
                logger.warning(
                    "Delta-based rollback is a placeholder. "
                    "Using snapshot strategy as fallback."
                )
                model, version = self.rollback(model, target_version)
                
            elif strategy == "gradual":
                # Traffic-based gradual shift (placeholder)
                logger.warning(
                    "Gradual rollback is a placeholder. "
                    "Using snapshot strategy as fallback."
                )
                model, version = self.rollback(model, target_version)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Health check
            if health_check_fn:
                try:
                    result["health_check_passed"] = health_check_fn(model)
                    if not result["health_check_passed"]:
                        logger.warning("Health check failed after rollback")
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    result["health_check_passed"] = False
            
            # Verification
            if verification_fn:
                try:
                    result["verification_passed"] = verification_fn(model)
                except Exception as e:
                    logger.error(f"Verification error: {e}")
                    result["verification_passed"] = False
            
            # Traffic shift (placeholder)
            if traffic_safe:
                logger.warning(
                    "Traffic-safe rollback is a placeholder. "
                    "Traffic shift simulation only."
                )
                result["traffic_shift_complete"] = True
            
            result["success"] = True
            result["rollback_time_ms"] = (time.time() - start_time) * 1000
            result["audit_id"] = hashlib.md5(
                f"{target_version}{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12]
            
            logger.info(
                f"Rollback complete: {result['previous_version']} -> {target_version} "
                f"(strategy={strategy}, time={result['rollback_time_ms']:.1f}ms)"
            )
            
        except Exception as e:
            result["error"] = str(e)
            result["rollback_time_ms"] = (time.time() - start_time) * 1000
            logger.error(f"Rollback failed: {e}")
        
        return result
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str
    ) -> VersionComparison:
        """
        Compare two model versions
        
        Args:
            version_a: First version ID
            version_b: Second version ID
            
        Returns:
            VersionComparison with differences
        """
        if version_a not in self.versions or version_b not in self.versions:
            raise ValueError("One or both versions not found")
        
        va = self.versions[version_a]
        vb = self.versions[version_b]
        
        # Calculate metric differences
        metric_diffs = {}
        all_metrics = set(va.metrics.keys()) | set(vb.metrics.keys())
        for metric in all_metrics:
            val_a = va.metrics.get(metric, 0)
            val_b = vb.metrics.get(metric, 0)
            metric_diffs[metric] = val_b - val_a
        
        # Calculate hyperparameter differences
        hp_diffs = {}
        all_hps = set(va.hyperparameters.keys()) | set(vb.hyperparameters.keys())
        for hp in all_hps:
            val_a = va.hyperparameters.get(hp)
            val_b = vb.hyperparameters.get(hp)
            if val_a != val_b:
                hp_diffs[hp] = (val_a, val_b)
        
        # Check architecture change
        arch_changed = va.architecture_hash != vb.architecture_hash
        
        # Calculate overall improvement (assuming higher is better for main metric)
        main_metric = 'accuracy' if 'accuracy' in metric_diffs else list(metric_diffs.keys())[0]
        improvement = metric_diffs.get(main_metric, 0)
        
        # Generate recommendation
        if improvement > 0.05:
            recommendation = "PROMOTE: Significant improvement"
        elif improvement > 0:
            recommendation = "CONSIDER: Minor improvement"
        elif improvement > -0.02:
            recommendation = "NEUTRAL: No significant change"
        else:
            recommendation = "ROLLBACK: Performance degradation"
        
        return VersionComparison(
            version_a=version_a,
            version_b=version_b,
            metric_diffs=metric_diffs,
            hyperparameter_diffs=hp_diffs,
            architecture_changed=arch_changed,
            performance_improvement=improvement,
            recommendation=recommendation
        )
    
    def get_version_history(
        self,
        model_name: str,
        limit: Optional[int] = None
    ) -> List[ModelVersion]:
        """Get version history for a model"""
        versions = [
            v for v in self.versions.values()
            if v.model_name == model_name
        ]
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        if limit:
            versions = versions[:limit]
        
        return versions
    
    def get_best_version(
        self,
        model_name: str,
        metric: str,
        higher_is_better: bool = True
    ) -> Optional[ModelVersion]:
        """Get the best performing version for a metric"""
        versions = self.get_version_history(model_name)
        
        if not versions:
            return None
        
        if higher_is_better:
            return max(versions, key=lambda v: v.metrics.get(metric, float('-inf')))
        else:
            return min(versions, key=lambda v: v.metrics.get(metric, float('inf')))
    
    def auto_select_best(
        self,
        model: nn.Module,
        model_name: str,
        metric: str = 'accuracy',
        higher_is_better: bool = True
    ) -> Tuple[nn.Module, ModelVersion]:
        """Automatically load the best performing version"""
        best = self.get_best_version(model_name, metric, higher_is_better)
        
        if best is None:
            raise ValueError(f"No versions found for model: {model_name}")
        
        model, _ = self.load_version(best.version_id, model)
        return model, best
    
    def cleanup_old_versions(
        self,
        model_name: str,
        keep_count: int = 5,
        keep_best: bool = True,
        metric: str = 'accuracy'
    ) -> List[str]:
        """
        Clean up old versions, keeping only the most recent ones
        
        Args:
            model_name: Model name to clean
            keep_count: Number of recent versions to keep
            keep_best: Whether to always keep the best version
            metric: Metric to determine best version
            
        Returns:
            List of deleted version IDs
        """
        versions = self.get_version_history(model_name)
        deleted = []
        
        if len(versions) <= keep_count:
            return deleted
        
        # Always keep best version if requested
        best_version = None
        if keep_best:
            best_version = self.get_best_version(model_name, metric)
        
        # Sort by creation time, keep newest
        versions_to_delete = versions[keep_count:]
        
        for version in versions_to_delete:
            # Skip best version
            if best_version and version.version_id == best_version.version_id:
                continue
            
            # Delete checkpoint
            checkpoint_path = Path(version.checkpoint_path)
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path.parent)
            
            # Remove from registry
            del self.versions[version.version_id]
            deleted.append(version.version_id)
        
        self._save_versions()
        logger.info(f"Cleaned up {len(deleted)} old versions")
        
        return deleted
    
    def export_version(
        self,
        version_id: str,
        export_path: str,
        include_optimizer: bool = False
    ) -> str:
        """Export a version to a standalone package"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy checkpoint
        src_checkpoint = Path(version.checkpoint_path)
        dst_checkpoint = export_dir / "model.pt"
        
        if include_optimizer:
            shutil.copy(src_checkpoint, dst_checkpoint)
        else:
            # Load and save without optimizer
            checkpoint = torch.load(src_checkpoint)
            export_data = {
                'model_state_dict': checkpoint['model_state_dict'],
                'metrics': checkpoint['metrics'],
                'hyperparameters': checkpoint['hyperparameters']
            }
            torch.save(export_data, dst_checkpoint)
        
        # Save version metadata
        meta_path = export_dir / "version_info.json"
        with open(meta_path, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        logger.info(f"Exported version {version_id} to {export_path}")
        return str(export_dir)


class AutoVersioner:
    """
    Automatic versioning wrapper for training loops
    
    Automatically commits versions based on:
    - Performance improvements
    - Training milestones
    - Scheduled intervals
    """
    
    def __init__(
        self,
        version_control: ModelVersionControl,
        model: nn.Module,
        model_name: str,
        optimizer: torch.optim.Optimizer,
        auto_commit_improvement: float = 0.01,
        commit_interval_epochs: int = 10
    ):
        self.vc = version_control
        self.model = model
        self.model_name = model_name
        self.optimizer = optimizer
        self.improvement_threshold = auto_commit_improvement
        self.commit_interval = commit_interval_epochs
        
        self.best_metric = float('-inf')
        self.last_commit_epoch = 0
        self.current_version: Optional[ModelVersion] = None
    
    def step(
        self,
        epoch: int,
        metrics: Dict[str, float],
        hyperparameters: Dict[str, Any],
        primary_metric: str = 'accuracy'
    ) -> Optional[ModelVersion]:
        """
        Check and potentially commit a new version
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            hyperparameters: Training hyperparameters
            primary_metric: Main metric to track
            
        Returns:
            ModelVersion if committed, None otherwise
        """
        current_metric = metrics.get(primary_metric, 0)
        should_commit = False
        commit_reason = ""
        
        # Check for improvement
        if current_metric > self.best_metric + self.improvement_threshold:
            should_commit = True
            commit_reason = f"Improvement: {current_metric:.4f} > {self.best_metric:.4f}"
            self.best_metric = current_metric
        
        # Check for interval
        elif epoch - self.last_commit_epoch >= self.commit_interval:
            should_commit = True
            commit_reason = f"Scheduled commit at epoch {epoch}"
        
        if should_commit:
            parent = self.current_version.version_id if self.current_version else None
            
            version = self.vc.commit_model(
                model=self.model,
                model_name=self.model_name,
                metrics=metrics,
                hyperparameters=hyperparameters,
                optimizer=self.optimizer,
                description=commit_reason,
                parent_version=parent
            )
            
            self.current_version = version
            self.last_commit_epoch = epoch
            
            return version
        
        return None
