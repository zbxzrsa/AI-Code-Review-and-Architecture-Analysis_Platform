"""
Model Registry - Centralized Model Management
Supports model registration, staging, and deployment tracking
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum
import logging
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ModelStage(Enum):
    """Model lifecycle stages"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"
    QUARANTINE = "quarantine"


@dataclass
class RegisteredModel:
    """Registered model metadata"""
    model_id: str
    name: str
    version: str
    stage: ModelStage
    created_at: str
    updated_at: str
    description: str
    tags: List[str]
    metrics: Dict[str, float]
    artifact_path: str
    signature: Dict[str, Any]  # Input/output schema
    requirements: List[str]  # Dependencies
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['stage'] = self.stage.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'RegisteredModel':
        data['stage'] = ModelStage(data['stage'])
        return cls(**data)


@dataclass
class ModelTransition:
    """Model stage transition record"""
    model_id: str
    from_stage: ModelStage
    to_stage: ModelStage
    timestamp: str
    user: str
    reason: str
    approved: bool


class ModelRegistry:
    """
    Centralized Model Registry
    
    Features:
    - Model registration and versioning
    - Stage management (dev -> staging -> production)
    - Model comparison and selection
    - Deployment tracking
    - A/B testing support
    """
    
    def __init__(self, registry_path: str):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        
        self.models_file = self.registry_path / "models.json"
        self.transitions_file = self.registry_path / "transitions.json"
        
        self.models: Dict[str, RegisteredModel] = {}
        self.transitions: List[ModelTransition] = []
        
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk"""
        if self.models_file.exists():
            with open(self.models_file, 'r') as f:
                data = json.load(f)
                self.models = {
                    k: RegisteredModel.from_dict(v)
                    for k, v in data.items()
                }
        
        if self.transitions_file.exists():
            with open(self.transitions_file, 'r') as f:
                transitions_data = json.load(f)
                self.transitions = [
                    ModelTransition(
                        **{**t, 'from_stage': ModelStage(t['from_stage']),
                           'to_stage': ModelStage(t['to_stage'])}
                    )
                    for t in transitions_data
                ]
    
    def _save_registry(self) -> None:
        """Save registry to disk"""
        with open(self.models_file, 'w') as f:
            json.dump(
                {k: v.to_dict() for k, v in self.models.items()},
                f,
                indent=2
            )
        
        with open(self.transitions_file, 'w') as f:
            transitions_data = [
                {**asdict(t), 'from_stage': t.from_stage.value,
                 'to_stage': t.to_stage.value}
                for t in self.transitions
            ]
            json.dump(transitions_data, f, indent=2)
    
    def _generate_model_id(self, name: str, version: str) -> str:
        """Generate unique model ID"""
        content = f"{name}:{version}:{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def register_model(
        self,
        name: str,
        version: str,
        model: nn.Module,
        metrics: Dict[str, float],
        description: str = "",
        tags: Optional[List[str]] = None,
        signature: Optional[Dict[str, Any]] = None,
        requirements: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> RegisteredModel:
        """
        Register a new model in the registry
        
        Args:
            name: Model name
            version: Version string
            model: PyTorch model
            metrics: Performance metrics
            description: Model description
            tags: Optional tags
            signature: Input/output schema
            requirements: Dependencies
            metadata: Additional metadata
            
        Returns:
            RegisteredModel instance
        """
        model_id = self._generate_model_id(name, version)
        
        # Create artifact directory
        artifact_dir = self.registry_path / "artifacts" / model_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        artifact_path = artifact_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'architecture': str(model)
        }, artifact_path)
        
        # Create registered model
        now = datetime.now().isoformat()
        registered = RegisteredModel(
            model_id=model_id,
            name=name,
            version=version,
            stage=ModelStage.DEVELOPMENT,
            created_at=now,
            updated_at=now,
            description=description,
            tags=tags or [],
            metrics=metrics,
            artifact_path=str(artifact_path),
            signature=signature or {},
            requirements=requirements or [],
            metadata=metadata or {}
        )
        
        self.models[model_id] = registered
        self._save_registry()
        
        logger.info(f"Registered model: {name}:{version} (ID: {model_id})")
        return registered
    
    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Get model by ID"""
        return self.models.get(model_id)
    
    def get_model_by_name(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[ModelStage] = None
    ) -> List[RegisteredModel]:
        """Get models by name, optionally filtered by version and stage"""
        models = [m for m in self.models.values() if m.name == name]
        
        if version:
            models = [m for m in models if m.version == version]
        if stage:
            models = [m for m in models if m.stage == stage]
        
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def get_production_model(self, name: str) -> Optional[RegisteredModel]:
        """Get the current production model for a given name"""
        models = self.get_model_by_name(name, stage=ModelStage.PRODUCTION)
        return models[0] if models else None
    
    def transition_model(
        self,
        model_id: str,
        to_stage: ModelStage,
        user: str,
        reason: str,
        approved: bool = True
    ) -> ModelTransition:
        """
        Transition a model to a new stage
        
        Args:
            model_id: Model ID
            to_stage: Target stage
            user: User performing transition
            reason: Reason for transition
            approved: Whether transition is approved
            
        Returns:
            ModelTransition record
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        model = self.models[model_id]
        from_stage = model.stage
        
        # Validate transition
        valid_transitions = {
            ModelStage.DEVELOPMENT: [ModelStage.STAGING, ModelStage.ARCHIVED],
            ModelStage.STAGING: [ModelStage.PRODUCTION, ModelStage.DEVELOPMENT, 
                                 ModelStage.ARCHIVED],
            ModelStage.PRODUCTION: [ModelStage.STAGING, ModelStage.ARCHIVED,
                                    ModelStage.QUARANTINE],
            ModelStage.ARCHIVED: [ModelStage.DEVELOPMENT],
            ModelStage.QUARANTINE: [ModelStage.DEVELOPMENT, ModelStage.ARCHIVED]
        }
        
        if to_stage not in valid_transitions.get(from_stage, []):
            raise ValueError(
                f"Invalid transition: {from_stage.value} -> {to_stage.value}"
            )
        
        # Create transition record
        transition = ModelTransition(
            model_id=model_id,
            from_stage=from_stage,
            to_stage=to_stage,
            timestamp=datetime.now().isoformat(),
            user=user,
            reason=reason,
            approved=approved
        )
        
        if approved:
            # If promoting to production, demote current production model
            if to_stage == ModelStage.PRODUCTION:
                current_prod = self.get_production_model(model.name)
                if current_prod and current_prod.model_id != model_id:
                    current_prod.stage = ModelStage.STAGING
                    current_prod.updated_at = datetime.now().isoformat()
            
            # Update model stage
            model.stage = to_stage
            model.updated_at = datetime.now().isoformat()
        
        self.transitions.append(transition)
        self._save_registry()
        
        logger.info(
            f"Model {model_id} transition: {from_stage.value} -> {to_stage.value}"
        )
        return transition
    
    def promote_to_staging(
        self,
        model_id: str,
        user: str,
        reason: str = "Ready for staging tests"
    ) -> ModelTransition:
        """Convenience method to promote model to staging"""
        return self.transition_model(
            model_id, ModelStage.STAGING, user, reason
        )
    
    def promote_to_production(
        self,
        model_id: str,
        user: str,
        reason: str = "Passed staging tests"
    ) -> ModelTransition:
        """Convenience method to promote model to production"""
        return self.transition_model(
            model_id, ModelStage.PRODUCTION, user, reason
        )
    
    def quarantine_model(
        self,
        model_id: str,
        user: str,
        reason: str
    ) -> ModelTransition:
        """Quarantine a problematic model"""
        return self.transition_model(
            model_id, ModelStage.QUARANTINE, user, reason
        )
    
    def compare_models(
        self,
        model_id_a: str,
        model_id_b: str
    ) -> Dict[str, Any]:
        """
        Compare two models
        
        Returns:
            Comparison results including metric differences
        """
        model_a = self.models.get(model_id_a)
        model_b = self.models.get(model_id_b)
        
        if not model_a or not model_b:
            raise ValueError("One or both models not found")
        
        # Calculate metric differences
        all_metrics = set(model_a.metrics.keys()) | set(model_b.metrics.keys())
        metric_comparison = {}
        
        for metric in all_metrics:
            val_a = model_a.metrics.get(metric, 0)
            val_b = model_b.metrics.get(metric, 0)
            metric_comparison[metric] = {
                'model_a': val_a,
                'model_b': val_b,
                'difference': val_b - val_a,
                'percent_change': ((val_b - val_a) / val_a * 100) if val_a != 0 else 0
            }
        
        return {
            'model_a': model_a.to_dict(),
            'model_b': model_b.to_dict(),
            'metric_comparison': metric_comparison,
            'winner': model_id_b if sum(
                m['difference'] for m in metric_comparison.values()
            ) > 0 else model_id_a
        }
    
    def load_model(
        self,
        model_id: str,
        model_class: type
    ) -> nn.Module:
        """
        Load a registered model
        
        Args:
            model_id: Model ID to load
            model_class: Model class to instantiate
            
        Returns:
            Loaded model instance
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        registered = self.models[model_id]
        checkpoint = torch.load(registered.artifact_path)
        
        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model
    
    def get_model_history(self, name: str) -> List[Dict[str, Any]]:
        """Get complete history of a model including all transitions"""
        models = self.get_model_by_name(name)
        history = []
        
        for model in models:
            model_transitions = [
                t for t in self.transitions
                if t.model_id == model.model_id
            ]
            
            history.append({
                'model': model.to_dict(),
                'transitions': [
                    {
                        'from': t.from_stage.value,
                        'to': t.to_stage.value,
                        'timestamp': t.timestamp,
                        'user': t.user,
                        'reason': t.reason
                    }
                    for t in model_transitions
                ]
            })
        
        return history
    
    def search_models(
        self,
        query: Optional[str] = None,
        tags: Optional[List[str]] = None,
        stage: Optional[ModelStage] = None,
        min_metric: Optional[Dict[str, float]] = None
    ) -> List[RegisteredModel]:
        """
        Search models with various filters
        
        Args:
            query: Search in name and description
            tags: Filter by tags
            stage: Filter by stage
            min_metric: Minimum metric values
            
        Returns:
            List of matching models
        """
        results = list(self.models.values())
        
        if query:
            query_lower = query.lower()
            results = [
                m for m in results
                if query_lower in m.name.lower() or 
                   query_lower in m.description.lower()
            ]
        
        if tags:
            results = [
                m for m in results
                if any(t in m.tags for t in tags)
            ]
        
        if stage:
            results = [m for m in results if m.stage == stage]
        
        if min_metric:
            for metric, min_val in min_metric.items():
                results = [
                    m for m in results
                    if m.metrics.get(metric, 0) >= min_val
                ]
        
        return results
    
    def delete_model(self, model_id: str, force: bool = False) -> bool:
        """
        Delete a model from registry
        
        Args:
            model_id: Model ID to delete
            force: Force deletion even if in production
            
        Returns:
            True if deleted successfully
        """
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        if model.stage == ModelStage.PRODUCTION and not force:
            raise ValueError("Cannot delete production model without force=True")
        
        # Remove artifact
        artifact_path = Path(model.artifact_path)
        if artifact_path.exists():
            artifact_path.unlink()
            artifact_path.parent.rmdir()
        
        # Remove from registry
        del self.models[model_id]
        self._save_registry()
        
        logger.info(f"Deleted model: {model_id}")
        return True
