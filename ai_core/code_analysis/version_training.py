"""
Version-Integrated Training System

Trains the code analysis and correction models across all three versions:
- V1: Experimental training with new patterns and techniques
- V2: Stable production models with validated patterns
- V3: Quarantine training for analyzing failed approaches

Training Modes:
- Incremental: Learn from new code samples
- Batch: Full retraining on accumulated data
- Feedback: Learn from user corrections and ratings
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import uuid

from .analysis_engine import (
    CodeAnalysisEngine,
    CodeIssue,
    ErrorType,
    Severity,
    Language,
    AnalysisResult,
    ProjectAnalysis,
)
from .correction_system import (
    IntelligentCorrectionSystem,
    CorrectionMode,
    CorrectionSuggestion,
    CorrectionResult,
    FeedbackType,
)
from .ml_pattern_recognition import (
    MLPatternRecognition,
    CodePattern,
    PatternMatch,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class TrainingMode(str, Enum):
    """Training mode types."""
    INCREMENTAL = "incremental"  # Learn from new samples
    BATCH = "batch"              # Full retraining
    FEEDBACK = "feedback"        # Learn from user corrections


class ModelVersion(str, Enum):
    """Model version (maps to system versions)."""
    V1_EXPERIMENTAL = "v1"  # Testing new patterns
    V2_PRODUCTION = "v2"    # Validated, stable
    V3_QUARANTINE = "v3"    # Failed/deprecated


@dataclass
class TrainingConfig:
    """Configuration for training."""
    mode: TrainingMode = TrainingMode.INCREMENTAL
    batch_size: int = 100
    learning_rate: float = 0.01
    validation_split: float = 0.2
    min_samples_for_pattern: int = 5
    promotion_threshold: float = 0.85  # Accuracy for V1→V2 promotion
    demotion_threshold: float = 0.6    # Accuracy below this → V3


@dataclass
class TrainingSample:
    """A single training sample."""
    sample_id: str
    code: str
    language: Language
    issues: List[Dict[str, Any]]
    corrections: List[Dict[str, Any]]
    user_feedback: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrainingMetrics:
    """Metrics from a training session."""
    session_id: str
    version: ModelVersion
    mode: TrainingMode
    
    # Data metrics
    samples_processed: int = 0
    patterns_learned: int = 0
    patterns_promoted: int = 0
    patterns_demoted: int = 0
    
    # Performance metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    
    # Time metrics
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version.value,
            "mode": self.mode.value,
            "samples_processed": self.samples_processed,
            "patterns_learned": self.patterns_learned,
            "accuracy": self.accuracy,
            "f1_score": self.f1_score,
            "duration_seconds": self.duration_seconds,
        }


@dataclass
class VersionModel:
    """Model for a specific version."""
    version: ModelVersion
    model_id: str
    patterns: Dict[str, CodePattern]
    accuracy: float = 0.0
    samples_trained: int = 0
    last_trained: Optional[datetime] = None
    status: str = "active"


# =============================================================================
# Version Training Engine
# =============================================================================

class VersionTrainingEngine:
    """
    Training engine for a specific version.
    
    Handles pattern learning, validation, and model updates.
    """
    
    def __init__(
        self,
        version: ModelVersion,
        ml_recognition: MLPatternRecognition,
        config: TrainingConfig,
    ):
        self.version = version
        self.ml = ml_recognition
        self.config = config
        
        # Training data
        self.samples: List[TrainingSample] = []
        self.validation_samples: List[TrainingSample] = []
        
        # Model state
        self.model = VersionModel(
            version=version,
            model_id=f"model-{version.value}-{uuid.uuid4().hex[:8]}",
            patterns={},
        )
        
        # Metrics history
        self.metrics_history: List[TrainingMetrics] = []
        
        logger.info(f"Version Training Engine initialized: {version.value}")
    
    async def add_sample(self, sample: TrainingSample) -> None:
        """Add a training sample."""
        # Split between training and validation
        import random
        if random.random() < self.config.validation_split:
            self.validation_samples.append(sample)
        else:
            self.samples.append(sample)
        
        # Incremental learning
        if self.config.mode == TrainingMode.INCREMENTAL:
            await self._learn_from_sample(sample)
    
    async def _learn_from_sample(self, sample: TrainingSample) -> None:
        """Learn from a single sample."""
        # Extract patterns from issues
        for issue in sample.issues:
            issue_code = issue.get("code_snippet", "")
            issue_type = issue.get("error_type", "unknown")
            
            if len(issue_code) < 10:
                continue
            
            # Add to similarity index
            self.ml.similarity_engine.add_code(
                code=issue_code,
                issue_type=issue_type,
                metadata={"language": sample.language.value},
            )
        
        # Learn from corrections
        for correction in sample.corrections:
            original = correction.get("original_code", "")
            fixed = correction.get("corrected_code", "")
            if original and fixed:
                self.ml.learn_from_fix(
                    buggy_code=original,
                    fixed_code=fixed,
                    issue_type=correction.get("issue_type", "unknown"),
                )
        
        self.model.samples_trained += 1
    
    async def train_batch(self) -> TrainingMetrics:
        """Perform batch training on accumulated samples."""
        metrics = TrainingMetrics(
            session_id=str(uuid.uuid4()),
            version=self.version,
            mode=TrainingMode.BATCH,
        )
        
        logger.info(f"Starting batch training for {self.version.value} with {len(self.samples)} samples")
        
        # Process all samples
        for sample in self.samples:
            await self._learn_from_sample(sample)
            metrics.samples_processed += 1
        
        # Validate on validation set
        if self.validation_samples:
            metrics.accuracy = await self._validate()
            metrics.precision = metrics.accuracy  # Simplified
            metrics.recall = metrics.accuracy
            if metrics.precision + metrics.recall > 0:
                metrics.f1_score = (
                    2 * metrics.precision * metrics.recall
                ) / (metrics.precision + metrics.recall)
        
        # Count learned patterns
        stats = self.ml.pattern_engine.get_statistics()
        if self.version == ModelVersion.V1_EXPERIMENTAL:
            metrics.patterns_learned = stats.get("v1_experimental", 0)
        elif self.version == ModelVersion.V2_PRODUCTION:
            metrics.patterns_learned = stats.get("v2_production", 0)
        
        metrics.completed_at = datetime.now(timezone.utc)
        metrics.duration_seconds = (
            metrics.completed_at - metrics.started_at
        ).total_seconds()
        
        self.metrics_history.append(metrics)
        self.model.last_trained = datetime.now(timezone.utc)
        self.model.accuracy = metrics.accuracy
        
        logger.info(
            f"Batch training complete: {metrics.samples_processed} samples, "
            f"accuracy={metrics.accuracy:.2%}"
        )
        
        return metrics
    
    async def _validate(self) -> float:
        """Validate model on validation set."""
        if not self.validation_samples:
            return 0.0
        
        correct = 0
        total = 0
        
        for sample in self.validation_samples:
            # Analyze the sample code
            analysis = await self.ml.analyze_code(sample.code)
            detected_patterns = set(p["name"] for p in analysis.get("patterns", []))
            
            # Compare with known issues
            expected_issues = set(
                i.get("rule_id", "") for i in sample.issues
            )
            
            if detected_patterns or expected_issues:
                # Simple accuracy: any overlap counts
                if detected_patterns & expected_issues:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status."""
        return {
            "version": self.version.value,
            "model_id": self.model.model_id,
            "samples_trained": self.model.samples_trained,
            "accuracy": self.model.accuracy,
            "last_trained": self.model.last_trained.isoformat() if self.model.last_trained else None,
            "status": self.model.status,
            "patterns_count": len(self.model.patterns),
        }


# =============================================================================
# Three-Version Training Coordinator
# =============================================================================

class ThreeVersionTrainingCoordinator:
    """
    Coordinates training across all three versions.
    
    Manages:
    - V1 experimental training
    - V2 production model stability
    - V3 quarantine analysis
    - Pattern promotion and demotion
    """
    
    def __init__(
        self,
        analysis_engine: Optional[CodeAnalysisEngine] = None,
        correction_system: Optional[IntelligentCorrectionSystem] = None,
        config: Optional[TrainingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        
        # Core components
        self.analysis_engine = analysis_engine or CodeAnalysisEngine()
        self.correction_system = correction_system or IntelligentCorrectionSystem()
        self.ml_recognition = MLPatternRecognition()
        
        # Version-specific trainers
        self.v1_trainer = VersionTrainingEngine(
            ModelVersion.V1_EXPERIMENTAL,
            self.ml_recognition,
            self.config,
        )
        self.v2_trainer = VersionTrainingEngine(
            ModelVersion.V2_PRODUCTION,
            self.ml_recognition,
            self.config,
        )
        self.v3_trainer = VersionTrainingEngine(
            ModelVersion.V3_QUARANTINE,
            self.ml_recognition,
            self.config,
        )
        
        # Training state
        self._running = False
        self._training_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._total_samples = 0
        self._promotions = 0
        self._demotions = 0
        
        logger.info("Three-Version Training Coordinator initialized")
    
    async def start(self) -> None:
        """Start the training coordinator."""
        if self._running:
            return
        
        self._running = True
        logger.info("Training Coordinator started")
        logger.info("  V1: Experimental patterns training")
        logger.info("  V2: Production model (stable)")
        logger.info("  V3: Quarantine analysis")
    
    async def stop(self) -> None:
        """Stop the training coordinator."""
        self._running = False
        if self._training_task:
            self._training_task.cancel()
        logger.info("Training Coordinator stopped")
    
    # =========================================================================
    # Training Methods
    # =========================================================================
    
    async def train_on_project(
        self,
        project_path: str,
        version: ModelVersion = ModelVersion.V1_EXPERIMENTAL,
    ) -> Dict[str, Any]:
        """
        Train on a project codebase.
        
        New training always starts in V1 (experimental).
        """
        logger.info(f"Training on project: {project_path}")
        
        # Analyze project
        analysis = await self.analysis_engine.analyze_directory(project_path)
        
        # Create training samples
        samples_created = 0
        for file_result in analysis.file_results:
            if file_result.issues:
                # Read code
                try:
                    with open(file_result.file_path, 'r', encoding='utf-8') as f:
                        code = f.read()
                except Exception:
                    continue
                
                sample = TrainingSample(
                    sample_id=str(uuid.uuid4()),
                    code=code,
                    language=file_result.language,
                    issues=[i.to_dict() for i in file_result.issues],
                    corrections=[],
                )
                
                # Add to appropriate trainer
                if version == ModelVersion.V1_EXPERIMENTAL:
                    await self.v1_trainer.add_sample(sample)
                elif version == ModelVersion.V2_PRODUCTION:
                    await self.v2_trainer.add_sample(sample)
                else:
                    await self.v3_trainer.add_sample(sample)
                
                samples_created += 1
                self._total_samples += 1
        
        return {
            "project": project_path,
            "files_analyzed": len(analysis.file_results),
            "samples_created": samples_created,
            "total_issues": analysis.total_issues,
            "version": version.value,
        }
    
    async def train_from_feedback(
        self,
        suggestion_id: str,
        feedback_type: FeedbackType,
        correct_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train from user feedback on a correction.
        
        Feedback training goes to V1 first.
        """
        # Get the original suggestion
        suggestion = self.correction_system._suggestions.get(suggestion_id)
        if not suggestion:
            return {"success": False, "error": "Suggestion not found"}
        
        # Create training sample from feedback
        sample = TrainingSample(
            sample_id=str(uuid.uuid4()),
            code=suggestion.original_code,
            language=Language.PYTHON,  # Infer from context
            issues=[suggestion.issue.to_dict()],
            corrections=[{
                "original_code": suggestion.original_code,
                "corrected_code": correct_code or suggestion.corrected_code,
                "issue_type": suggestion.issue.error_type.value,
            }],
            user_feedback={
                "type": feedback_type.value,
                "suggestion_id": suggestion_id,
            },
        )
        
        # Add to V1 trainer (experimental feedback learning)
        await self.v1_trainer.add_sample(sample)
        
        # Learn the fix pattern
        if correct_code:
            self.ml_recognition.learn_from_fix(
                buggy_code=suggestion.original_code,
                fixed_code=correct_code,
                issue_type=suggestion.issue.error_type.value,
            )
        
        self._total_samples += 1
        
        return {
            "success": True,
            "sample_id": sample.sample_id,
            "feedback_type": feedback_type.value,
        }
    
    async def run_batch_training(
        self,
        version: Optional[ModelVersion] = None,
    ) -> Dict[str, TrainingMetrics]:
        """
        Run batch training on all versions or a specific version.
        """
        results = {}
        
        trainers = []
        if version:
            if version == ModelVersion.V1_EXPERIMENTAL:
                trainers.append(("v1", self.v1_trainer))
            elif version == ModelVersion.V2_PRODUCTION:
                trainers.append(("v2", self.v2_trainer))
            else:
                trainers.append(("v3", self.v3_trainer))
        else:
            trainers = [
                ("v1", self.v1_trainer),
                ("v2", self.v2_trainer),
                ("v3", self.v3_trainer),
            ]
        
        for name, trainer in trainers:
            metrics = await trainer.train_batch()
            results[name] = metrics
        
        return results
    
    # =========================================================================
    # Pattern Lifecycle
    # =========================================================================
    
    async def evaluate_patterns(self) -> Dict[str, Any]:
        """
        Evaluate patterns and handle promotion/demotion.
        
        - High-performing V1 patterns → promote to V2
        - Low-performing V2 patterns → demote to V3
        - V3 patterns analyzed for failure reasons
        """
        promotions = []
        demotions = []
        
        pattern_engine = self.ml_recognition.pattern_engine
        
        for pattern_id, pattern in pattern_engine.patterns.items():
            if pattern.version == "v1":
                # Check for promotion
                if (pattern.precision >= self.config.promotion_threshold and
                    pattern.detection_count >= self.config.min_samples_for_pattern):
                    if pattern_engine.promote_pattern(pattern_id):
                        promotions.append(pattern_id)
                        self._promotions += 1
            
            elif pattern.version == "v2":
                # Check for demotion
                if (pattern.precision < self.config.demotion_threshold and
                    pattern.detection_count >= self.config.min_samples_for_pattern):
                    if pattern_engine.quarantine_pattern(pattern_id, "Low accuracy"):
                        demotions.append(pattern_id)
                        self._demotions += 1
        
        return {
            "patterns_promoted": promotions,
            "patterns_demoted": demotions,
            "total_promotions": self._promotions,
            "total_demotions": self._demotions,
        }
    
    # =========================================================================
    # API Methods
    # =========================================================================
    
    async def analyze_and_correct(
        self,
        code: str,
        language: Language,
        mode: CorrectionMode = CorrectionMode.BASIC,
    ) -> Dict[str, Any]:
        """
        Analyze code and generate corrections.
        
        Uses trained models for pattern detection.
        """
        # ML pattern detection
        ml_analysis = await self.ml_recognition.analyze_code(code)
        
        # Create a temporary file-like analysis result
        # In production, this would use the full analysis engine
        issues = []
        for pattern_data in ml_analysis.get("patterns", []):
            # Convert pattern to issue format
            location_data = pattern_data.get("location", {})
            issues.append(CodeIssue(
                issue_id=str(uuid.uuid4()),
                error_type=ErrorType.SECURITY,  # Pattern type would map to error type
                severity=Severity.MEDIUM,
                message=f"Pattern detected: {pattern_data.get('name', 'unknown')}",
                location=CodeLocation(
                    file_path="<inline>",
                    line_start=location_data.get("line", 1) if isinstance(location_data, dict) else 1,
                    line_end=location_data.get("line", 1) if isinstance(location_data, dict) else 1,
                ),
                code_snippet=pattern_data.get("match", "")[:100],
                rule_id=f"ML-{pattern_data.get('name', 'unknown')}",
                confidence=pattern_data.get("confidence", 0.8),
            ))
        
        # Generate corrections for each issue
        corrections = []
        for issue in issues:
            suggestion = await self.correction_system.suggest_correction(issue, code, mode)
            if suggestion:
                corrections.append(suggestion.to_dict())
        
        return {
            "analysis": ml_analysis,
            "issues_found": len(issues),
            "corrections_available": len(corrections),
            "corrections": corrections,
        }
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get comprehensive training status."""
        return {
            "running": self._running,
            "total_samples_trained": self._total_samples,
            "total_promotions": self._promotions,
            "total_demotions": self._demotions,
            "v1_status": self.v1_trainer.get_model_status(),
            "v2_status": self.v2_trainer.get_model_status(),
            "v3_status": self.v3_trainer.get_model_status(),
            "ml_statistics": self.ml_recognition.get_statistics(),
        }


# =============================================================================
# Factory Function
# =============================================================================

def create_training_system(
    config: Optional[TrainingConfig] = None,
) -> ThreeVersionTrainingCoordinator:
    """
    Create a fully configured training system.
    
    This is the recommended entry point.
    """
    return ThreeVersionTrainingCoordinator(config=config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Enums
    "TrainingMode",
    "ModelVersion",
    # Config
    "TrainingConfig",
    # Data classes
    "TrainingSample",
    "TrainingMetrics",
    "VersionModel",
    # Engines
    "VersionTrainingEngine",
    # Coordinator
    "ThreeVersionTrainingCoordinator",
    # Factory
    "create_training_system",
]
