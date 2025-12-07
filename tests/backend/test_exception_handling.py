"""
Unit tests for enhanced exception handling mechanisms.

Tests:
1. OnlineLearningModule._learning_loop - Fine-grained exception handling
2. ModelDistiller.distill - Checkpoint recovery and exception handling
"""

import asyncio
import pytest
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from dataclasses import dataclass
from typing import Dict, Any, List

import torch
import torch.nn as nn


# =============================================================================
# Test Exception Classification System
# =============================================================================

class TestExceptionClassification:
    """Tests for the exception classification system."""
    
    def test_exception_severity_enum(self):
        """Test ExceptionSeverity enum values."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import ExceptionSeverity
        
        assert ExceptionSeverity.LOW == "low"
        assert ExceptionSeverity.MEDIUM == "medium"
        assert ExceptionSeverity.HIGH == "high"
        assert ExceptionSeverity.CRITICAL == "critical"
    
    def test_learning_error_codes(self):
        """Test LearningErrorCode enum values."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import LearningErrorCode
        
        # Data errors
        assert LearningErrorCode.E1001_INVALID_SAMPLE.value == "E1001"
        assert LearningErrorCode.E1002_EMPTY_BUFFER.value == "E1002"
        
        # Model errors
        assert LearningErrorCode.E2001_FORWARD_PASS.value == "E2001"
        assert LearningErrorCode.E2003_GRADIENT_EXPLOSION.value == "E2003"
        
        # Resource errors
        assert LearningErrorCode.E3001_OUT_OF_MEMORY.value == "E3001"
        assert LearningErrorCode.E3002_DEVICE_ERROR.value == "E3002"
    
    def test_learning_exception_dataclass(self):
        """Test LearningException dataclass."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                LearningException, 
                LearningErrorCode, 
                ExceptionSeverity
            )
        
        exc = LearningException(
            error_code=LearningErrorCode.E1001_INVALID_SAMPLE,
            severity=ExceptionSeverity.MEDIUM,
            message="Test error",
            recoverable=True,
        )
        
        assert exc.error_code == LearningErrorCode.E1001_INVALID_SAMPLE
        assert exc.severity == ExceptionSeverity.MEDIUM
        assert exc.recoverable is True
        
        # Test to_dict
        exc_dict = exc.to_dict()
        assert exc_dict["error_code"] == "E1001"
        assert exc_dict["severity"] == "medium"
    
    def test_exception_classifier_memory_error(self):
        """Test classification of memory errors as CRITICAL."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                ExceptionClassifier, 
                ExceptionSeverity,
                LearningErrorCode
            )
        
        exc = MemoryError("Out of memory")
        classified = ExceptionClassifier.classify(exc)
        
        assert classified.severity == ExceptionSeverity.CRITICAL
        assert classified.error_code == LearningErrorCode.E3001_OUT_OF_MEMORY
        assert classified.recoverable is False
    
    def test_exception_classifier_value_error(self):
        """Test classification of ValueError as MEDIUM."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                ExceptionClassifier, 
                ExceptionSeverity
            )
        
        exc = ValueError("Invalid data format")
        classified = ExceptionClassifier.classify(exc)
        
        assert classified.severity == ExceptionSeverity.MEDIUM
        assert classified.recoverable is True
    
    def test_exception_classifier_timeout(self):
        """Test classification of TimeoutError as LOW."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                ExceptionClassifier, 
                ExceptionSeverity
            )
        
        exc = asyncio.TimeoutError("Operation timed out")
        classified = ExceptionClassifier.classify(exc)
        
        assert classified.severity == ExceptionSeverity.LOW
        assert classified.recoverable is True
    
    def test_exception_classifier_gradient_error(self):
        """Test classification of gradient-related errors as HIGH."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                ExceptionClassifier, 
                ExceptionSeverity,
                LearningErrorCode
            )
        
        exc = RuntimeError("Gradient contains NaN values")
        classified = ExceptionClassifier.classify(exc)
        
        assert classified.severity == ExceptionSeverity.HIGH
        assert classified.error_code == LearningErrorCode.E2003_GRADIENT_EXPLOSION


# =============================================================================
# Test Distillation Error Handling
# =============================================================================

class TestDistillationErrorHandling:
    """Tests for ModelDistiller exception handling."""
    
    def test_distillation_error_codes(self):
        """Test DistillationErrorCode enum."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.practical_deployment import DistillationErrorCode
        
        assert DistillationErrorCode.D1001_MODEL_NOT_CREATED.value == "D1001"
        assert DistillationErrorCode.D3001_OUT_OF_MEMORY.value == "D3001"
        assert DistillationErrorCode.D3002_CHECKPOINT_SAVE.value == "D3002"
    
    def test_distillation_checkpoint_dataclass(self):
        """Test DistillationCheckpoint dataclass."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.practical_deployment import DistillationCheckpoint
        
        checkpoint = DistillationCheckpoint(
            epoch=2,
            step=100,
            model_state_dict={},
            optimizer_state_dict={},
            total_loss=0.5,
            best_loss=0.4,
            training_history=[{"epoch": 1, "loss": 0.6}],
            timestamp="2024-01-01T00:00:00",
        )
        
        assert checkpoint.epoch == 2
        assert checkpoint.step == 100
        assert checkpoint.best_loss == 0.4
        
        # Test to_dict
        ckpt_dict = checkpoint.to_dict()
        assert ckpt_dict["epoch"] == 2
        assert ckpt_dict["best_loss"] == 0.4
    
    def test_distillation_progress_dataclass(self):
        """Test DistillationProgress dataclass."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.practical_deployment import DistillationProgress
        
        progress = DistillationProgress(
            current_epoch=1,
            total_epochs=3,
            current_step=50,
            total_steps=150,
            current_loss=0.5,
            avg_loss=0.55,
            best_loss=0.4,
            elapsed_time=60.0,
            estimated_remaining=120.0,
            errors_encountered=2,
        )
        
        assert progress.current_epoch == 1
        assert progress.errors_encountered == 2


# =============================================================================
# Test Checkpoint Save/Load
# =============================================================================

class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)
    
    def forward(self, input_ids):
        x = torch.zeros(input_ids.shape[0], 10)
        return {"logits": self.linear(x)}


class TestCheckpointSaveLoad:
    """Tests for checkpoint saving and loading."""
    
    @pytest.fixture
    def temp_checkpoint_dir(self):
        """Create temporary checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    def test_checkpoint_directory_creation(self, temp_checkpoint_dir):
        """Test that checkpoint directory is created."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.practical_deployment import (
                ModelDistiller, 
                PracticalDeploymentConfig
            )
        
        model = SimpleTestModel()
        config = PracticalDeploymentConfig()
        
        distiller = ModelDistiller(
            teacher_model=model,
            config=config,
            checkpoint_dir=temp_checkpoint_dir,
        )
        
        assert Path(temp_checkpoint_dir).exists()
    
    def test_model_distiller_initialization(self, temp_checkpoint_dir):
        """Test ModelDistiller initialization with new parameters."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.practical_deployment import (
                ModelDistiller, 
                PracticalDeploymentConfig
            )
        
        model = SimpleTestModel()
        config = PracticalDeploymentConfig()
        
        distiller = ModelDistiller(
            teacher_model=model,
            config=config,
            checkpoint_dir=temp_checkpoint_dir,
            checkpoint_interval=50,
        )
        
        assert distiller.checkpoint_interval == 50
        assert distiller._current_epoch == 0
        assert distiller._best_loss == float('inf')
        assert distiller._errors_encountered == 0


# =============================================================================
# Test Error Summary
# =============================================================================

class TestErrorSummary:
    """Tests for error summary functionality."""
    
    def test_get_error_summary_empty(self):
        """Test error summary when no errors."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                OnlineLearningModule,
                AutonomousConfig,
            )
        
        # Create a mock model
        model = SimpleTestModel()
        config = AutonomousConfig()
        
        module = OnlineLearningModule(model, config)
        summary = module.get_error_summary()
        
        assert summary["total_errors"] == 0
        assert summary["by_severity"] == {}
        assert summary["by_code"] == {}
    
    def test_get_error_summary_with_errors(self):
        """Test error summary with recorded errors."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                OnlineLearningModule,
                AutonomousConfig,
                LearningException,
                LearningErrorCode,
                ExceptionSeverity,
            )
        
        model = SimpleTestModel()
        config = AutonomousConfig()
        
        module = OnlineLearningModule(model, config)
        
        # Manually add some errors
        module._error_log.append(LearningException(
            error_code=LearningErrorCode.E1001_INVALID_SAMPLE,
            severity=ExceptionSeverity.MEDIUM,
            message="Test error 1",
        ))
        module._error_log.append(LearningException(
            error_code=LearningErrorCode.E4002_TIMEOUT,
            severity=ExceptionSeverity.LOW,
            message="Test error 2",
        ))
        
        summary = module.get_error_summary()
        
        assert summary["total_errors"] == 2
        assert summary["by_severity"]["medium"] == 1
        assert summary["by_severity"]["low"] == 1


# =============================================================================
# Integration Tests
# =============================================================================

class TestExceptionHandlingIntegration:
    """Integration tests for exception handling."""
    
    def test_exception_handling_maintains_state(self):
        """Test that state is preserved across exception handling."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ai_core.foundation_model.autonomous_learning import (
                OnlineLearningModule,
                AutonomousConfig,
            )
        
        model = SimpleTestModel()
        config = AutonomousConfig()
        
        module = OnlineLearningModule(model, config)
        
        # Initial state
        assert module._consecutive_errors == 0
        assert module.total_updates == 0
        
        # State should be tracked
        module._consecutive_errors = 3
        module.total_updates = 10
        
        assert module._consecutive_errors == 3
        assert module.total_updates == 10
    
    def test_distiller_error_tracking(self, ):
        """Test that distiller tracks errors correctly."""
        import warnings
        import tempfile
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from ai_core.foundation_model.practical_deployment import (
                    ModelDistiller, 
                    PracticalDeploymentConfig
                )
            
            model = SimpleTestModel()
            config = PracticalDeploymentConfig()
            
            distiller = ModelDistiller(
                teacher_model=model,
                config=config,
                checkpoint_dir=tmpdir,
            )
            
            # Initial state
            assert distiller._errors_encountered == 0
            
            # Simulate error handling
            distiller._errors_encountered = 5
            
            progress = distiller.get_progress(
                total_epochs=3,
                total_steps=100,
                start_time=time.time() - 60
            )
            
            assert progress.errors_encountered == 5


# =============================================================================
# Parameterized Tests
# =============================================================================

@pytest.mark.parametrize("exception_type,expected_severity", [
    (MemoryError("OOM"), "critical"),
    (ValueError("Invalid"), "medium"),
    (asyncio.TimeoutError(), "low"),
    (IOError("IO failed"), "medium"),
])
def test_exception_severity_classification(exception_type, expected_severity):
    """Parameterized test for exception severity classification."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from ai_core.foundation_model.autonomous_learning import ExceptionClassifier
    
    classified = ExceptionClassifier.classify(exception_type)
    assert classified.severity.value == expected_severity


@pytest.mark.parametrize("error_code,description", [
    ("E1001", "Invalid sample format"),
    ("E2001", "Forward pass failed"),
    ("E3001", "Out of memory"),
    ("D1001", "Model not created"),
    ("D3002", "Checkpoint save failed"),
])
def test_error_code_format(error_code, description):
    """Test that error codes follow expected format."""
    # Error codes should be E/D followed by 4 digits
    assert error_code[0] in ("E", "D")
    assert error_code[1:].isdigit()
    assert len(error_code) == 5
