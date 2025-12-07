#!/usr/bin/env python3
"""
Verification Script for Networked Learning System
网络化学习系统验证脚本

Verifies:
1. Module imports
2. System startup/shutdown
3. Data flow between components
4. Configuration validation
5. Integration health checks

Usage:
    python scripts/verify_networked_learning.py
"""

import asyncio
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("verification")


class VerificationResult:
    """Result of a verification check."""
    
    def __init__(self, name: str):
        self.name = name
        self.passed = False
        self.error = None
        self.duration_ms = 0
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        msg = f"{status} {self.name}"
        if self.error:
            msg += f" - {self.error}"
        if self.duration_ms > 0:
            msg += f" ({self.duration_ms:.0f}ms)"
        return msg


async def verify_module_imports() -> VerificationResult:
    """Verify all modules can be imported."""
    result = VerificationResult("Module Imports")
    
    try:
        import time
        start = time.time()
        
        # Auto Network Learning
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
            DataSource,
            LearningData,
            QualityAssessor,
        )
        
        # Data Cleansing Pipeline
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
            CleansingStage,
        )
        
        # Infinite Learning Manager
        from ai_core.distributed_vc.infinite_learning_manager import (
            InfiniteLearningManager,
            MemoryConfig,
            StorageTier,
        )
        
        # Data Lifecycle Manager
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
            DataState,
        )
        
        # Technology Elimination
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except ImportError as e:
        result.error = str(e)
    
    return result


async def verify_learning_system_lifecycle() -> VerificationResult:
    """Verify V1/V3 learning system starts and stops correctly."""
    result = VerificationResult("Learning System Lifecycle")
    
    try:
        import time
        start = time.time()
        
        from ai_core.distributed_vc.auto_network_learning import (
            V1V3AutoLearningSystem,
            NetworkLearningConfig,
        )
        
        config = NetworkLearningConfig(
            v1_learning_interval_minutes=60,
            v2_push_enabled=False,
        )
        
        system = V1V3AutoLearningSystem("v1", config)
        
        # Start
        await system.start()
        assert system._running, "System should be running"
        assert len(system.connectors) > 0, "Should have connectors"
        
        # Get stats
        stats = system.get_stats()
        assert "version" in stats
        assert stats["version"] == "v1"
        
        # Stop
        await system.stop()
        assert not system._running, "System should be stopped"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def verify_cleansing_pipeline() -> VerificationResult:
    """Verify data cleansing pipeline works."""
    result = VerificationResult("Cleansing Pipeline")
    
    try:
        import time
        start = time.time()
        
        from ai_core.distributed_vc.data_cleansing_pipeline import (
            DataCleansingPipeline,
            CleansingConfig,
        )
        
        config = CleansingConfig(
            min_content_length=10,
            min_final_quality=0.3,
            v2_push_enabled=False,
        )
        
        pipeline = DataCleansingPipeline(config)
        
        # Create mock item
        class MockItem:
            data_id = "verify_test"
            title = "Verification Test Item"
            content = "This is test content for verification. " * 10
            source = "github"
            quality_score = 0.7
            is_cleaned = False
            is_validated = False
            metadata = {}
        
        # Process
        item_result = await pipeline.process_item(MockItem())
        
        assert item_result.data_id == "verify_test"
        assert item_result.final_quality > 0
        
        # Check stats
        stats = pipeline.get_stats()
        assert stats["total_received"] == 1
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def verify_infinite_learning_manager() -> VerificationResult:
    """Verify infinite learning manager."""
    result = VerificationResult("Infinite Learning Manager")
    
    try:
        import time
        import tempfile
        start = time.time()
        
        from ai_core.distributed_vc.infinite_learning_manager import (
            InfiniteLearningManager,
            MemoryConfig,
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MemoryConfig(
                max_memory_mb=100,
                checkpoint_interval_minutes=60,
            )
            
            manager = InfiniteLearningManager(tmp_dir, config)
            
            # Add data
            item_id = await manager.add_learning_data(
                {"content": "test data for verification"},
                source="verification",
            )
            
            assert item_id is not None
            assert manager.total_learned == 1
            
            # Get stats
            stats = manager.get_stats()
            assert stats["hot_data_count"] == 1
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def verify_lifecycle_manager() -> VerificationResult:
    """Verify data lifecycle manager."""
    result = VerificationResult("Data Lifecycle Manager")
    
    try:
        import time
        import tempfile
        start = time.time()
        
        from ai_core.distributed_vc.data_lifecycle_manager import (
            DataLifecycleManager,
            DataLifecycleConfig,
            DataState,
        )
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = DataLifecycleConfig(cleanup_interval_hours=24)
            manager = DataLifecycleManager(tmp_dir, config)
            
            # Register data
            entry = manager.register_data("verify_1", source="test")
            assert entry.state == DataState.ACTIVE
            
            # Mark obsolete
            manager.mark_obsolete("verify_1", "Test obsolete")
            entry = manager.get_entry("verify_1")
            assert entry.state == DataState.OBSOLETE
            
            # Get stats
            stats = manager.get_stats()
            assert stats["total_registered"] == 1
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def verify_tech_elimination() -> VerificationResult:
    """Verify technology elimination manager."""
    result = VerificationResult("Technology Elimination")
    
    try:
        import time
        from unittest.mock import AsyncMock, MagicMock
        start = time.time()
        
        from ai_core.three_version_cycle.spiral_evolution_manager import (
            TechEliminationManager,
            TechEliminationConfig,
        )
        
        config = TechEliminationConfig(
            min_accuracy_threshold=0.75,
            consecutive_failures_to_eliminate=3,
        )
        
        # Mock version manager
        version_manager = MagicMock()
        version_manager.get_technology = AsyncMock(return_value={
            "name": "test_tech",
            "metrics": {"accuracy": 0.70, "error_rate": 0.10}
        })
        
        manager = TechEliminationManager(version_manager, config=config)
        
        # Evaluate
        eval_result = await manager.evaluate_technology("test_tech")
        
        assert eval_result["found"] is True
        assert eval_result["should_eliminate"] is True
        
        # Check status
        status = manager.get_elimination_status()
        assert status["statistics"]["total_evaluations"] == 1
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def verify_quality_assessment() -> VerificationResult:
    """Verify quality assessment logic."""
    result = VerificationResult("Quality Assessment")
    
    try:
        import time
        start = time.time()
        
        from ai_core.distributed_vc.auto_network_learning import (
            QualityAssessor,
            NetworkLearningConfig,
            LearningData,
            DataSource,
        )
        
        config = NetworkLearningConfig()
        assessor = QualityAssessor(config)
        
        # High quality item
        high_item = LearningData(
            data_id="high_quality",
            source=DataSource.GITHUB_TRENDING,
            title="Comprehensive Framework for Machine Learning",
            content=(
                "This framework provides comprehensive tools for ML development. "
                "Includes data processing, model training, and deployment features. " * 20
            ),
            url="https://github.com/test/framework",
            fetched_at=datetime.now(timezone.utc),
            tags=["python", "ml"],
        )
        
        high_score = assessor.assess(high_item)
        
        # Low quality item
        low_item = LearningData(
            data_id="low_quality",
            source=DataSource.DEV_TO,
            title="Hi",
            content="Short",
            url="",
            fetched_at=datetime.now(timezone.utc),
        )
        
        low_score = assessor.assess(low_item)
        
        assert 0.0 <= high_score <= 1.0
        assert 0.0 <= low_score <= 1.0
        assert high_score > low_score, "High quality should score higher"
        
        result.duration_ms = (time.time() - start) * 1000
        result.passed = True
        
    except Exception as e:
        result.error = str(e)
    
    return result


async def run_all_verifications():
    """Run all verification checks."""
    print("=" * 60)
    print("NETWORKED LEARNING SYSTEM VERIFICATION")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print("-" * 60)
    
    verifications = [
        verify_module_imports,
        verify_learning_system_lifecycle,
        verify_cleansing_pipeline,
        verify_infinite_learning_manager,
        verify_lifecycle_manager,
        verify_tech_elimination,
        verify_quality_assessment,
    ]
    
    results = []
    
    for verify_func in verifications:
        try:
            result = await verify_func()
        except Exception as e:
            result = VerificationResult(verify_func.__name__)
            result.error = str(e)
        
        results.append(result)
        print(result)
    
    print("-" * 60)
    
    # Summary
    passed = sum(1 for r in results if r.passed)
    total = len(results)
    
    print(f"\nRESULTS: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ ALL VERIFICATIONS PASSED")
        return 0
    else:
        print("\n❌ SOME VERIFICATIONS FAILED")
        failed = [r for r in results if not r.passed]
        print("\nFailed checks:")
        for r in failed:
            print(f"  - {r.name}: {r.error}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_verifications())
    sys.exit(exit_code)
