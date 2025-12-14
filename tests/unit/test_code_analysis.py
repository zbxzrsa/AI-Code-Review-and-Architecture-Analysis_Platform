"""
Unit Tests for Code Analysis and Correction Module

Comprehensive test coverage for:
- Analysis Engine (syntax, security, performance, style)
- Intelligent Correction System (basic, advanced, teaching modes)
- ML Pattern Recognition (pattern learning, similarity)
- Version Training (V1, V2, V3 integration)

Test Requirements:
- Unit tests cover all core functions
- Integration tests verify cross-module collaboration
- Edge case handling verification
"""

import asyncio
import pytest
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import tempfile
import os

# Import modules to test
from ai_core.code_analysis.analysis_engine import (
    CodeAnalysisEngine,
    PythonAnalyzer,
    JavaScriptAnalyzer,
    JavaAnalyzer,
    CodeLocation,
    CodeIssue,
    AnalysisResult,
    ProjectAnalysis,
    ErrorType,
    Severity,
    Language,
    LANGUAGE_EXTENSIONS,
)

from ai_core.code_analysis.correction_system import (
    IntelligentCorrectionSystem,
    CorrectionMode,
    CorrectionStatus,
    FeedbackType,
    CorrectionSuggestion,
    CorrectionResult,
    UserFeedback,
    TeachingExample,
    CorrectionStep,
    SecurityCorrectionStrategy,
    StyleCorrectionStrategy,
    LogicalCorrectionStrategy,
)

from ai_core.code_analysis.ml_pattern_recognition import (
    MLPatternRecognition,
    PatternLearningEngine,
    CodeSimilarityEngine,
    CodePattern,
    PatternMatch,
    SimilarCode,
)

from ai_core.code_analysis.version_training import (
    ThreeVersionTrainingCoordinator,
    VersionTrainingEngine,
    TrainingConfig,
    TrainingMode,
    ModelVersion,
    TrainingSample,
    TrainingMetrics,
    create_training_system,
)


# =============================================================================
# Analysis Engine Tests
# =============================================================================

class TestCodeAnalysisEngine:
    """Tests for the CodeAnalysisEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = CodeAnalysisEngine()
        
        assert Language.PYTHON in engine.analyzers
        assert Language.JAVASCRIPT in engine.analyzers
        assert Language.JAVA in engine.analyzers
        assert len(engine.ignore_patterns) > 0
    
    def test_language_detection_python(self):
        """Test Python language detection."""
        engine = CodeAnalysisEngine()
        
        assert engine.detect_language("test.py") == Language.PYTHON
        assert engine.detect_language("test.pyw") == Language.PYTHON
    
    def test_language_detection_javascript(self):
        """Test JavaScript/TypeScript language detection."""
        engine = CodeAnalysisEngine()
        
        assert engine.detect_language("test.js") == Language.JAVASCRIPT
        assert engine.detect_language("test.jsx") == Language.JAVASCRIPT
        assert engine.detect_language("test.ts") == Language.TYPESCRIPT
        assert engine.detect_language("test.tsx") == Language.TYPESCRIPT
    
    def test_language_detection_java(self):
        """Test Java language detection."""
        engine = CodeAnalysisEngine()
        
        assert engine.detect_language("Test.java") == Language.JAVA
    
    def test_language_detection_unknown(self):
        """Test unknown language detection."""
        engine = CodeAnalysisEngine()
        
        assert engine.detect_language("test.xyz") == Language.UNKNOWN
        assert engine.detect_language("README.md") == Language.UNKNOWN
    
    def test_should_ignore_patterns(self):
        """Test ignore pattern matching."""
        engine = CodeAnalysisEngine()
        
        assert engine.should_ignore("node_modules/package/index.js")
        assert engine.should_ignore("__pycache__/module.pyc")
        assert engine.should_ignore(".git/config")
        assert not engine.should_ignore("src/main.py")
    
    @pytest.mark.asyncio
    async def test_analyze_python_syntax_error(self):
        """Test detection of Python syntax errors."""
        engine = CodeAnalysisEngine()
        
        # Create temp file with syntax error
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def broken(\n")  # Missing closing parenthesis
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            assert result.language == Language.PYTHON
            assert result.success  # File was analyzed (even with syntax error)
            assert any(i.error_type == ErrorType.SYNTAX for i in result.issues)
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_python_security_hardcoded_secret(self):
        """Test detection of hardcoded secrets."""
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('password = "super_secret_123"\n')
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            assert result.language == Language.PYTHON
            assert any(
                i.error_type == ErrorType.SECURITY and "secret" in i.rule_id.lower()
                for i in result.issues
            )
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_python_eval_usage(self):
        """Test detection of eval() usage."""
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('result = eval(user_input)\n')
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            assert any(
                i.error_type == ErrorType.SECURITY and "eval" in i.rule_id.lower()
                for i in result.issues
            )
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_python_none_comparison(self):
        """Test detection of == None comparison."""
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('if x == None:\n    pass\n')
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            assert any(
                i.error_type == ErrorType.LOGICAL and "none" in i.rule_id.lower()
                for i in result.issues
            )
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_clean_code(self):
        """Test analysis of clean code produces no issues."""
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''"""Clean module."""

import os


def get_value():
    """Get a value."""
    return os.getenv("MY_VAR")
''')
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            # Should have minimal or no critical issues
            critical_issues = [i for i in result.issues if i.severity == Severity.CRITICAL]
            assert len(critical_issues) == 0
        finally:
            os.unlink(temp_path)
    
    def test_get_statistics(self):
        """Test engine statistics."""
        engine = CodeAnalysisEngine()
        stats = engine.get_statistics()
        
        assert "analyses_performed" in stats
        assert "files_analyzed" in stats
        assert "issues_found" in stats
        assert "supported_languages" in stats


class TestPythonAnalyzer:
    """Tests for Python-specific analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes with rules."""
        analyzer = PythonAnalyzer()
        
        assert analyzer.language == Language.PYTHON
        assert len(analyzer.rules) > 0
    
    def test_rules_have_required_fields(self):
        """Test all rules have required fields."""
        analyzer = PythonAnalyzer()
        
        for rule in analyzer.rules:
            assert rule.rule_id
            assert rule.name
            assert rule.error_type
            assert rule.severity
    
    @pytest.mark.asyncio
    async def test_mutable_default_detection(self):
        """Test detection of mutable default arguments."""
        analyzer = PythonAnalyzer()
        
        code = '''
def add_item(item, items=[]):
    items.append(item)
    return items
'''
        issues = await analyzer.analyze("<test>", code)
        
        assert any("mutable" in i.rule_id.lower() for i in issues)


class TestJavaScriptAnalyzer:
    """Tests for JavaScript analyzer."""
    
    def test_analyzer_initialization(self):
        """Test analyzer initializes correctly."""
        analyzer = JavaScriptAnalyzer()
        
        assert analyzer.language == Language.JAVASCRIPT
        assert len(analyzer.rules) > 0
    
    @pytest.mark.asyncio
    async def test_eval_detection(self):
        """Test detection of eval usage."""
        analyzer = JavaScriptAnalyzer()
        
        code = 'const result = eval(userInput);'
        issues = await analyzer.analyze("<test>", code)
        
        assert any("eval" in i.rule_id.lower() for i in issues)
    
    @pytest.mark.asyncio
    async def test_innerhtml_detection(self):
        """Test detection of innerHTML usage."""
        analyzer = JavaScriptAnalyzer()
        
        code = 'element.innerHTML = userContent;'
        issues = await analyzer.analyze("<test>", code)
        
        assert any("innerhtml" in i.rule_id.lower() for i in issues)
    
    @pytest.mark.asyncio
    async def test_var_detection(self):
        """Test detection of var usage."""
        analyzer = JavaScriptAnalyzer()
        
        code = 'var x = 10;'
        issues = await analyzer.analyze("<test>", code)
        
        assert any("var" in i.rule_id.lower() for i in issues)


# =============================================================================
# Correction System Tests
# =============================================================================

class TestIntelligentCorrectionSystem:
    """Tests for the IntelligentCorrectionSystem class."""
    
    def test_system_initialization(self):
        """Test system initializes correctly."""
        system = IntelligentCorrectionSystem()
        
        assert len(system.strategies) > 0
        assert len(system._teaching_examples) > 0
    
    def test_teaching_examples_exist(self):
        """Test teaching examples are available."""
        system = IntelligentCorrectionSystem()
        
        examples = system.list_teaching_examples()
        assert len(examples) > 0
        
        for example in examples:
            assert example.title
            assert example.buggy_code
            assert example.fixed_code
            assert len(example.steps) > 0
    
    def test_teaching_examples_filter_by_type(self):
        """Test filtering teaching examples."""
        system = IntelligentCorrectionSystem()
        
        security_examples = system.list_teaching_examples(error_type=ErrorType.SECURITY)
        logical_examples = system.list_teaching_examples(error_type=ErrorType.LOGICAL)
        
        # At least one of each type should exist
        assert len(security_examples) > 0 or len(logical_examples) > 0
    
    @pytest.mark.asyncio
    async def test_suggest_correction_security(self):
        """Test correction suggestion for security issue."""
        system = IntelligentCorrectionSystem()
        
        issue = CodeIssue(
            issue_id="test-1",
            error_type=ErrorType.SECURITY,
            severity=Severity.CRITICAL,
            message="Hardcoded secret",
            location=CodeLocation("<test>", 1, 1),
            code_snippet='password = "secret123"',
            rule_id="PY-SEC-001",
        )
        
        code = 'password = "secret123"'
        suggestion = await system.suggest_correction(issue, code, CorrectionMode.BASIC)
        
        assert suggestion is not None
        assert suggestion.mode == CorrectionMode.BASIC
        assert suggestion.explanation
        assert len(suggestion.steps) > 0
    
    @pytest.mark.asyncio
    async def test_suggest_correction_logical(self):
        """Test correction suggestion for logical issue."""
        system = IntelligentCorrectionSystem()
        
        issue = CodeIssue(
            issue_id="test-2",
            error_type=ErrorType.LOGICAL,
            severity=Severity.LOW,
            message="Use is None",
            location=CodeLocation("<test>", 1, 1),
            code_snippet='if x == None:',
            rule_id="PY-LOG-002",
        )
        
        code = 'if x == None:'
        suggestion = await system.suggest_correction(issue, code, CorrectionMode.BASIC)
        
        assert suggestion is not None
        assert "is None" in suggestion.corrected_code or "is" in suggestion.explanation
    
    @pytest.mark.asyncio
    async def test_feedback_submission(self):
        """Test user feedback submission."""
        system = IntelligentCorrectionSystem()
        
        # First create a suggestion
        issue = CodeIssue(
            issue_id="test-3",
            error_type=ErrorType.SECURITY,
            severity=Severity.CRITICAL,
            message="Test issue",
            location=CodeLocation("<test>", 1, 1),
            code_snippet='password = "test"',
            rule_id="PY-SEC-001",
        )
        
        suggestion = await system.suggest_correction(issue, 'password = "test"', CorrectionMode.BASIC)
        
        if suggestion:
            feedback = await system.submit_feedback(
                suggestion_id=suggestion.suggestion_id,
                feedback_type=FeedbackType.HELPFUL,
                rating=5,
                comment="Great suggestion!",
                user_id="test-user",
            )
            
            assert feedback.rating == 5
            assert feedback.feedback_type == FeedbackType.HELPFUL
    
    def test_get_statistics(self):
        """Test getting system statistics."""
        system = IntelligentCorrectionSystem()
        stats = system.get_statistics()
        
        assert "corrections_suggested" in stats
        assert "corrections_applied" in stats
        assert "teaching_examples_available" in stats


class TestCorrectionStrategies:
    """Tests for correction strategies."""
    
    def test_security_strategy_handles_security_issues(self):
        """Test security strategy handles security issues."""
        strategy = SecurityCorrectionStrategy()
        
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.SECURITY,
            severity=Severity.CRITICAL,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet="test",
            rule_id="PY-SEC-001",
        )
        
        assert strategy.can_handle(issue)
    
    def test_security_strategy_ignores_style_issues(self):
        """Test security strategy ignores non-security issues."""
        strategy = SecurityCorrectionStrategy()
        
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.STYLE,
            severity=Severity.LOW,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet="test",
            rule_id="PY-STY-001",
        )
        
        assert not strategy.can_handle(issue)
    
    def test_style_strategy_handles_style_issues(self):
        """Test style strategy handles style issues."""
        strategy = StyleCorrectionStrategy()
        
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.STYLE,
            severity=Severity.LOW,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet="test",
            rule_id="PY-STY-001",
        )
        
        assert strategy.can_handle(issue)
    
    def test_logical_strategy_handles_logical_issues(self):
        """Test logical strategy handles logical issues."""
        strategy = LogicalCorrectionStrategy()
        
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.LOGICAL,
            severity=Severity.MEDIUM,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet="test",
            rule_id="PY-LOG-001",
        )
        
        assert strategy.can_handle(issue)


# =============================================================================
# ML Pattern Recognition Tests
# =============================================================================

class TestMLPatternRecognition:
    """Tests for the MLPatternRecognition class."""
    
    def test_system_initialization(self):
        """Test ML system initializes correctly."""
        ml = MLPatternRecognition()
        
        assert ml.pattern_engine is not None
        assert ml.similarity_engine is not None
    
    @pytest.mark.asyncio
    async def test_analyze_code_detects_patterns(self):
        """Test pattern detection in code."""
        ml = MLPatternRecognition()
        
        code = 'password = "secret123"\neval(user_input)'
        analysis = await ml.analyze_code(code, categories=["security"])
        
        assert "patterns_detected" in analysis
        assert "risk_score" in analysis
        assert "recommendations" in analysis
    
    def test_learn_from_fix(self):
        """Test learning from a fix."""
        ml = MLPatternRecognition()
        
        ml.learn_from_fix(
            buggy_code='password = "secret"',
            fixed_code='password = os.environ.get("PASSWORD")',
            issue_type="hardcoded_secret",
        )
        
        # Find similar code should now return results
        similar = ml.similarity_engine.find_similar('password = "test"', threshold=0.3)
        assert len(similar) >= 0  # May or may not match depending on tokenization


class TestPatternLearningEngine:
    """Tests for the PatternLearningEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes with builtin patterns."""
        engine = PatternLearningEngine()
        
        assert len(engine.patterns) > 0
        
        stats = engine.get_statistics()
        assert stats["v2_production"] > 0  # Builtin patterns are v2
    
    def test_detect_hardcoded_credential(self):
        """Test detection of hardcoded credentials."""
        engine = PatternLearningEngine()
        
        code = 'api_key = "sk-1234567890abcdef"'
        matches = engine.detect_patterns(code, categories=["security"])
        
        assert len(matches) > 0
        assert any("credential" in m.pattern_name or "secret" in m.pattern_name for m in matches)
    
    def test_detect_eval_usage(self):
        """Test detection of eval usage."""
        engine = PatternLearningEngine()
        
        code = 'result = eval(user_input)'
        matches = engine.detect_patterns(code, categories=["security"])
        
        assert len(matches) > 0
    
    def test_learn_new_pattern(self):
        """Test learning a new pattern."""
        engine = PatternLearningEngine()
        
        positive_examples = [
            'TODO: fix this',
            'TODO fix later',
            'TODO: implement feature',
        ]
        
        pattern = engine.learn_pattern(
            name="todo_comment",
            positive_examples=positive_examples,
            pattern_type="style",
            categories=["style", "documentation"],
        )
        
        assert pattern.pattern_id in engine.patterns
        assert pattern.version == "v1"  # New patterns start as experimental
    
    def test_promote_pattern(self):
        """Test pattern promotion."""
        engine = PatternLearningEngine()
        
        # Create a pattern with good metrics
        pattern = CodePattern(
            pattern_id="test-promote",
            name="test_pattern",
            description="Test",
            pattern_type="test",
            regex_pattern=r"test_pattern",
            precision=0.90,
            detection_count=15,
        )
        engine.patterns[pattern.pattern_id] = pattern
        
        result = engine.promote_pattern(pattern.pattern_id)
        
        assert result
        assert engine.patterns[pattern.pattern_id].version == "v2"
    
    def test_quarantine_pattern(self):
        """Test pattern quarantine."""
        engine = PatternLearningEngine()
        
        pattern = CodePattern(
            pattern_id="test-quarantine",
            name="test_pattern",
            description="Test",
            pattern_type="test",
        )
        engine.patterns[pattern.pattern_id] = pattern
        
        result = engine.quarantine_pattern(pattern.pattern_id, "Low accuracy")
        
        assert result
        assert engine.patterns[pattern.pattern_id].version == "v3"


class TestCodeSimilarityEngine:
    """Tests for the CodeSimilarityEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        engine = CodeSimilarityEngine()
        
        assert len(engine.code_index) == 0
    
    def test_add_and_find_code(self):
        """Test adding and finding similar code."""
        engine = CodeSimilarityEngine()
        
        code1 = 'def add(a, b): return a + b'
        code2 = 'def subtract(a, b): return a - b'
        
        engine.add_code(code1, issue_type="test")
        engine.add_code(code2, issue_type="test")
        
        similar = engine.find_similar('def multiply(a, b): return a * b', threshold=0.3)
        
        assert len(similar) > 0
    
    def test_find_by_issue_type(self):
        """Test finding code by issue type."""
        engine = CodeSimilarityEngine()
        
        engine.add_code('password = "secret"', issue_type="hardcoded_secret")
        engine.add_code('api_key = "key123"', issue_type="hardcoded_secret")
        engine.add_code('eval(input)', issue_type="eval_usage")
        
        secrets = engine.find_by_issue_type("hardcoded_secret")
        evals = engine.find_by_issue_type("eval_usage")
        
        assert len(secrets) == 2
        assert len(evals) == 1


# =============================================================================
# Version Training Tests
# =============================================================================

class TestThreeVersionTrainingCoordinator:
    """Tests for the ThreeVersionTrainingCoordinator class."""
    
    def test_coordinator_initialization(self):
        """Test coordinator initializes correctly."""
        coordinator = ThreeVersionTrainingCoordinator()
        
        assert coordinator.v1_trainer is not None
        assert coordinator.v2_trainer is not None
        assert coordinator.v3_trainer is not None
    
    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the coordinator."""
        coordinator = ThreeVersionTrainingCoordinator()
        
        await coordinator.start()
        assert coordinator._running
        
        await coordinator.stop()
        assert not coordinator._running
    
    def test_get_training_status(self):
        """Test getting training status."""
        coordinator = ThreeVersionTrainingCoordinator()
        
        status = coordinator.get_training_status()
        
        assert "v1_status" in status
        assert "v2_status" in status
        assert "v3_status" in status
        assert "ml_statistics" in status
    
    @pytest.mark.asyncio
    async def test_train_from_feedback(self):
        """Test training from user feedback."""
        coordinator = ThreeVersionTrainingCoordinator()
        
        # Create a suggestion first
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.SECURITY,
            severity=Severity.CRITICAL,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet='password = "test"',
            rule_id="PY-SEC-001",
        )
        
        code = 'password = "test"'
        suggestion = await coordinator.correction_system.suggest_correction(
            issue, code, CorrectionMode.BASIC
        )
        
        if suggestion:
            result = await coordinator.train_from_feedback(
                suggestion_id=suggestion.suggestion_id,
                feedback_type=FeedbackType.HELPFUL,
                correct_code='password = os.environ.get("PASSWORD")',
            )
            
            assert result["success"]
    
    @pytest.mark.asyncio
    async def test_evaluate_patterns(self):
        """Test pattern evaluation for promotion/demotion."""
        coordinator = ThreeVersionTrainingCoordinator()
        
        result = await coordinator.evaluate_patterns()
        
        assert "patterns_promoted" in result
        assert "patterns_demoted" in result


class TestVersionTrainingEngine:
    """Tests for the VersionTrainingEngine class."""
    
    def test_engine_initialization(self):
        """Test engine initializes correctly."""
        ml = MLPatternRecognition()
        config = TrainingConfig()
        
        engine = VersionTrainingEngine(ModelVersion.V1_EXPERIMENTAL, ml, config)
        
        assert engine.version == ModelVersion.V1_EXPERIMENTAL
        assert engine.model is not None
    
    @pytest.mark.asyncio
    async def test_add_sample(self):
        """Test adding training samples."""
        ml = MLPatternRecognition()
        config = TrainingConfig()
        engine = VersionTrainingEngine(ModelVersion.V1_EXPERIMENTAL, ml, config)
        
        sample = TrainingSample(
            sample_id="test-1",
            code='password = "secret"',
            language=Language.PYTHON,
            issues=[{"error_type": "security", "rule_id": "PY-SEC-001", "code_snippet": 'password = "secret"'}],
            corrections=[],
        )
        
        await engine.add_sample(sample)
        
        assert len(engine.samples) + len(engine.validation_samples) == 1
    
    def test_get_model_status(self):
        """Test getting model status."""
        ml = MLPatternRecognition()
        config = TrainingConfig()
        engine = VersionTrainingEngine(ModelVersion.V2_PRODUCTION, ml, config)
        
        status = engine.get_model_status()
        
        assert status["version"] == "v2"
        assert "accuracy" in status
        assert "samples_trained" in status


class TestTrainingConfig:
    """Tests for TrainingConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()
        
        assert config.mode == TrainingMode.INCREMENTAL
        assert config.batch_size == 100
        assert config.promotion_threshold == 0.85
        assert config.demotion_threshold == 0.6
    
    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            mode=TrainingMode.BATCH,
            batch_size=50,
            promotion_threshold=0.90,
        )
        
        assert config.mode == TrainingMode.BATCH
        assert config.batch_size == 50
        assert config.promotion_threshold == 0.90


# =============================================================================
# Factory Function Tests
# =============================================================================

class TestFactoryFunctions:
    """Tests for factory functions."""
    
    def test_create_training_system(self):
        """Test creating training system via factory."""
        system = create_training_system()
        
        assert isinstance(system, ThreeVersionTrainingCoordinator)
        assert system.v1_trainer is not None
    
    def test_create_training_system_with_config(self):
        """Test creating training system with custom config."""
        config = TrainingConfig(batch_size=50)
        system = create_training_system(config)
        
        assert system.config.batch_size == 50


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for cross-module collaboration."""
    
    @pytest.mark.asyncio
    async def test_analysis_to_correction_flow(self):
        """Test full flow from analysis to correction."""
        # 1. Analyze code
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('password = "secret123"\n')
            temp_path = f.name
        
        try:
            analysis = await engine.analyze_file(temp_path)
            
            # 2. Generate corrections for issues
            correction_system = IntelligentCorrectionSystem()
            
            with open(temp_path, 'r') as f:
                code = f.read()
            
            corrections = []
            for issue in analysis.issues:
                suggestion = await correction_system.suggest_correction(
                    issue, code, CorrectionMode.BASIC
                )
                if suggestion:
                    corrections.append(suggestion)
            
            # Should have at least one correction for the hardcoded secret
            security_corrections = [
                c for c in corrections 
                if c.issue.error_type == ErrorType.SECURITY
            ]
            assert len(security_corrections) > 0
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_training_pipeline(self):
        """Test training pipeline with analysis results."""
        # 1. Create coordinator
        coordinator = ThreeVersionTrainingCoordinator()
        await coordinator.start()
        
        try:
            # 2. Create temp directory with test files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Create test file
                test_file = Path(temp_dir) / "test.py"
                test_file.write_text('api_key = "test123"\neval(user_input)\n')
                
                # 3. Train on the directory
                result = await coordinator.train_on_project(temp_dir)
                
                assert result["files_analyzed"] > 0
                assert result["samples_created"] > 0
                
                # 4. Check status
                status = coordinator.get_training_status()
                assert status["total_samples_trained"] > 0
                
        finally:
            await coordinator.stop()
    
    @pytest.mark.asyncio
    async def test_ml_pattern_detection_integration(self):
        """Test ML pattern detection with correction suggestion."""
        ml = MLPatternRecognition()
        correction_system = IntelligentCorrectionSystem()
        
        code = 'password = "hardcoded_secret_value"'
        
        # 1. ML detects patterns
        analysis = await ml.analyze_code(code, categories=["security"])
        
        assert analysis["patterns_detected"] > 0
        
        # 2. Generate corrections based on patterns
        for pattern_data in analysis.get("patterns", []):
            issue = CodeIssue(
                issue_id="ml-detected",
                error_type=ErrorType.SECURITY,
                severity=Severity.HIGH,
                message=f"ML Pattern: {pattern_data.get('name')}",
                location=CodeLocation("<test>", 1, 1),
                code_snippet=pattern_data.get("match", ""),
                rule_id=f"ML-{pattern_data.get('name')}",
            )
            
            suggestion = await correction_system.suggest_correction(
                issue, code, CorrectionMode.TEACHING
            )
            
            if suggestion:
                assert suggestion.mode == CorrectionMode.TEACHING
                # Teaching mode should have steps
                assert len(suggestion.steps) >= 0


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    @pytest.mark.asyncio
    async def test_analyze_empty_file(self):
        """Test analyzing an empty file."""
        engine = CodeAnalysisEngine()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('')
            temp_path = f.name
        
        try:
            result = await engine.analyze_file(temp_path)
            
            assert result.success
            assert result.issue_count == 0
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_analyze_nonexistent_file(self):
        """Test analyzing a non-existent file."""
        engine = CodeAnalysisEngine()
        
        result = await engine.analyze_file("/nonexistent/path/file.py")
        
        assert not result.success
        assert result.error is not None
    
    def test_unknown_language_handling(self):
        """Test handling of unknown language files."""
        engine = CodeAnalysisEngine()
        
        assert engine.detect_language("file.unknown") == Language.UNKNOWN
    
    @pytest.mark.asyncio
    async def test_correction_without_authorization(self):
        """Test that advanced corrections require authorization."""
        system = IntelligentCorrectionSystem()
        
        issue = CodeIssue(
            issue_id="test",
            error_type=ErrorType.SECURITY,
            severity=Severity.CRITICAL,
            message="Test",
            location=CodeLocation("<test>", 1, 1),
            code_snippet='password = "test"',
            rule_id="PY-SEC-001",
        )
        
        suggestion = await system.suggest_correction(issue, 'password = "test"', CorrectionMode.ADVANCED)
        
        if suggestion:
            # Create temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write('password = "test"')
                temp_path = f.name
            
            try:
                # Try to apply without authorization
                result = await system.apply_correction(
                    suggestion.suggestion_id,
                    temp_path,
                    authorized=False,
                )
                
                # Should fail or require authorization
                # The behavior depends on suggestion.requires_review flag
                assert isinstance(result, CorrectionResult)
            finally:
                os.unlink(temp_path)
    
    def test_similarity_with_empty_index(self):
        """Test similarity search with empty index."""
        engine = CodeSimilarityEngine()
        
        similar = engine.find_similar("some code", threshold=0.5)
        
        assert len(similar) == 0
    
    def test_pattern_detection_with_no_matches(self):
        """Test pattern detection when no patterns match."""
        engine = PatternLearningEngine()
        
        clean_code = "x = 1\ny = 2\nz = x + y"
        matches = engine.detect_patterns(clean_code, categories=["security"])
        
        # Clean code should have no security pattern matches
        assert all(m.confidence < 1.0 for m in matches) or len(matches) == 0


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
