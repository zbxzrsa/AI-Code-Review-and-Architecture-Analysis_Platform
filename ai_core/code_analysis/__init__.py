"""
Code Analysis and Intelligent Correction Module

A comprehensive code analysis engine that scans project directories and identifies:
- Syntax Errors
- Runtime Errors
- Logical Errors
- Security Vulnerabilities
- Performance Issues
- Coding Standard Violations

Features:
- Multi-language support (Python, JavaScript/TypeScript, Java, etc.)
- Intelligent correction with three modes:
  * Basic: Detailed explanations and step-by-step instructions
  * Advanced: Automatic corrections with user authorization
  * Teaching: Interactive learning examples
- Machine learning pattern recognition
- Version control integration for traceable corrections
- User feedback mechanism for continuous improvement

Three-Version Integration:
- V1 (Experimental): Trains on new patterns and techniques
- V2 (Production): Uses validated models with >95% accuracy
- V3 (Quarantine): Analyzes failed approaches for learning

Usage:
    from ai_core.code_analysis import (
        CodeAnalysisEngine,
        IntelligentCorrectionSystem,
        ThreeVersionTrainingCoordinator,
        CorrectionMode,
    )
    
    # Analyze a project
    engine = CodeAnalysisEngine()
    analysis = await engine.analyze_directory("./my_project")
    
    # Generate corrections
    correction_system = IntelligentCorrectionSystem()
    for issue in analysis.file_results[0].issues:
        suggestion = await correction_system.suggest_correction(
            issue,
            code,
            CorrectionMode.BASIC  # or ADVANCED, TEACHING
        )
    
    # Train models
    trainer = ThreeVersionTrainingCoordinator()
    await trainer.start()
    await trainer.train_on_project("./my_project")
"""

from .analysis_engine import (
    # Enums
    ErrorType,
    Severity,
    Language,
    LANGUAGE_EXTENSIONS,
    # Data classes
    CodeLocation,
    CodeIssue,
    AnalysisResult,
    ProjectAnalysis,
    AnalysisRule,
    # Analyzers
    LanguageAnalyzer,
    PythonAnalyzer,
    JavaScriptAnalyzer,
    JavaAnalyzer,
    # Main engine
    CodeAnalysisEngine,
)

from .correction_system import (
    # Enums
    CorrectionMode,
    CorrectionStatus,
    FeedbackType,
    # Data classes
    CorrectionStep,
    CorrectionSuggestion,
    CorrectionResult,
    UserFeedback,
    TeachingExample,
    # Strategies
    CorrectionStrategy,
    SecurityCorrectionStrategy,
    StyleCorrectionStrategy,
    LogicalCorrectionStrategy,
    # Main system
    IntelligentCorrectionSystem,
)

from .ml_pattern_recognition import (
    # Data classes
    CodePattern,
    PatternMatch,
    CodeEmbedding,
    SimilarCode,
    # Engines
    PatternLearningEngine,
    CodeSimilarityEngine,
    # Main system
    MLPatternRecognition,
)

from .version_training import (
    # Enums
    TrainingMode,
    ModelVersion,
    # Config
    TrainingConfig,
    # Data classes
    TrainingSample,
    TrainingMetrics,
    VersionModel,
    # Engines
    VersionTrainingEngine,
    # Coordinator
    ThreeVersionTrainingCoordinator,
    # Factory
    create_training_system,
)


__all__ = [
    # ==========================================================================
    # Analysis Engine
    # ==========================================================================
    # Enums
    "ErrorType",
    "Severity",
    "Language",
    "LANGUAGE_EXTENSIONS",
    # Data classes
    "CodeLocation",
    "CodeIssue",
    "AnalysisResult",
    "ProjectAnalysis",
    "AnalysisRule",
    # Analyzers
    "LanguageAnalyzer",
    "PythonAnalyzer",
    "JavaScriptAnalyzer",
    "JavaAnalyzer",
    # Main engine
    "CodeAnalysisEngine",
    
    # ==========================================================================
    # Correction System
    # ==========================================================================
    # Enums
    "CorrectionMode",
    "CorrectionStatus",
    "FeedbackType",
    # Data classes
    "CorrectionStep",
    "CorrectionSuggestion",
    "CorrectionResult",
    "UserFeedback",
    "TeachingExample",
    # Strategies
    "CorrectionStrategy",
    "SecurityCorrectionStrategy",
    "StyleCorrectionStrategy",
    "LogicalCorrectionStrategy",
    # Main system
    "IntelligentCorrectionSystem",
    
    # ==========================================================================
    # ML Pattern Recognition
    # ==========================================================================
    # Data classes
    "CodePattern",
    "PatternMatch",
    "CodeEmbedding",
    "SimilarCode",
    # Engines
    "PatternLearningEngine",
    "CodeSimilarityEngine",
    # Main system
    "MLPatternRecognition",
    
    # ==========================================================================
    # Version Training
    # ==========================================================================
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


# Module version
__version__ = "1.0.0"
