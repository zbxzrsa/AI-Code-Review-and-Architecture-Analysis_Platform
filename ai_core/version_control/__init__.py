"""
Model Version Control Module

Module Description:
    Provides AI model version control and management functionality,
    as well as project self-update capabilities.

Main Features:
    - Model version tracking and management
    - Model registry
    - Performance metrics tracking
    - Version history
    - Project self-update engine (new)
    - Enhanced version control AI (new)

Main Components:
    - ModelVersionControl: Main model version control class
    - ModelRegistry: Model registry
    - VersionTracker: Version tracker
    - PerformanceTracker: Performance tracker
    - ProjectSelfUpdateEngine: Project self-update engine (new)
    - EnhancedVersionControlAI: Enhanced version control AI (new)
    - ProjectSelfUpdateService: Project self-update service (new)

Last Modified: 2024-12-07
"""

from .model_version_control import ModelVersionControl
from .model_registry import ModelRegistry
from .version_tracker import VersionTracker
from .performance_tracker import PerformanceTracker

# New: Project self-update related
from .project_self_update_engine import (
    ProjectSelfUpdateEngine,
    ImprovementCategory,
    ImprovementPriority,
    ImprovementStatus,
    CodeIssue,
    ImprovementPatch,
    ProjectScanResult,
    ImprovementCycle,
)
from .enhanced_version_control_ai import (
    EnhancedVersionControlAI,
    VersionControlAIConfig,
)
from .self_update_service import (
    ProjectSelfUpdateService,
    create_self_update_service,
)

__all__ = [
    # Original components
    'ModelVersionControl',
    'ModelRegistry',
    'VersionTracker',
    'PerformanceTracker',
    # New: Project self-update
    'ProjectSelfUpdateEngine',
    'EnhancedVersionControlAI',
    'ProjectSelfUpdateService',
    'create_self_update_service',
    # Enums and data types
    'ImprovementCategory',
    'ImprovementPriority',
    'ImprovementStatus',
    'CodeIssue',
    'ImprovementPatch',
    'ProjectScanResult',
    'ImprovementCycle',
    'VersionControlAIConfig',
]
