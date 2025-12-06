"""
Aggregate Factories and Builders

Provides factory methods and builders for creating aggregates
while enforcing invariants and business rules.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .domain_models import (
    Analysis, AnalysisStatus, CodeHash, Issue, IssueLocation,
    IssueType, Severity, FixSuggestion, Experiment, ExperimentStatus,
    ModelConfiguration, EvaluationMetrics,
)


class AnalysisFactory:
    """Factory for creating Analysis aggregates."""
    
    @staticmethod
    def create(
        project_id: str,
        code: str,
        language: str,
        rules: Optional[List[str]] = None
    ) -> Analysis:
        """
        Create a new Analysis aggregate.
        
        Args:
            project_id: ID of the project
            code: Source code to analyze
            language: Programming language
            rules: Optional list of rules to apply
            
        Returns:
            New Analysis aggregate
        """
        return Analysis.create(
            project_id=project_id,
            code=code,
            language=language,
        )
    
    @staticmethod
    def reconstitute(
        analysis_id: str,
        project_id: str,
        code_hash: str,
        language: str,
        status: str,
        created_at: datetime,
        completed_at: Optional[datetime] = None,
        issues: Optional[List[Dict]] = None,
        version: int = 0
    ) -> Analysis:
        """
        Reconstitute an Analysis from persistence.
        
        Used by repositories to rebuild aggregates from stored data.
        """
        analysis = Analysis(
            _id=analysis_id,
            project_id=project_id,
            code_hash=CodeHash(value=code_hash),
            language=language,
            status=AnalysisStatus(status),
            created_at=created_at,
            completed_at=completed_at,
        )
        analysis._version = version
        
        # Reconstitute issues
        if issues:
            for issue_data in issues:
                issue = Issue(
                    _id=issue_data["id"],
                    issue_type=IssueType(issue_data["type"]),
                    severity=Severity(issue_data["severity"]),
                    message=issue_data["message"],
                    location=IssueLocation(
                        file_path=issue_data["location"]["file"],
                        line_start=issue_data["location"]["line_start"],
                        line_end=issue_data["location"]["line_end"],
                    ),
                    rule_id=issue_data["rule_id"],
                    is_resolved=issue_data.get("is_resolved", False),
                )
                analysis._issues.append(issue)
        
        return analysis


class ExperimentFactory:
    """Factory for creating Experiment aggregates."""
    
    @staticmethod
    def create(
        name: str,
        provider: str,
        model_name: str,
        created_by: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        baseline_id: Optional[str] = None,
        additional_params: Optional[Dict[str, Any]] = None
    ) -> Experiment:
        """
        Create a new Experiment aggregate.
        
        Args:
            name: Experiment name
            provider: AI provider (e.g., "openai", "anthropic")
            model_name: Model identifier
            created_by: ID of user creating the experiment
            temperature: Model temperature
            max_tokens: Max tokens for generation
            baseline_id: Optional baseline experiment ID
            additional_params: Additional model parameters
            
        Returns:
            New Experiment aggregate
        """
        configuration = ModelConfiguration(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=additional_params or {},
        )
        
        return Experiment.create(
            name=name,
            configuration=configuration,
            created_by=created_by,
            baseline_id=baseline_id,
        )
    
    @staticmethod
    def reconstitute(
        experiment_id: str,
        name: str,
        configuration: Dict[str, Any],
        status: str,
        created_at: datetime,
        created_by: str,
        evaluations: Optional[List[Dict]] = None,
        baseline_id: Optional[str] = None,
        promoted_at: Optional[datetime] = None,
        quarantined_at: Optional[datetime] = None,
        quarantine_reason: Optional[str] = None,
        version: int = 0
    ) -> Experiment:
        """Reconstitute an Experiment from persistence."""
        config = ModelConfiguration(
            provider=configuration["provider"],
            model_name=configuration["model_name"],
            temperature=configuration["temperature"],
            max_tokens=configuration["max_tokens"],
            additional_params=configuration.get("additional_params", {}),
        )
        
        experiment = Experiment(
            _id=experiment_id,
            name=name,
            configuration=config,
            status=ExperimentStatus(status),
            created_at=created_at,
            created_by=created_by,
            baseline_id=baseline_id,
            promoted_at=promoted_at,
            quarantined_at=quarantined_at,
            quarantine_reason=quarantine_reason,
        )
        experiment._version = version
        
        return experiment


class IssueBuilder:
    """Builder for creating Issue entities with fluent API."""
    
    def __init__(self):
        self._issue_type: IssueType = IssueType.QUALITY
        self._severity: Severity = Severity.MEDIUM
        self._message: str = ""
        self._file_path: str = ""
        self._line_start: int = 1
        self._line_end: int = 1
        self._column_start: int = 0
        self._column_end: int = 0
        self._rule_id: str = ""
        self._suggestion: Optional[FixSuggestion] = None
    
    def with_type(self, issue_type: IssueType) -> 'IssueBuilder':
        self._issue_type = issue_type
        return self
    
    def with_severity(self, severity: Severity) -> 'IssueBuilder':
        self._severity = severity
        return self
    
    def with_message(self, message: str) -> 'IssueBuilder':
        self._message = message
        return self
    
    def at_location(
        self,
        file_path: str,
        line_start: int,
        line_end: Optional[int] = None,
        column_start: int = 0,
        column_end: int = 0
    ) -> 'IssueBuilder':
        self._file_path = file_path
        self._line_start = line_start
        self._line_end = line_end or line_start
        self._column_start = column_start
        self._column_end = column_end
        return self
    
    def with_rule(self, rule_id: str) -> 'IssueBuilder':
        self._rule_id = rule_id
        return self
    
    def with_suggestion(
        self,
        description: str,
        old_code: str,
        new_code: str,
        confidence: float = 0.8
    ) -> 'IssueBuilder':
        self._suggestion = FixSuggestion(
            description=description,
            old_code=old_code,
            new_code=new_code,
            confidence=confidence,
        )
        return self
    
    def build(self) -> Issue:
        """Build the Issue entity."""
        if not self._message:
            raise ValueError("Issue message is required")
        if not self._file_path:
            raise ValueError("File path is required")
        if not self._rule_id:
            raise ValueError("Rule ID is required")
        
        location = IssueLocation(
            file_path=self._file_path,
            line_start=self._line_start,
            line_end=self._line_end,
            column_start=self._column_start,
            column_end=self._column_end,
        )
        
        return Issue.create(
            issue_type=self._issue_type,
            severity=self._severity,
            message=self._message,
            location=location,
            rule_id=self._rule_id,
            suggestion=self._suggestion,
        )


# Convenience functions
def create_security_issue(
    message: str,
    file_path: str,
    line: int,
    severity: Severity = Severity.HIGH
) -> Issue:
    """Create a security issue with common defaults."""
    return (
        IssueBuilder()
        .with_type(IssueType.SECURITY)
        .with_severity(severity)
        .with_message(message)
        .at_location(file_path, line)
        .with_rule("SEC001")
        .build()
    )


def create_performance_issue(
    message: str,
    file_path: str,
    line: int,
    severity: Severity = Severity.MEDIUM
) -> Issue:
    """Create a performance issue with common defaults."""
    return (
        IssueBuilder()
        .with_type(IssueType.PERFORMANCE)
        .with_severity(severity)
        .with_message(message)
        .at_location(file_path, line)
        .with_rule("PERF001")
        .build()
    )
