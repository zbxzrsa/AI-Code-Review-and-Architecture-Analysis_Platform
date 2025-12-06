"""
Risk Management Registry (Risk Assessment Implementation)

Provides risk tracking and monitoring with:
- Risk registration and classification
- Mitigation tracking
- KRI monitoring
- Alert generation

Supports risks:
- R-001: AI Provider Service Interruption
- R-002: Data Leakage
- R-003: Insufficient Test Coverage
- R-004: Technical Debt Accumulation
- R-005: Inadequate Key Management
- R-006: Third-Party Dependency Vulnerabilities
"""
import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import hashlib

logger = logging.getLogger(__name__)


class RiskLevel(str, Enum):
    """Risk severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskCategory(str, Enum):
    """Risk categories."""
    SECURITY = "security"
    RELIABILITY = "reliability"
    QUALITY = "quality"
    TECHNICAL = "technical"
    COMPLIANCE = "compliance"
    OPERATIONAL = "operational"


class RiskStatus(str, Enum):
    """Risk status."""
    OPEN = "open"
    MITIGATING = "mitigating"
    MITIGATED = "mitigated"
    ACCEPTED = "accepted"
    CLOSED = "closed"


class MitigationStatus(str, Enum):
    """Mitigation implementation status."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    VERIFIED = "verified"


@dataclass
class RiskIndicator:
    """Key Risk Indicator (KRI)."""
    name: str
    description: str
    threshold_warning: float
    threshold_critical: float
    current_value: Optional[float] = None
    unit: str = ""
    last_updated: Optional[datetime] = None
    
    @property
    def status(self) -> str:
        if self.current_value is None:
            return "unknown"
        if self.current_value >= self.threshold_critical:
            return "critical"
        if self.current_value >= self.threshold_warning:
            return "warning"
        return "healthy"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "current_value": self.current_value,
            "unit": self.unit,
            "status": self.status,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }


@dataclass
class Mitigation:
    """Risk mitigation action."""
    id: str
    description: str
    owner: str
    status: MitigationStatus = MitigationStatus.NOT_STARTED
    due_date: Optional[datetime] = None
    completed_date: Optional[datetime] = None
    acceptance_criteria: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "owner": self.owner,
            "status": self.status.value,
            "due_date": self.due_date.isoformat() if self.due_date else None,
            "completed_date": self.completed_date.isoformat() if self.completed_date else None,
            "acceptance_criteria": self.acceptance_criteria,
            "notes": self.notes,
        }


@dataclass
class Risk:
    """Risk definition."""
    id: str
    title: str
    description: str
    category: RiskCategory
    probability: str  # low, medium, high
    impact: str  # low, medium, high, critical
    level: RiskLevel
    status: RiskStatus = RiskStatus.OPEN
    owner: str = ""
    created_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_review: Optional[datetime] = None
    indicators: List[RiskIndicator] = field(default_factory=list)
    mitigations: List[Mitigation] = field(default_factory=list)
    notes: str = ""
    
    def calculate_score(self) -> int:
        """Calculate risk score (1-25)."""
        probability_scores = {"low": 1, "medium": 2, "high": 3}
        impact_scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        
        p = probability_scores.get(self.probability.lower(), 2)
        i = impact_scores.get(self.impact.lower(), 2)
        
        return p * i
    
    @property
    def mitigation_progress(self) -> float:
        """Calculate mitigation progress percentage."""
        if not self.mitigations:
            return 0.0
        
        completed = sum(1 for m in self.mitigations 
                       if m.status in [MitigationStatus.COMPLETED, MitigationStatus.VERIFIED])
        return (completed / len(self.mitigations)) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "category": self.category.value,
            "probability": self.probability,
            "impact": self.impact,
            "level": self.level.value,
            "score": self.calculate_score(),
            "status": self.status.value,
            "owner": self.owner,
            "created_date": self.created_date.isoformat(),
            "last_review": self.last_review.isoformat() if self.last_review else None,
            "indicators": [i.to_dict() for i in self.indicators],
            "mitigations": [m.to_dict() for m in self.mitigations],
            "mitigation_progress": self.mitigation_progress,
            "notes": self.notes,
        }


class RiskRegistry:
    """
    Central registry for risk management.
    
    Features:
    - Risk registration and tracking
    - KRI monitoring
    - Mitigation tracking
    - Alert generation
    - Reporting
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.risks: Dict[str, Risk] = {}
        self.storage_path = Path(storage_path) if storage_path else None
        self._alert_callbacks: List[Callable] = []
        self._initialize_standard_risks()
    
    def _initialize_standard_risks(self):
        """Initialize standard platform risks."""
        standard_risks = [
            Risk(
                id="R-001",
                title="AI Provider Service Interruption",
                description="Main AI service provider API unavailable or response delay exceeds 30 seconds",
                category=RiskCategory.RELIABILITY,
                probability="medium",
                impact="high",
                level=RiskLevel.HIGH,
                owner="DevOps Team",
                indicators=[
                    RiskIndicator(
                        name="API Availability",
                        description="AI provider API uptime",
                        threshold_warning=99.5,
                        threshold_critical=99.0,
                        unit="%",
                    ),
                    RiskIndicator(
                        name="Response Time P95",
                        description="95th percentile response time",
                        threshold_warning=5000,
                        threshold_critical=10000,
                        unit="ms",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-001-1",
                        description="Multi-provider fallback chain",
                        owner="DevOps",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["Failover time < 5 seconds"],
                    ),
                    Mitigation(
                        id="M-001-2",
                        description="Local Ollama backup",
                        owner="DevOps",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["Local model functional"],
                    ),
                ],
                status=RiskStatus.MITIGATED,
            ),
            Risk(
                id="R-002",
                title="Data Leakage",
                description="Sensitive data leaked through API responses, logs, or storage layers",
                category=RiskCategory.SECURITY,
                probability="low",
                impact="critical",
                level=RiskLevel.HIGH,
                owner="Security Team",
                indicators=[
                    RiskIndicator(
                        name="Security Incidents",
                        description="Number of security incidents",
                        threshold_warning=1,
                        threshold_critical=3,
                        unit="count",
                    ),
                    RiskIndicator(
                        name="PII in Logs",
                        description="PII detected in logs",
                        threshold_warning=1,
                        threshold_critical=5,
                        unit="occurrences",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-002-1",
                        description="AES-256 encryption at rest",
                        owner="Security",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["All data encrypted"],
                    ),
                    Mitigation(
                        id="M-002-2",
                        description="TLS 1.3 encryption in transit",
                        owner="Security",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["TLS 1.3 enforced"],
                    ),
                    Mitigation(
                        id="M-002-3",
                        description="RBAC implementation",
                        owner="Security",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["Minimum privilege enforced"],
                    ),
                    Mitigation(
                        id="M-002-4",
                        description="Audit logging (365 days)",
                        owner="Security",
                        status=MitigationStatus.VERIFIED,
                        acceptance_criteria=["Tamper-proof logs"],
                    ),
                ],
                status=RiskStatus.MITIGATED,
            ),
            Risk(
                id="R-003",
                title="Insufficient Test Coverage",
                description="Unit test coverage below 80%, integration test coverage below 60%",
                category=RiskCategory.QUALITY,
                probability="medium",
                impact="medium",
                level=RiskLevel.MEDIUM,
                owner="QA Team",
                indicators=[
                    RiskIndicator(
                        name="Unit Test Coverage",
                        description="Code coverage percentage",
                        threshold_warning=75,
                        threshold_critical=70,
                        unit="%",
                    ),
                    RiskIndicator(
                        name="Critical Bugs",
                        description="Critical bugs in production",
                        threshold_warning=2,
                        threshold_critical=5,
                        unit="count",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-003-1",
                        description="Increase unit test coverage to 80%",
                        owner="QA",
                        status=MitigationStatus.IN_PROGRESS,
                        acceptance_criteria=["SonarQube shows >= 80%"],
                    ),
                    Mitigation(
                        id="M-003-2",
                        description="Introduce mutation testing",
                        owner="QA",
                        status=MitigationStatus.NOT_STARTED,
                        acceptance_criteria=["Mutation score >= 70%"],
                    ),
                ],
                status=RiskStatus.MITIGATING,
            ),
            Risk(
                id="R-004",
                title="Technical Debt Accumulation",
                description="Code duplication > 15%, documentation missing > 20%",
                category=RiskCategory.TECHNICAL,
                probability="medium",
                impact="medium",
                level=RiskLevel.MEDIUM,
                owner="Engineering Lead",
                indicators=[
                    RiskIndicator(
                        name="Code Duplication",
                        description="Percentage of duplicated code",
                        threshold_warning=12,
                        threshold_critical=15,
                        unit="%",
                    ),
                    RiskIndicator(
                        name="Documentation Coverage",
                        description="Percentage of documented code",
                        threshold_warning=85,
                        threshold_critical=80,
                        unit="%",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-004-1",
                        description="Quarterly technical debt sprint",
                        owner="Engineering",
                        status=MitigationStatus.NOT_STARTED,
                        acceptance_criteria=["Duplication < 10%"],
                    ),
                ],
                status=RiskStatus.OPEN,
            ),
            Risk(
                id="R-005",
                title="Inadequate Key Management",
                description="Hard-coded keys or credentials not using HSM/KMS",
                category=RiskCategory.SECURITY,
                probability="low",
                impact="high",
                level=RiskLevel.MEDIUM,
                owner="Security Team",
                indicators=[
                    RiskIndicator(
                        name="Hardcoded Secrets",
                        description="Number of hardcoded secrets detected",
                        threshold_warning=1,
                        threshold_critical=5,
                        unit="count",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-005-1",
                        description="Migrate to AWS KMS/Azure Key Vault",
                        owner="Security",
                        status=MitigationStatus.NOT_STARTED,
                        acceptance_criteria=["HSM-backed keys", "Auto rotation"],
                    ),
                ],
                status=RiskStatus.OPEN,
            ),
            Risk(
                id="R-006",
                title="Third-Party Dependency Vulnerabilities",
                description="Dependencies with known CVE vulnerabilities (score >= 7.0)",
                category=RiskCategory.SECURITY,
                probability="medium",
                impact="medium",
                level=RiskLevel.MEDIUM,
                owner="Security Team",
                indicators=[
                    RiskIndicator(
                        name="Critical CVEs",
                        description="Number of critical vulnerabilities",
                        threshold_warning=1,
                        threshold_critical=3,
                        unit="count",
                    ),
                    RiskIndicator(
                        name="Days Since Last Scan",
                        description="Days since last dependency scan",
                        threshold_warning=3,
                        threshold_critical=7,
                        unit="days",
                    ),
                ],
                mitigations=[
                    Mitigation(
                        id="M-006-1",
                        description="Enable Dependabot daily scanning",
                        owner="DevOps",
                        status=MitigationStatus.IN_PROGRESS,
                        acceptance_criteria=["Daily scans active"],
                    ),
                    Mitigation(
                        id="M-006-2",
                        description="24-hour SLA for critical fixes",
                        owner="Security",
                        status=MitigationStatus.NOT_STARTED,
                        acceptance_criteria=["CVE rate < 5%"],
                    ),
                ],
                status=RiskStatus.MITIGATING,
            ),
        ]
        
        for risk in standard_risks:
            self.risks[risk.id] = risk
    
    def register_risk(self, risk: Risk) -> str:
        """Register a new risk."""
        self.risks[risk.id] = risk
        logger.info(f"Risk registered: {risk.id} - {risk.title}")
        return risk.id
    
    def update_risk(self, risk_id: str, **updates) -> Optional[Risk]:
        """Update an existing risk."""
        risk = self.risks.get(risk_id)
        if not risk:
            return None
        
        for key, value in updates.items():
            if hasattr(risk, key):
                setattr(risk, key, value)
        
        risk.last_review = datetime.now(timezone.utc)
        return risk
    
    def update_indicator(
        self,
        risk_id: str,
        indicator_name: str,
        value: float
    ) -> Optional[RiskIndicator]:
        """Update a risk indicator value."""
        risk = self.risks.get(risk_id)
        if not risk:
            return None
        
        for indicator in risk.indicators:
            if indicator.name == indicator_name:
                old_status = indicator.status
                indicator.current_value = value
                indicator.last_updated = datetime.now(timezone.utc)
                
                # Check for status change
                new_status = indicator.status
                if old_status != new_status and new_status in ["warning", "critical"]:
                    self._trigger_alert(risk, indicator)
                
                return indicator
        
        return None
    
    def update_mitigation(
        self,
        risk_id: str,
        mitigation_id: str,
        status: MitigationStatus,
        notes: str = ""
    ) -> Optional[Mitigation]:
        """Update mitigation status."""
        risk = self.risks.get(risk_id)
        if not risk:
            return None
        
        for mitigation in risk.mitigations:
            if mitigation.id == mitigation_id:
                mitigation.status = status
                mitigation.notes = notes
                
                if status in [MitigationStatus.COMPLETED, MitigationStatus.VERIFIED]:
                    mitigation.completed_date = datetime.now(timezone.utc)
                
                # Update risk status if all mitigations complete
                if risk.mitigation_progress == 100:
                    risk.status = RiskStatus.MITIGATED
                
                return mitigation
        
        return None
    
    def _trigger_alert(self, risk: Risk, indicator: RiskIndicator):
        """Trigger alert for indicator threshold breach."""
        alert = {
            "risk_id": risk.id,
            "risk_title": risk.title,
            "indicator": indicator.name,
            "status": indicator.status,
            "value": indicator.current_value,
            "threshold": indicator.threshold_critical if indicator.status == "critical" else indicator.threshold_warning,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.warning(f"Risk alert: {alert}")
        
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for risk alerts."""
        self._alert_callbacks.append(callback)
    
    def get_risk(self, risk_id: str) -> Optional[Risk]:
        """Get a specific risk."""
        return self.risks.get(risk_id)
    
    def get_all_risks(self) -> List[Risk]:
        """Get all registered risks."""
        return list(self.risks.values())
    
    def get_risks_by_level(self, level: RiskLevel) -> List[Risk]:
        """Get risks by level."""
        return [r for r in self.risks.values() if r.level == level]
    
    def get_risks_by_category(self, category: RiskCategory) -> List[Risk]:
        """Get risks by category."""
        return [r for r in self.risks.values() if r.category == category]
    
    def get_open_risks(self) -> List[Risk]:
        """Get open/mitigating risks."""
        return [r for r in self.risks.values() 
                if r.status in [RiskStatus.OPEN, RiskStatus.MITIGATING]]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate risk assessment report."""
        risks = self.get_all_risks()
        
        # Calculate summary
        by_level = {level.value: 0 for level in RiskLevel}
        by_status = {status.value: 0 for status in RiskStatus}
        by_category = {cat.value: 0 for cat in RiskCategory}
        
        for risk in risks:
            by_level[risk.level.value] += 1
            by_status[risk.status.value] += 1
            by_category[risk.category.value] += 1
        
        # Calculate overall score
        total_score = sum(r.calculate_score() for r in risks)
        max_score = len(risks) * 12  # Max score per risk
        overall_score = (1 - total_score / max_score) * 10 if max_score > 0 else 10
        
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_risks": len(risks),
                "overall_score": round(overall_score, 1),
                "by_level": by_level,
                "by_status": by_status,
                "by_category": by_category,
            },
            "high_priority": [
                r.to_dict() for r in risks 
                if r.level in [RiskLevel.CRITICAL, RiskLevel.HIGH] 
                and r.status not in [RiskStatus.MITIGATED, RiskStatus.CLOSED]
            ],
            "risks": [r.to_dict() for r in risks],
        }
    
    def save(self):
        """Save registry to storage."""
        if not self.storage_path:
            return
        
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "version": "1.0",
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "risks": [r.to_dict() for r in self.risks.values()],
        }
        
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load registry from storage."""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        with open(self.storage_path) as f:
            data = json.load(f)
        
        # Load risks from file
        # (Implementation would reconstruct Risk objects from dict)


# Global registry instance
_registry: Optional[RiskRegistry] = None


def get_risk_registry() -> RiskRegistry:
    """Get or create global risk registry."""
    global _registry
    if _registry is None:
        _registry = RiskRegistry()
    return _registry


# Example usage
if __name__ == "__main__":
    registry = get_risk_registry()
    
    # Update an indicator
    registry.update_indicator("R-003", "Unit Test Coverage", 72.5)
    
    # Generate report
    report = registry.generate_report()
    print(json.dumps(report, indent=2))
