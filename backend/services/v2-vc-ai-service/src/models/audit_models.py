"""
V2 VC-AI Audit Models

Data models for compliance and audit logging.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class AuditAction(str, Enum):
    """Types of auditable actions"""
    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    ACCESS = "ACCESS"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    ANALYZE = "ANALYZE"
    RELEASE = "RELEASE"
    ROLLBACK = "ROLLBACK"
    PROMOTE = "PROMOTE"
    DEPRECATE = "DEPRECATE"
    EXPORT = "EXPORT"


class AuditStatus(str, Enum):
    """Audit entry status"""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    DENIED = "DENIED"
    ERROR = "ERROR"


class AuditExportFormat(str, Enum):
    """Audit export formats"""
    JSON = "JSON"
    CSV = "CSV"
    PDF = "PDF"
    XML = "XML"


class ComplianceStandard(str, Enum):
    """Compliance standards"""
    SOC2 = "SOC2"
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    ISO27001 = "ISO27001"
    PCI_DSS = "PCI_DSS"


class AuditEntry(BaseModel):
    """Individual audit log entry"""
    id: str = Field(..., description="Unique audit entry ID")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Actor information
    user_id: str = Field(..., description="User who performed the action")
    user_email: Optional[str] = None
    user_role: Optional[str] = None
    user_ip: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Action details
    action: AuditAction
    status: AuditStatus
    resource_type: str = Field(..., description="Type of resource affected")
    resource_id: str = Field(..., description="ID of resource affected")
    
    # Request details
    request_id: Optional[str] = None
    request_path: Optional[str] = None
    request_method: Optional[str] = None
    
    # Change details
    previous_state: Optional[Dict[str, Any]] = None
    new_state: Optional[Dict[str, Any]] = None
    changes: Optional[Dict[str, Any]] = None
    
    # Security
    signature: Optional[str] = Field(None, description="Cryptographic signature")
    previous_hash: Optional[str] = Field(None, description="Hash chain link")
    
    # Metadata
    service: str = Field(default="v2-vc-ai-service")
    environment: str = Field(default="production")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AuditQuery(BaseModel):
    """Query parameters for audit log retrieval"""
    start_date: datetime
    end_date: datetime
    action: Optional[AuditAction] = None
    user_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    status: Optional[AuditStatus] = None
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=100, ge=1, le=1000)


class AuditLogResponse(BaseModel):
    """Response for audit log query"""
    entries: List[AuditEntry]
    total_count: int
    page: int
    page_size: int
    has_more: bool
    query_time_ms: float


class AuditExportRequest(BaseModel):
    """Request to export audit logs"""
    start_date: datetime
    end_date: datetime
    format: AuditExportFormat = Field(default=AuditExportFormat.JSON)
    include_signatures: bool = Field(default=True)
    filters: Optional[AuditQuery] = None
    compliance_standard: Optional[ComplianceStandard] = None


class AuditExportResponse(BaseModel):
    """Response for audit log export"""
    export_id: str
    format: AuditExportFormat
    download_url: str
    expires_at: datetime
    entry_count: int
    file_size_bytes: int
    checksum: str


class ComplianceCheck(BaseModel):
    """Individual compliance check result"""
    check_id: str
    name: str
    description: str
    standard: ComplianceStandard
    category: str
    status: str = Field(description="passed, failed, warning, not_applicable")
    findings: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)


class ComplianceReport(BaseModel):
    """Comprehensive compliance report"""
    id: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    period_start: datetime
    period_end: datetime
    
    # Standards covered
    standards: List[ComplianceStandard]
    
    # Overall status
    overall_status: str = Field(description="compliant, non_compliant, partial")
    compliance_score: float = Field(..., ge=0, le=100)
    
    # Checks
    total_checks: int
    passed_checks: int
    failed_checks: int
    warning_checks: int
    checks: List[ComplianceCheck] = Field(default_factory=list)
    
    # Findings summary
    critical_findings: int = Field(default=0, ge=0)
    high_findings: int = Field(default=0, ge=0)
    medium_findings: int = Field(default=0, ge=0)
    low_findings: int = Field(default=0, ge=0)
    
    # Audit trail summary
    total_audit_entries: int
    actions_breakdown: Dict[str, int] = Field(default_factory=dict)
    
    # Generated artifacts
    report_url: Optional[str] = None
    certificate_url: Optional[str] = None


class RetentionPolicy(BaseModel):
    """Data retention policy"""
    name: str
    description: str
    retention_days: int = Field(..., ge=1)
    data_types: List[str] = Field(default_factory=list)
    compliance_requirement: Optional[ComplianceStandard] = None
    auto_delete: bool = Field(default=False)
    archive_before_delete: bool = Field(default=True)


class DataAccessLog(BaseModel):
    """Log of data access for compliance"""
    id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    user_id: str
    data_type: str
    data_id: str
    access_type: str = Field(description="view, download, export, modify, delete")
    purpose: Optional[str] = None
    consent_reference: Optional[str] = None
    legal_basis: Optional[str] = None


class GDPRRequest(BaseModel):
    """GDPR data subject request"""
    id: str
    request_type: str = Field(description="access, rectification, erasure, portability, restriction, objection")
    subject_id: str
    subject_email: str
    submitted_at: datetime = Field(default_factory=datetime.utcnow)
    due_date: datetime
    status: str = Field(description="pending, in_progress, completed, rejected")
    completed_at: Optional[datetime] = None
    response_sent: bool = Field(default=False)
    notes: Optional[str] = None
