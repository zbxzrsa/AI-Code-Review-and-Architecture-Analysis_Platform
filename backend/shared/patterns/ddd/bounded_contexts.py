"""
Bounded Contexts Definition

Defines clear boundaries between different parts of the domain.

Context Map:
┌────────────────────────────────────────────────────────────────────────┐
│                         AI Code Review Platform                        │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌─────────────────┐        ┌─────────────────┐                       │
│  │  Code Analysis  │◄──────►│Version Management│                       │
│  │    (Core)       │  U/D   │     (Core)      │                       │
│  └────────┬────────┘        └────────┬────────┘                       │
│           │                          │                                 │
│           │ ACL                      │ ACL                             │
│           ▼                          ▼                                 │
│  ┌─────────────────┐        ┌─────────────────┐                       │
│  │    User &       │        │    Provider     │                       │
│  │ Authentication  │◄──────►│   Management    │                       │
│  │  (Supporting)   │  CS    │  (Supporting)   │                       │
│  └────────┬────────┘        └────────┬────────┘                       │
│           │                          │                                 │
│           │ PL                       │ PL                              │
│           ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────┐                      │
│  │         Audit & Compliance (Generic)        │                      │
│  └─────────────────────────────────────────────┘                      │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘

Relationship Types:
- U/D: Upstream/Downstream
- ACL: Anti-Corruption Layer
- CS: Conformist
- PL: Published Language
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type
from uuid import uuid4

logger = logging.getLogger(__name__)


# =============================================================================
# Constants for Bounded Context Names
# =============================================================================

CONTEXT_CODE_ANALYSIS = "Code Analysis"
CONTEXT_VERSION_MANAGEMENT = "Version Management"
CONTEXT_USER_AUTHENTICATION = "User Authentication"
CONTEXT_PROVIDER_MANAGEMENT = "Provider Management"
CONTEXT_AUDIT_COMPLIANCE = "Audit & Compliance"


class SubdomainType(str, Enum):
    """Types of subdomains in DDD."""
    CORE = "core"           # Competitive advantage, highest priority
    SUPPORTING = "supporting"  # Necessary but not differentiating
    GENERIC = "generic"     # Common functionality, can be outsourced


class RelationshipType(str, Enum):
    """Types of relationships between bounded contexts."""
    UPSTREAM_DOWNSTREAM = "upstream_downstream"
    PARTNERSHIP = "partnership"
    SHARED_KERNEL = "shared_kernel"
    CUSTOMER_SUPPLIER = "customer_supplier"
    CONFORMIST = "conformist"
    ANTICORRUPTION_LAYER = "anticorruption_layer"
    OPEN_HOST_SERVICE = "open_host_service"
    PUBLISHED_LANGUAGE = "published_language"


@dataclass
class BoundedContext:
    """
    Represents a bounded context in the domain.

    A bounded context defines a clear boundary within which
    a particular domain model is defined and applicable.
    """
    name: str
    subdomain_type: SubdomainType
    description: str

    # Entities in this context
    entities: List[str] = field(default_factory=list)

    # Value objects
    value_objects: List[str] = field(default_factory=list)

    # Aggregates (root entities)
    aggregates: List[str] = field(default_factory=list)

    # Domain events published
    events_published: List[str] = field(default_factory=list)

    # Domain events subscribed to
    events_subscribed: List[str] = field(default_factory=list)

    # Services
    domain_services: List[str] = field(default_factory=list)
    application_services: List[str] = field(default_factory=list)

    # Repositories
    repositories: List[str] = field(default_factory=list)

    # Team ownership
    team: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "subdomain_type": self.subdomain_type.value,
            "description": self.description,
            "entities": self.entities,
            "value_objects": self.value_objects,
            "aggregates": self.aggregates,
            "events_published": self.events_published,
            "events_subscribed": self.events_subscribed,
            "domain_services": self.domain_services,
            "application_services": self.application_services,
            "repositories": self.repositories,
            "team": self.team,
        }


@dataclass
class ContextRelationship:
    """Relationship between two bounded contexts."""
    upstream_context: str
    downstream_context: str
    relationship_type: RelationshipType
    description: str
    integration_pattern: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "upstream": self.upstream_context,
            "downstream": self.downstream_context,
            "type": self.relationship_type.value,
            "description": self.description,
            "integration_pattern": self.integration_pattern,
        }


class ContextMap:
    """
    Maps all bounded contexts and their relationships.

    Provides a high-level view of the domain structure.
    """

    def __init__(self):
        self._contexts: Dict[str, BoundedContext] = {}
        self._relationships: List[ContextRelationship] = []

    def add_context(self, context: BoundedContext):
        """Add a bounded context."""
        self._contexts[context.name] = context
        logger.info(f"Added bounded context: {context.name}")

    def add_relationship(self, relationship: ContextRelationship):
        """Add a relationship between contexts."""
        self._relationships.append(relationship)
        logger.info(
            f"Added relationship: {relationship.upstream_context} -> "
            f"{relationship.downstream_context} ({relationship.relationship_type.value})"
        )

    def get_context(self, name: str) -> Optional[BoundedContext]:
        """Get a bounded context by name."""
        return self._contexts.get(name)

    def get_contexts_by_type(self, subdomain_type: SubdomainType) -> List[BoundedContext]:
        """Get all contexts of a specific subdomain type."""
        return [
            ctx for ctx in self._contexts.values()
            if ctx.subdomain_type == subdomain_type
        ]

    def get_upstream_contexts(self, context_name: str) -> List[str]:
        """Get all upstream contexts for a given context."""
        return [
            rel.upstream_context
            for rel in self._relationships
            if rel.downstream_context == context_name
        ]

    def get_downstream_contexts(self, context_name: str) -> List[str]:
        """Get all downstream contexts for a given context."""
        return [
            rel.downstream_context
            for rel in self._relationships
            if rel.upstream_context == context_name
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Export context map as dictionary."""
        return {
            "contexts": {
                name: ctx.to_dict()
                for name, ctx in self._contexts.items()
            },
            "relationships": [rel.to_dict() for rel in self._relationships],
        }

    def generate_plantuml(self) -> str:
        """Generate PlantUML diagram for context map."""
        lines = [
            "@startuml",
            "!define CORE #LightBlue",
            "!define SUPPORTING #LightGreen",
            "!define GENERIC #LightGray",
            "",
            "title AI Code Review Platform - Context Map",
            "",
        ]

        # Add contexts
        for name, ctx in self._contexts.items():
            color = {
                SubdomainType.CORE: "CORE",
                SubdomainType.SUPPORTING: "SUPPORTING",
                SubdomainType.GENERIC: "GENERIC",
            }[ctx.subdomain_type]

            lines.append(f'rectangle "{name}\\n({ctx.subdomain_type.value})" as {name.replace(" ", "_")} {color}')

        lines.append("")

        # Add relationships
        for rel in self._relationships:
            upstream = rel.upstream_context.replace(" ", "_")
            downstream = rel.downstream_context.replace(" ", "_")
            label = rel.relationship_type.value.replace("_", " ")
            lines.append(f'{upstream} --> {downstream} : {label}')

        lines.append("")
        lines.append("@enduml")

        return "\n".join(lines)


# =============================================================================
# Pre-defined Bounded Contexts for AI Code Review Platform
# =============================================================================

def create_platform_context_map() -> ContextMap:
    """Create the context map for the AI Code Review Platform."""
    ctx_map = ContextMap()

    # Core Domain: Code Analysis
    code_analysis = BoundedContext(
        name=CONTEXT_CODE_ANALYSIS,
        subdomain_type=SubdomainType.CORE,
        description="Core code analysis and review functionality using AI models",
        entities=[
            "Analysis", "Issue", "Fix", "CodeSnippet", "ReviewResult"
        ],
        value_objects=[
            "CodeHash", "Severity", "IssueType", "FixSuggestion", "AnalysisMetrics"
        ],
        aggregates=["Analysis"],
        events_published=[
            "AnalysisCreated", "AnalysisCompleted", "IssueDetected",
            "FixApplied", "AnalysisFailed"
        ],
        events_subscribed=[
            "ExperimentCompleted", "ModelPromoted"
        ],
        domain_services=["CodeAnalyzer", "IssueDetector", "FixGenerator"],
        application_services=["AnalysisService", "ReviewService"],
        repositories=["AnalysisRepository", "IssueRepository"],
        team="Core Platform Team"
    )
    ctx_map.add_context(code_analysis)

    # Core Domain: Version Management
    version_management = BoundedContext(
        name=CONTEXT_VERSION_MANAGEMENT,
        subdomain_type=SubdomainType.CORE,
        description="Three-version cycle management (V1/V2/V3) for AI model evolution",
        entities=[
            "Experiment", "Version", "Promotion", "Degradation", "Evaluation"
        ],
        value_objects=[
            "VersionId", "ExperimentMetrics", "PromotionCriteria",
            "ModelConfiguration", "EvaluationResult"
        ],
        aggregates=["Experiment", "Version"],
        events_published=[
            "ExperimentCreated", "ExperimentCompleted", "VersionPromoted",
            "VersionDemoted", "ModelQuarantined", "ReEvaluationRequested"
        ],
        events_subscribed=[
            "AnalysisCompleted", "ProviderHealthChanged"
        ],
        domain_services=[
            "VersionController", "ExperimentRunner", "PromotionDecider"
        ],
        application_services=["VersionControlService", "ExperimentService"],
        repositories=["ExperimentRepository", "VersionRepository"],
        team="AI Platform Team"
    )
    ctx_map.add_context(version_management)

    # Supporting Domain: User & Authentication
    user_auth = BoundedContext(
        name=CONTEXT_USER_AUTHENTICATION,
        subdomain_type=SubdomainType.SUPPORTING,
        description="User management, authentication, and authorization",
        entities=[
            "User", "Session", "Role", "Permission", "Invitation"
        ],
        value_objects=[
            "Email", "Password", "Token", "RoleType"
        ],
        aggregates=["User"],
        events_published=[
            "UserCreated", "UserLoggedIn", "UserLoggedOut",
            "PasswordChanged", "RoleAssigned"
        ],
        events_subscribed=[],
        domain_services=["Authenticator", "Authorizer", "TokenManager"],
        application_services=["AuthService", "UserService"],
        repositories=["UserRepository", "SessionRepository"],
        team="Platform Security Team"
    )
    ctx_map.add_context(user_auth)

    # Supporting Domain: Provider Management
    provider_management = BoundedContext(
        name=CONTEXT_PROVIDER_MANAGEMENT,
        subdomain_type=SubdomainType.SUPPORTING,
        description="AI provider configuration, health monitoring, and quota management",
        entities=[
            "Provider", "UserProvider", "Quota", "Usage", "HealthStatus"
        ],
        value_objects=[
            "APIKey", "Endpoint", "RateLimit", "CostLimit"
        ],
        aggregates=["Provider"],
        events_published=[
            "ProviderRegistered", "ProviderHealthChanged",
            "QuotaExceeded", "CostAlertTriggered"
        ],
        events_subscribed=[
            "AnalysisCompleted"
        ],
        domain_services=[
            "HealthChecker", "QuotaEnforcer", "CostTracker"
        ],
        application_services=["ProviderService"],
        repositories=["ProviderRepository", "QuotaRepository"],
        team="Infrastructure Team"
    )
    ctx_map.add_context(provider_management)

    # Generic Domain: Audit & Compliance
    audit_compliance = BoundedContext(
        name="Audit Compliance",
        subdomain_type=SubdomainType.GENERIC,
        description="Audit logging, compliance reporting, and security monitoring",
        entities=[
            "AuditLog", "ComplianceReport", "SecurityAlert"
        ],
        value_objects=[
            "LogEntry", "Signature", "Hash", "ChainLink"
        ],
        aggregates=["AuditLog"],
        events_published=[
            "AuditEntryCreated", "IntegrityViolation", "ComplianceReportGenerated"
        ],
        events_subscribed=[
            "UserLoggedIn", "AnalysisCompleted", "VersionPromoted",
            "ProviderHealthChanged"
        ],
        domain_services=[
            "AuditLogger", "IntegrityVerifier", "ComplianceChecker"
        ],
        application_services=["AuditService", "ComplianceService"],
        repositories=["AuditLogRepository"],
        team="Security & Compliance Team"
    )
    ctx_map.add_context(audit_compliance)

    # Define Relationships

    # Code Analysis <-> Version Management (Partnership)
    ctx_map.add_relationship(ContextRelationship(
        upstream_context="Version Management",
        downstream_context="Code Analysis",
        relationship_type=RelationshipType.UPSTREAM_DOWNSTREAM,
        description="Version Management provides model configuration to Code Analysis",
        integration_pattern="Domain Events + Shared Kernel"
    ))

    # User Auth -> Code Analysis (ACL)
    ctx_map.add_relationship(ContextRelationship(
        upstream_context="User Authentication",
        downstream_context="Code Analysis",
        relationship_type=RelationshipType.ANTICORRUPTION_LAYER,
        description="Code Analysis uses Auth through anti-corruption layer",
        integration_pattern="ACL with User context translation"
    ))

    # User Auth -> Version Management (ACL)
    ctx_map.add_relationship(ContextRelationship(
        upstream_context="User Authentication",
        downstream_context="Version Management",
        relationship_type=RelationshipType.ANTICORRUPTION_LAYER,
        description="Version Management uses Auth for admin access control",
        integration_pattern="ACL with Role-based access"
    ))

    # Provider Management <-> Code Analysis (Customer-Supplier)
    ctx_map.add_relationship(ContextRelationship(
        upstream_context="Provider Management",
        downstream_context="Code Analysis",
        relationship_type=RelationshipType.CUSTOMER_SUPPLIER,
        description="Code Analysis is customer of Provider Management",
        integration_pattern="Provider interface abstraction"
    ))

    # Provider Management <-> Version Management (Conformist)
    ctx_map.add_relationship(ContextRelationship(
        upstream_context="Provider Management",
        downstream_context="Version Management",
        relationship_type=RelationshipType.CONFORMIST,
        description="Version Management conforms to Provider health model",
        integration_pattern="Health status events"
    ))

    # All -> Audit (Published Language)
    for ctx_name in ["Code Analysis", "Version Management", "User Authentication", "Provider Management"]:
        ctx_map.add_relationship(ContextRelationship(
            upstream_context=ctx_name,
            downstream_context="Audit Compliance",
            relationship_type=RelationshipType.PUBLISHED_LANGUAGE,
            description=f"{ctx_name} publishes events to Audit using standard schema",
            integration_pattern="Domain Events with standard audit schema"
        ))

    return ctx_map


# =============================================================================
# Anti-Corruption Layer Implementation
# =============================================================================

class AntiCorruptionLayer(ABC):
    """
    Base class for anti-corruption layers.

    Translates between different bounded contexts while
    protecting the domain model from external influence.
    """

    @abstractmethod
    def translate_inbound(self, external_data: Dict[str, Any]) -> Any:
        """Translate external data to domain model."""
        pass

    @abstractmethod
    def translate_outbound(self, domain_object: Any) -> Dict[str, Any]:
        """Translate domain object to external format."""
        pass


class UserContextACL(AntiCorruptionLayer):
    """
    ACL for translating User Authentication context to other contexts.
    """

    def translate_inbound(self, external_data: Dict[str, Any]) -> 'ActorInfo':
        """Translate user data to actor info used in other contexts."""
        return ActorInfo(
            actor_id=external_data.get("user_id", ""),
            actor_type="user",
            permissions=external_data.get("permissions", []),
            roles=external_data.get("roles", []),
        )

    def translate_outbound(self, domain_object: Any) -> Dict[str, Any]:
        """Translate domain action to audit format."""
        return {
            "user_id": getattr(domain_object, "actor_id", "unknown"),
            "action": getattr(domain_object, "action", "unknown"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class ActorInfo:
    """Actor information used across contexts."""
    actor_id: str
    actor_type: str
    permissions: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)


class ProviderContextACL(AntiCorruptionLayer):
    """
    ACL for translating Provider Management context to Code Analysis.
    """

    def translate_inbound(self, external_data: Dict[str, Any]) -> 'ProviderEndpoint':
        """Translate provider data to endpoint info."""
        return ProviderEndpoint(
            provider_id=external_data.get("provider_id", ""),
            endpoint_url=external_data.get("endpoint", ""),
            is_available=external_data.get("status") == "healthy",
            latency_ms=external_data.get("avg_latency_ms", 0),
        )

    def translate_outbound(self, domain_object: Any) -> Dict[str, Any]:
        """Translate analysis result to usage tracking format."""
        return {
            "provider_id": getattr(domain_object, "provider_id", ""),
            "tokens_used": getattr(domain_object, "tokens_used", 0),
            "duration_ms": getattr(domain_object, "duration_ms", 0),
            "success": getattr(domain_object, "success", True),
        }


@dataclass
class ProviderEndpoint:
    """Provider endpoint information used in Code Analysis context."""
    provider_id: str
    endpoint_url: str
    is_available: bool
    latency_ms: float


# Global context map instance
_context_map: Optional[ContextMap] = None


def get_context_map() -> ContextMap:
    """Get or create the global context map."""
    global _context_map
    if _context_map is None:
        _context_map = create_platform_context_map()
    return _context_map
