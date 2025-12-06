"""
Domain-Driven Design (DDD) Implementation

Provides clear domain boundaries and bounded contexts for the platform.

Sub-domains:
- Code Analysis (Core)
- Version Management (Core)
- User & Auth (Supporting)
- Provider Management (Supporting)
- Audit & Compliance (Generic)
"""
from .bounded_contexts import *
from .domain_models import *
from .aggregates import *
from .domain_events import *
from .repositories import *
