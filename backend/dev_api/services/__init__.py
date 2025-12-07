"""
Business Logic Services

Separates business logic from API routes for better testability and maintainability.

Services:
- code_review_service: Code review and analysis logic
- vulnerability_service: Vulnerability detection and management
- analytics_service: Data analytics and reporting

Usage:
    from dev_api.services import CodeReviewService, VulnerabilityService
    
    review_service = CodeReviewService()
    result = await review_service.analyze_code(code, language)
"""

from .code_review_service import CodeReviewService
from .vulnerability_service import VulnerabilityService
from .analytics_service import AnalyticsService

__all__ = [
    "CodeReviewService",
    "VulnerabilityService",
    "AnalyticsService",
]
