"""
Risk Management Module

Provides risk tracking and monitoring for the platform.
"""
from .risk_registry import (
    Risk,
    RiskCategory,
    RiskIndicator,
    RiskLevel,
    RiskRegistry,
    RiskStatus,
    Mitigation,
    MitigationStatus,
    get_risk_registry,
)
