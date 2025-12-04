"""
Test Fixtures Package

Provides reusable test data and sample code for testing.
"""

from .sample_code import (
    SQL_INJECTION_VULNERABLE,
    SQL_INJECTION_SAFE,
    XSS_VULNERABLE,
    XSS_SAFE,
    COMMAND_INJECTION_VULNERABLE,
    COMMAND_INJECTION_SAFE,
    HARDCODED_SECRET_VULNERABLE,
    HARDCODED_SECRET_SAFE,
    PATH_TRAVERSAL_VULNERABLE,
    PATH_TRAVERSAL_SAFE,
    CLEAN_CODE_PYTHON,
    CLEAN_CODE_JAVASCRIPT,
    CLEAN_CODE_TYPESCRIPT,
    SAMPLES_BY_LANGUAGE,
    get_sample,
    get_all_vulnerable_samples,
    get_all_safe_samples,
)

__all__ = [
    "SQL_INJECTION_VULNERABLE",
    "SQL_INJECTION_SAFE",
    "XSS_VULNERABLE",
    "XSS_SAFE",
    "COMMAND_INJECTION_VULNERABLE",
    "COMMAND_INJECTION_SAFE",
    "HARDCODED_SECRET_VULNERABLE",
    "HARDCODED_SECRET_SAFE",
    "PATH_TRAVERSAL_VULNERABLE",
    "PATH_TRAVERSAL_SAFE",
    "CLEAN_CODE_PYTHON",
    "CLEAN_CODE_JAVASCRIPT",
    "CLEAN_CODE_TYPESCRIPT",
    "SAMPLES_BY_LANGUAGE",
    "get_sample",
    "get_all_vulnerable_samples",
    "get_all_safe_samples",
]
