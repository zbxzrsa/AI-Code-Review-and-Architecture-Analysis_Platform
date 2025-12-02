"""
Failure logging and V3 integration module.

Handles:
- Failure detection and documentation
- Automatic V3 quarantine integration
- Root cause analysis tracking
- Experiment blacklisting
"""

from .logger import (
    FailureLogger,
    FailureRecord,
    FailureType,
    BlockingLevel,
    FailureTrigger,
)

__all__ = [
    "FailureLogger",
    "FailureRecord",
    "FailureType",
    "BlockingLevel",
    "FailureTrigger",
]
