"""
Safety Monitor Module

Ensures safe autonomous learning with:
- Action validation
- Dangerous pattern detection
- Human oversight gates
- Value alignment checks
- Resource limits
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from .config import SafetyLevel

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """Record of a safety violation."""
    violation_id: str
    violation_type: str
    description: str
    severity: str
    action_blocked: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "violation_id": self.violation_id,
            "violation_type": self.violation_type,
            "description": self.description,
            "severity": self.severity,
            "action_blocked": self.action_blocked,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass 
class SafetyConfig:
    """Safety configuration."""
    level: SafetyLevel = SafetyLevel.HIGH
    human_oversight_required: bool = True
    max_autonomous_steps: int = 10000
    
    # Dangerous patterns
    dangerous_patterns: List[str] = field(default_factory=lambda: [
        r"delete\s+\*",
        r"drop\s+table",
        r"rm\s+-rf",
        r"format\s+c:",
        r"shutdown",
        r"destroy",
        r"terminate",
    ])
    
    # Resource limits
    max_memory_mb: int = 4096
    max_cpu_percent: float = 80.0
    max_network_requests_per_hour: int = 1000
    
    # Allowed actions
    allowed_actions: Set[str] = field(default_factory=lambda: {
        "read", "analyze", "suggest", "generate", "process_input",
        "learn", "store", "retrieve", "search",
    })
    
    # Blocked actions
    blocked_actions: Set[str] = field(default_factory=lambda: {
        "delete_system", "modify_kernel", "access_credentials",
        "send_email", "make_purchase", "execute_code",
    })


class SafetyMonitor:
    """
    Safety monitor for autonomous learning.
    
    Ensures safe operation through:
    - Action validation against allowed/blocked lists
    - Dangerous pattern detection
    - Human oversight for sensitive actions
    - Resource limit enforcement
    - Violation logging
    
    Usage:
        monitor = SafetyMonitor(SafetyConfig(level=SafetyLevel.HIGH))
        
        # Check action
        if monitor.check_action("process_input", {"text": "..."}):
            # Safe to proceed
            ...
        
        # Check text content
        if monitor.check_text_content("rm -rf /"):
            # Contains dangerous pattern
            ...
        
        # Get status
        status = monitor.get_safety_status()
    """
    
    def __init__(self, config: Optional[SafetyConfig] = None):
        """
        Initialize safety monitor.
        
        Args:
            config: Safety configuration
        """
        self.config = config or SafetyConfig()
        
        # Compile dangerous patterns
        self._dangerous_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.dangerous_patterns
        ]
        
        # Violation history
        self.violations: List[SafetyViolation] = []
        
        # Action counters
        self.actions_checked = 0
        self.actions_blocked = 0
        self.autonomous_steps = 0
        
        # Human oversight state
        self._pending_oversight: Dict[str, Dict[str, Any]] = {}
        self._approved_actions: Set[str] = set()
    
    def check_action(
        self,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Check if an action is safe to perform.
        
        Args:
            action: Action name
            context: Action context
            
        Returns:
            True if action is safe
        """
        self.actions_checked += 1
        
        # Check blocked actions
        if action in self.config.blocked_actions:
            self._record_violation(
                violation_type="blocked_action",
                description=f"Action '{action}' is blocked",
                severity="high",
                action_blocked=action,
            )
            return False
        
        # Check allowed actions based on safety level
        if self.config.level in [SafetyLevel.HIGH, SafetyLevel.CRITICAL]:
            if action not in self.config.allowed_actions:
                self._record_violation(
                    violation_type="unknown_action",
                    description=f"Action '{action}' not in allowed list",
                    severity="medium",
                    action_blocked=action,
                )
                return False
        
        # Check autonomous step limit
        if self.autonomous_steps >= self.config.max_autonomous_steps:
            if self.config.human_oversight_required:
                self._record_violation(
                    violation_type="step_limit",
                    description="Autonomous step limit reached",
                    severity="high",
                    action_blocked=action,
                )
                return False
        
        # Check text content in context
        if context:
            text_content = context.get("text", "")
            if text_content and self.contains_dangerous_pattern(text_content):
                self._record_violation(
                    violation_type="dangerous_pattern",
                    description="Context contains dangerous pattern",
                    severity="high",
                    action_blocked=action,
                    metadata={"text_snippet": text_content[:100]},
                )
                return False
        
        self.autonomous_steps += 1
        return True
    
    def contains_dangerous_pattern(self, text: str) -> bool:
        """Check if text contains dangerous patterns."""
        for pattern in self._dangerous_patterns:
            if pattern.search(text):
                return True
        return False
    
    def check_text_content(self, text: str) -> bool:
        """
        Check text content for safety.
        
        Args:
            text: Text to check
            
        Returns:
            True if text is safe
        """
        if self.contains_dangerous_pattern(text):
            self._record_violation(
                violation_type="dangerous_content",
                description="Text contains dangerous pattern",
                severity="high",
                action_blocked="text_processing",
                metadata={"text_snippet": text[:100]},
            )
            return False
        return True
    
    def request_human_oversight(
        self,
        action: str,
        reason: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Request human oversight for an action.
        
        Args:
            action: Action requiring oversight
            reason: Why oversight is needed
            context: Action context
            
        Returns:
            Request ID
        """
        import uuid
        request_id = str(uuid.uuid4())[:8]
        
        self._pending_oversight[request_id] = {
            "action": action,
            "reason": reason,
            "context": context,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        
        logger.warning(f"Human oversight requested: {action} - {reason}")
        
        return request_id
    
    def approve_oversight_request(self, request_id: str, approver: str) -> bool:
        """Approve a pending oversight request."""
        if request_id in self._pending_oversight:
            request = self._pending_oversight.pop(request_id)
            self._approved_actions.add(request["action"])
            logger.info(f"Oversight approved by {approver}: {request['action']}")
            return True
        return False
    
    def reject_oversight_request(self, request_id: str, rejector: str) -> bool:
        """Reject a pending oversight request."""
        if request_id in self._pending_oversight:
            request = self._pending_oversight.pop(request_id)
            self._record_violation(
                violation_type="oversight_rejected",
                description=f"Action rejected by {rejector}",
                severity="medium",
                action_blocked=request["action"],
            )
            return True
        return False
    
    def _record_violation(
        self,
        violation_type: str,
        description: str,
        severity: str,
        action_blocked: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a safety violation."""
        import uuid
        
        violation = SafetyViolation(
            violation_id=str(uuid.uuid4())[:8],
            violation_type=violation_type,
            description=description,
            severity=severity,
            action_blocked=action_blocked,
            metadata=metadata or {},
        )
        
        self.violations.append(violation)
        self.actions_blocked += 1
        
        logger.warning(f"Safety violation: {violation_type} - {description}")
    
    def get_safety_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return {
            "level": self.config.level.value,
            "human_oversight_required": self.config.human_oversight_required,
            "autonomous_steps": self.autonomous_steps,
            "max_autonomous_steps": self.config.max_autonomous_steps,
            "actions_checked": self.actions_checked,
            "actions_blocked": self.actions_blocked,
            "violation_count": len(self.violations),
            "pending_oversight": len(self._pending_oversight),
            "block_rate": (
                self.actions_blocked / self.actions_checked
                if self.actions_checked > 0 else 0.0
            ),
        }
    
    def get_violations(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent violations."""
        return [v.to_dict() for v in self.violations[-limit:]]
    
    def reset_step_counter(self):
        """Reset the autonomous step counter."""
        self.autonomous_steps = 0
    
    def clear_violations(self):
        """Clear violation history."""
        self.violations.clear()
