"""
Cost monitoring and control module.

Tracks:
- Token usage (input/output)
- API costs
- Compute costs
- Resource utilization
"""

import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .config import PracticalDeploymentConfig

logger = logging.getLogger(__name__)


class CostController:
    """
    Monitors and controls costs.
    
    Tracks:
    - Token usage (daily/monthly)
    - API costs
    - Compute costs
    - Resource utilization
    """
    
    def __init__(self, config: PracticalDeploymentConfig):
        self.config = config
        
        # Current period counters
        self.daily_tokens = 0
        self.daily_input_tokens = 0
        self.daily_output_tokens = 0
        self.monthly_cost = 0.0
        
        # Period tracking
        self.current_date = datetime.now(timezone.utc).date()
        self.current_month = datetime.now(timezone.utc).month
        
        # History
        self.token_history: List[Dict[str, Any]] = []
        self.cost_history: List[Dict[str, Any]] = []
        self.hourly_usage: Dict[int, int] = defaultdict(int)
        
        # Cost per operation
        self.cost_per_1k_tokens = {
            "input": 0.01,   # $0.01 per 1K input tokens
            "output": 0.03,  # $0.03 per 1K output tokens
        }
        
        self._lock = threading.Lock()
    
    def _check_and_reset_periods(self):
        """Check and reset daily/monthly counters if needed."""
        now = datetime.now(timezone.utc)
        
        # Reset daily counters
        if now.date() != self.current_date:
            logger.info(f"Resetting daily counters. Previous: {self.daily_tokens} tokens")
            self.daily_tokens = 0
            self.daily_input_tokens = 0
            self.daily_output_tokens = 0
            self.hourly_usage.clear()
            self.current_date = now.date()
        
        # Reset monthly counters
        if now.month != self.current_month:
            logger.info(f"Resetting monthly counters. Previous cost: ${self.monthly_cost:.2f}")
            self.monthly_cost = 0.0
            self.current_month = now.month
    
    def record_usage(
        self,
        input_tokens: int = 0,
        output_tokens: int = 0,
        operation: str = "inference",
        model: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Record token usage and calculate cost.
        
        Returns:
            Tuple of (within_limits, warning_message)
        """
        with self._lock:
            self._check_and_reset_periods()
            
            total_tokens = input_tokens + output_tokens
            
            # Calculate cost
            input_cost = (input_tokens / 1000) * self.cost_per_1k_tokens["input"]
            output_cost = (output_tokens / 1000) * self.cost_per_1k_tokens["output"]
            total_cost = input_cost + output_cost
            
            # Update counters
            self.daily_tokens += total_tokens
            self.daily_input_tokens += input_tokens
            self.daily_output_tokens += output_tokens
            self.monthly_cost += total_cost
            
            # Track hourly usage
            current_hour = datetime.now(timezone.utc).hour
            self.hourly_usage[current_hour] += total_tokens
            
            # Record history
            self.token_history.append({
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost": total_cost,
                "operation": operation,
                "model": model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            
            # Trim history
            if len(self.token_history) > 10000:
                self.token_history = self.token_history[-5000:]
        
        # Check limits
        return self.check_limits()
    
    def check_limits(self) -> Tuple[bool, Optional[str]]:
        """
        Check if within usage limits.
        
        Returns:
            Tuple of (within_limits, warning_message)
        """
        warnings = []
        
        # Check daily token limit
        if self.daily_tokens > self.config.max_daily_tokens:
            return False, f"Daily token limit exceeded: {self.daily_tokens:,}/{self.config.max_daily_tokens:,}"
        
        # Check monthly cost limit
        if self.monthly_cost > self.config.max_monthly_cost_usd:
            return False, f"Monthly cost limit exceeded: ${self.monthly_cost:.2f}/${self.config.max_monthly_cost_usd:.2f}"
        
        # Warnings at 80% and 90%
        daily_percent = self.daily_tokens / self.config.max_daily_tokens * 100
        monthly_percent = self.monthly_cost / self.config.max_monthly_cost_usd * 100
        
        if daily_percent >= 90:
            warnings.append(f"Daily tokens at {daily_percent:.1f}%")
        elif daily_percent >= 80:
            warnings.append(f"Daily tokens at {daily_percent:.1f}%")
        
        if monthly_percent >= 90:
            warnings.append(f"Monthly cost at {monthly_percent:.1f}%")
        elif monthly_percent >= 80:
            warnings.append(f"Monthly cost at {monthly_percent:.1f}%")
        
        if warnings:
            return True, "; ".join(warnings)
        
        return True, None
    
    def estimate_remaining_capacity(self) -> Dict[str, Any]:
        """Estimate remaining capacity for the current period."""
        with self._lock:
            self._check_and_reset_periods()
            
            daily_remaining = max(0, self.config.max_daily_tokens - self.daily_tokens)
            monthly_remaining = max(0, self.config.max_monthly_cost_usd - self.monthly_cost)
            
            # Estimate tokens remaining based on cost
            avg_cost_per_token = sum(self.cost_per_1k_tokens.values()) / 2 / 1000
            monthly_tokens_remaining = int(monthly_remaining / avg_cost_per_token) if avg_cost_per_token > 0 else 0
            
            return {
                "daily_tokens_remaining": daily_remaining,
                "daily_tokens_used": self.daily_tokens,
                "daily_percent_used": self.daily_tokens / self.config.max_daily_tokens * 100,
                "monthly_cost_remaining": monthly_remaining,
                "monthly_cost_used": self.monthly_cost,
                "monthly_percent_used": self.monthly_cost / self.config.max_monthly_cost_usd * 100,
                "estimated_monthly_tokens_remaining": monthly_tokens_remaining,
            }
    
    def set_cost_rates(
        self,
        input_per_1k: float,
        output_per_1k: float,
    ):
        """Update cost rates."""
        with self._lock:
            self.cost_per_1k_tokens["input"] = input_per_1k
            self.cost_per_1k_tokens["output"] = output_per_1k
        
        logger.info(f"Updated cost rates: input=${input_per_1k}/1K, output=${output_per_1k}/1K")
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary."""
        with self._lock:
            self._check_and_reset_periods()
            
            # Calculate averages
            if self.token_history:
                recent = self.token_history[-100:]
                avg_tokens = sum(h["input_tokens"] + h["output_tokens"] for h in recent) / len(recent)
                avg_cost = sum(h["cost"] for h in recent) / len(recent)
            else:
                avg_tokens = 0
                avg_cost = 0
            
            return {
                "daily": {
                    "tokens_used": self.daily_tokens,
                    "tokens_limit": self.config.max_daily_tokens,
                    "usage_percent": self.daily_tokens / self.config.max_daily_tokens * 100,
                    "input_tokens": self.daily_input_tokens,
                    "output_tokens": self.daily_output_tokens,
                },
                "monthly": {
                    "cost_used": self.monthly_cost,
                    "cost_limit": self.config.max_monthly_cost_usd,
                    "usage_percent": self.monthly_cost / self.config.max_monthly_cost_usd * 100,
                },
                "averages": {
                    "tokens_per_request": avg_tokens,
                    "cost_per_request": avg_cost,
                },
                "hourly_distribution": dict(self.hourly_usage),
                "cost_rates": self.cost_per_1k_tokens.copy(),
            }
    
    def get_cost_breakdown(self, days: int = 7) -> Dict[str, Any]:
        """Get cost breakdown for the last N days."""
        from collections import defaultdict
        
        with self._lock:
            breakdown = defaultdict(lambda: {"tokens": 0, "cost": 0.0, "requests": 0})
            
            for entry in self.token_history:
                # Parse date
                timestamp = entry.get("timestamp", "")
                if timestamp:
                    date = timestamp[:10]
                    breakdown[date]["tokens"] += entry.get("input_tokens", 0) + entry.get("output_tokens", 0)
                    breakdown[date]["cost"] += entry.get("cost", 0)
                    breakdown[date]["requests"] += 1
            
            # Get last N days
            sorted_dates = sorted(breakdown.keys(), reverse=True)[:days]
            
            return {
                "by_date": {d: breakdown[d] for d in sorted_dates},
                "total_cost": sum(breakdown[d]["cost"] for d in sorted_dates),
                "total_tokens": sum(breakdown[d]["tokens"] for d in sorted_dates),
                "total_requests": sum(breakdown[d]["requests"] for d in sorted_dates),
            }
