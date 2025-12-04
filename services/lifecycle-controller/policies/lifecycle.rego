# OPA Policy - Lifecycle Controller Decisions
# Policy-as-code for promotion, gray-scale, and downgrade decisions

package lifecycle

import future.keywords.if
import future.keywords.in

# ==================== Promotion Policy ====================
# Determines if a version can be promoted from V1 shadow to V2 gray-scale

promotion := result if {
    result := {
        "allow": allow_promotion,
        "downgrade": should_downgrade,
        "reason": promotion_reason,
        "details": promotion_details,
    }
}

# Allow promotion if all conditions met
allow_promotion if {
    latency_ok
    error_rate_ok
    accuracy_ok
    security_ok
    cost_ok
    statistical_significance_ok
    sufficient_data
}

# Check latency threshold
latency_ok if {
    input.metrics.p95_latency_ms <= input.thresholds.p95_latency_ms
}

# Check error rate threshold
error_rate_ok if {
    input.metrics.error_rate <= input.thresholds.error_rate
}

# Check accuracy improvement (must be significantly better)
accuracy_ok if {
    input.metrics.accuracy_delta >= input.thresholds.accuracy_delta
}

# Check security pass rate
security_ok if {
    input.metrics.security_pass_rate >= input.thresholds.security_pass_rate
}

# Check cost increase threshold
cost_ok if {
    input.metrics.cost_delta <= input.thresholds.cost_increase_max
}

# Check statistical significance for accuracy
statistical_significance_ok if {
    input.statistical_tests.accuracy_p_value < input.thresholds.statistical_significance_p
}

# Ensure sufficient data for decision
sufficient_data if {
    input.metrics.total_requests >= 1000
}

# Determine if should downgrade
should_downgrade if {
    # Security failure always triggers downgrade
    input.metrics.security_pass_rate < 0.95
}

should_downgrade if {
    # Severe performance degradation
    input.metrics.p95_latency_ms > input.thresholds.p95_latency_ms * 2
}

should_downgrade if {
    # High error rate
    input.metrics.error_rate > input.thresholds.error_rate * 3
}

should_downgrade if {
    # Significant accuracy regression (worse than baseline)
    input.metrics.accuracy_delta < -0.05
}

default should_downgrade := false

# Generate reason for decision
promotion_reason := "All thresholds met - promotion approved" if {
    allow_promotion
}

promotion_reason := concat("; ", reasons) if {
    not allow_promotion
    not should_downgrade
    reasons := [r | r := failed_checks[_]]
}

promotion_reason := concat("; ", reasons) if {
    should_downgrade
    reasons := [r | r := downgrade_reasons[_]]
}

# Collect failed checks
failed_checks[msg] {
    not latency_ok
    msg := sprintf("p95 latency %.0fms exceeds threshold %.0fms", 
        [input.metrics.p95_latency_ms, input.thresholds.p95_latency_ms])
}

failed_checks[msg] {
    not error_rate_ok
    msg := sprintf("error rate %.2f%% exceeds threshold %.2f%%",
        [input.metrics.error_rate * 100, input.thresholds.error_rate * 100])
}

failed_checks[msg] {
    not accuracy_ok
    msg := sprintf("accuracy delta %.2f%% below threshold %.2f%%",
        [input.metrics.accuracy_delta * 100, input.thresholds.accuracy_delta * 100])
}

failed_checks[msg] {
    not security_ok
    msg := sprintf("security pass rate %.2f%% below threshold %.2f%%",
        [input.metrics.security_pass_rate * 100, input.thresholds.security_pass_rate * 100])
}

failed_checks[msg] {
    not cost_ok
    msg := sprintf("cost increase %.2f%% exceeds threshold %.2f%%",
        [input.metrics.cost_delta * 100, input.thresholds.cost_increase_max * 100])
}

failed_checks[msg] {
    not statistical_significance_ok
    msg := sprintf("accuracy p-value %.3f not significant (threshold %.3f)",
        [input.statistical_tests.accuracy_p_value, input.thresholds.statistical_significance_p])
}

failed_checks[msg] {
    not sufficient_data
    msg := sprintf("insufficient requests: %d (minimum 1000)", [input.metrics.total_requests])
}

# Collect downgrade reasons
downgrade_reasons[msg] {
    input.metrics.security_pass_rate < 0.95
    msg := sprintf("CRITICAL: Security pass rate %.2f%% is dangerously low",
        [input.metrics.security_pass_rate * 100])
}

downgrade_reasons[msg] {
    input.metrics.p95_latency_ms > input.thresholds.p95_latency_ms * 2
    msg := sprintf("CRITICAL: p95 latency %.0fms is 2x threshold",
        [input.metrics.p95_latency_ms])
}

downgrade_reasons[msg] {
    input.metrics.error_rate > input.thresholds.error_rate * 3
    msg := sprintf("CRITICAL: error rate %.2f%% is 3x threshold",
        [input.metrics.error_rate * 100])
}

downgrade_reasons[msg] {
    input.metrics.accuracy_delta < -0.05
    msg := sprintf("CRITICAL: accuracy regression %.2f%% vs baseline",
        [input.metrics.accuracy_delta * 100])
}

# Detailed breakdown
promotion_details := {
    "latency": {"ok": latency_ok, "value": input.metrics.p95_latency_ms, "threshold": input.thresholds.p95_latency_ms},
    "error_rate": {"ok": error_rate_ok, "value": input.metrics.error_rate, "threshold": input.thresholds.error_rate},
    "accuracy": {"ok": accuracy_ok, "value": input.metrics.accuracy_delta, "threshold": input.thresholds.accuracy_delta},
    "security": {"ok": security_ok, "value": input.metrics.security_pass_rate, "threshold": input.thresholds.security_pass_rate},
    "cost": {"ok": cost_ok, "value": input.metrics.cost_delta, "threshold": input.thresholds.cost_increase_max},
    "significance": {"ok": statistical_significance_ok, "p_value": input.statistical_tests.accuracy_p_value},
    "data_sufficient": sufficient_data,
}


# ==================== Gray-Scale Progress Policy ====================
# Determines if gray-scale should advance, hold, or rollback

gray_progress := result if {
    result := {
        "advance": should_advance,
        "rollback": should_rollback,
        "hold": should_hold,
        "reason": gray_reason,
    }
}

# Advance if SLOs are healthy
should_advance if {
    gray_latency_ok
    gray_error_rate_ok
    gray_accuracy_ok
    input.version.consecutive_failures == 0
}

# Rollback on critical failures
should_rollback if {
    input.version.consecutive_failures >= 3
}

should_rollback if {
    input.metrics.error_rate > 0.10  # 10% error rate
}

should_rollback if {
    input.metrics.p95_latency_ms > input.thresholds.p95_latency_ms * 3
}

default should_rollback := false

# Hold if not advancing and not rolling back
should_hold if {
    not should_advance
    not should_rollback
}

default should_hold := false

# Gray-scale specific checks (more lenient during rollout)
gray_latency_ok if {
    input.metrics.p95_latency_ms <= input.thresholds.p95_latency_ms * 1.2  # 20% tolerance
}

gray_error_rate_ok if {
    input.metrics.error_rate <= input.thresholds.error_rate * 1.5  # 50% tolerance
}

gray_accuracy_ok if {
    input.metrics.accuracy >= 0.85  # Absolute minimum
}

gray_reason := "Advancing to next phase - all checks passed" if should_advance
gray_reason := sprintf("Rolling back - %d consecutive failures", [input.version.consecutive_failures]) if {
    should_rollback
    input.version.consecutive_failures >= 3
}
gray_reason := "Rolling back - critical error rate" if {
    should_rollback
    input.metrics.error_rate > 0.10
}
gray_reason := "Rolling back - critical latency" if {
    should_rollback
    input.metrics.p95_latency_ms > input.thresholds.p95_latency_ms * 3
}
gray_reason := "Holding - monitoring metrics" if should_hold


# ==================== Cost Budget Policy ====================
# Enforces cost limits per job/promotion

cost_budget := result if {
    result := {
        "allow": budget_ok,
        "remaining": budget_remaining,
        "warning": budget_warning,
    }
}

budget_ok if {
    input.spent < input.budget.daily_limit
    input.spent < input.budget.promotion_limit
}

budget_remaining := input.budget.daily_limit - input.spent

budget_warning if {
    input.spent > input.budget.daily_limit * 0.8
}

default budget_warning := false


# ==================== Access Control Policy ====================
# Controls who can trigger lifecycle actions

allow_action if {
    action_authorized
    resource_authorized
}

action_authorized if {
    input.action == "view"
}

action_authorized if {
    input.action in ["promote", "rollback", "downgrade"]
    input.user.role in ["admin", "platform-engineer"]
}

action_authorized if {
    input.action == "evaluate"
    input.user.role in ["admin", "platform-engineer", "ml-engineer"]
}

resource_authorized if {
    input.resource.version in input.user.allowed_versions
}

resource_authorized if {
    input.user.role == "admin"
}
