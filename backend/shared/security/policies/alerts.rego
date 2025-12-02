# Alerts and Compliance Policy
# Defines when alerts should be triggered

package code_review.alerts

import future.keywords.if
import future.keywords.in

# ============================================
# Baseline Violation Alerts
# ============================================

# Alert on significant baseline violation
alert_on_violation if {
    input.event.type == "baseline_violated"
    input.metrics.deviation > 0.15
}

# Alert on accuracy drop
alert_accuracy_drop if {
    input.event.type == "baseline_violated"
    input.metrics.metric_type == "accuracy"
    input.metrics.deviation > 0.10
}

# Alert on error rate increase
alert_error_rate_increase if {
    input.event.type == "baseline_violated"
    input.metrics.metric_type == "error_rate"
    input.metrics.deviation > 0.05
}

# Alert on latency increase
alert_latency_increase if {
    input.event.type == "baseline_violated"
    input.metrics.metric_type == "latency"
    input.metrics.deviation > 0.20
}

# Alert on cost increase
alert_cost_increase if {
    input.event.type == "baseline_violated"
    input.metrics.metric_type == "cost"
    input.metrics.deviation > 0.25
}

# ============================================
# Provider Health Alerts
# ============================================

# Alert on provider failure
alert_provider_failure if {
    input.event.type == "provider_failure"
    input.event.consecutive_failures >= 3
}

# Alert on provider recovery
alert_provider_recovery if {
    input.event.type == "provider_recovery"
}

# Alert on provider degradation
alert_provider_degradation if {
    input.event.type == "provider_degraded"
    input.event.response_time_ms > 5000
}

# ============================================
# Quota Alerts
# ============================================

# Alert when approaching daily cost limit
alert_daily_cost_warning if {
    input.event.type == "quota_warning"
    input.metrics.daily_cost_percentage > 80
}

# Alert when exceeding daily cost limit
alert_daily_cost_exceeded if {
    input.event.type == "quota_exceeded"
    input.metrics.daily_cost_percentage >= 100
}

# Alert when approaching monthly cost limit
alert_monthly_cost_warning if {
    input.event.type == "quota_warning"
    input.metrics.monthly_cost_percentage > 80
}

# Alert when exceeding monthly cost limit
alert_monthly_cost_exceeded if {
    input.event.type == "quota_exceeded"
    input.metrics.monthly_cost_percentage >= 100
}

# ============================================
# Security Alerts
# ============================================

# Alert on failed authentication
alert_auth_failure if {
    input.event.type == "auth_failure"
    input.event.consecutive_failures >= 3
}

# Alert on unauthorized access attempt
alert_unauthorized_access if {
    input.event.type == "unauthorized_access"
}

# Alert on permission denied
alert_permission_denied if {
    input.event.type == "permission_denied"
    input.event.resource.type in ["api_key", "user", "admin"]
}

# ============================================
# Experiment Alerts
# ============================================

# Alert on experiment failure
alert_experiment_failure if {
    input.event.type == "experiment_failed"
}

# Alert on experiment timeout
alert_experiment_timeout if {
    input.event.type == "experiment_timeout"
    input.event.duration_seconds > 3600
}

# Alert on promotion failure
alert_promotion_failure if {
    input.event.type == "promotion_failed"
    input.event.reason != ""
}

# Alert on quarantine
alert_quarantine if {
    input.event.type == "experiment_quarantined"
}

# ============================================
# Alert Severity
# ============================================

# Critical alerts
critical_alerts = [
    "alert_daily_cost_exceeded",
    "alert_monthly_cost_exceeded",
    "alert_auth_failure",
    "alert_unauthorized_access",
    "alert_provider_failure",
    "alert_experiment_failure"
]

# Warning alerts
warning_alerts = [
    "alert_daily_cost_warning",
    "alert_monthly_cost_warning",
    "alert_provider_degradation",
    "alert_accuracy_drop",
    "alert_error_rate_increase",
    "alert_latency_increase"
]

# Info alerts
info_alerts = [
    "alert_provider_recovery",
    "alert_cost_increase",
    "alert_experiment_timeout"
]

# Get alert severity
alert_severity = severity if {
    alert_name := input.event.alert_name
    severity := "critical" if alert_name in critical_alerts
    severity := "warning" if alert_name in warning_alerts
    severity := "info" if alert_name in info_alerts
    severity := "unknown"
}

# ============================================
# Alert Routing
# ============================================

# Route critical alerts to on-call engineer
route_to_oncall if {
    input.event.alert_name in critical_alerts
}

# Route warning alerts to team
route_to_team if {
    input.event.alert_name in warning_alerts
}

# Route info alerts to logs
route_to_logs if {
    input.event.alert_name in info_alerts
}

# ============================================
# Alert Deduplication
# ============================================

# Suppress duplicate alerts within 5 minutes
suppress_duplicate if {
    input.event.last_alert_time != null
    (input.event.current_time - input.event.last_alert_time) < 300
}
