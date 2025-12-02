# Audit and Compliance Policy
# Defines what actions should be audited

package code_review.audit

import future.keywords.if
import future.keywords.in

# ============================================
# Audit Rules
# ============================================

# Always audit admin actions
should_audit if {
    input.user.role == "admin"
}

# Always audit sensitive operations
should_audit if {
    input.action in [
        "promote_version",
        "quarantine_experiment",
        "delete_experiment",
        "update_baseline",
        "manage_provider",
        "update_quota",
        "create_api_key",
        "delete_api_key"
    ]
}

# Always audit access to sensitive resources
should_audit if {
    input.resource.type in [
        "api_key",
        "user",
        "admin_settings",
        "provider_config"
    ]
}

# Always audit failed access attempts
should_audit if {
    input.action == "access_denied"
}

# Always audit authentication events
should_audit if {
    input.action in [
        "login",
        "logout",
        "token_refresh",
        "password_change"
    ]
}

# Audit user actions on shared resources
should_audit if {
    input.user.role == "user"
    input.action in ["create", "update", "delete"]
}

# ============================================
# Audit Exclusions
# ============================================

# Don't audit read-only operations by viewers
dont_audit if {
    input.user.role == "viewer"
    input.action == "read"
}

# Don't audit health checks
dont_audit if {
    input.action == "health_check"
}

# Don't audit metrics collection
dont_audit if {
    input.action == "collect_metrics"
}

# ============================================
# Audit Details
# ============================================

# Audit details to capture
audit_details = details if {
    details := {
        "user_id": input.user.id,
        "user_role": input.user.role,
        "action": input.action,
        "resource_type": input.resource.type,
        "resource_id": input.resource.id,
        "timestamp": input.timestamp,
        "ip_address": input.ip_address,
        "user_agent": input.user_agent,
        "status": input.status,
        "details": input.details
    }
}

# ============================================
# Compliance Rules
# ============================================

# Ensure all sensitive operations are audited
compliance_audit_sensitive if {
    input.action in [
        "promote_version",
        "quarantine_experiment",
        "delete_experiment"
    ]
}

# Ensure all user data access is audited
compliance_audit_data_access if {
    input.resource.type in ["api_key", "user"]
    input.action in ["read", "update", "delete"]
}

# Ensure all admin actions are audited
compliance_audit_admin if {
    input.user.role == "admin"
}

# ============================================
# Retention Policies
# ============================================

# Retention periods by audit type
retention_policies = {
    "authentication": 90,          # 90 days
    "authorization": 90,           # 90 days
    "data_access": 365,            # 1 year
    "sensitive_operations": 365,   # 1 year
    "admin_actions": 365,          # 1 year
    "compliance": 2555             # 7 years
}

# Get retention period for audit
retention_period = period if {
    audit_type := input.audit_type
    period := retention_policies[audit_type]
}

# ============================================
# Audit Log Integrity
# ============================================

# Ensure audit logs are immutable
immutable_audit_log if {
    input.action != "delete"
    input.action != "update"
}

# Ensure audit logs are signed
signed_audit_log if {
    input.signature != null
    input.signature != ""
}

# Ensure audit logs are chained
chained_audit_log if {
    input.prev_hash != null
    input.prev_hash != ""
}

# ============================================
# Audit Queries
# ============================================

# Query audit logs for user
audit_logs_for_user = logs if {
    logs := [
        entry |
        entry := input.audit_logs[_];
        entry.user_id == input.user_id
    ]
}

# Query audit logs for resource
audit_logs_for_resource = logs if {
    logs := [
        entry |
        entry := input.audit_logs[_];
        entry.resource_id == input.resource_id
    ]
}

# Query audit logs for action
audit_logs_for_action = logs if {
    logs := [
        entry |
        entry := input.audit_logs[_];
        entry.action == input.action
    ]
}

# ============================================
# Audit Reporting
# ============================================

# Generate audit summary
audit_summary = summary if {
    total_entries := count(input.audit_logs)
    admin_actions := count([
        entry |
        entry := input.audit_logs[_];
        entry.user_role == "admin"
    ])
    failed_actions := count([
        entry |
        entry := input.audit_logs[_];
        entry.status == "denied"
    ])
    summary := {
        "total_entries": total_entries,
        "admin_actions": admin_actions,
        "failed_actions": failed_actions,
        "period": input.period
    }
}
