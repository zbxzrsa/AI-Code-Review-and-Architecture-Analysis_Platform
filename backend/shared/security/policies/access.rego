# Access Control Policy
# Defines who can access what resources and perform what actions

package code_review.access

import future.keywords.if
import future.keywords.in

# Default deny
default allow = false

# ============================================
# Version Access Control
# ============================================

# Users can only access v2 (production)
allow if {
    input.user.role == "user"
    input.resource.version == "v2"
    input.action in ["read", "execute"]
}

# Admins can access all versions
allow if {
    input.user.role == "admin"
}

# Viewers can only read v2
allow if {
    input.user.role == "viewer"
    input.resource.version == "v2"
    input.action == "read"
}

# ============================================
# Resource-Specific Access
# ============================================

# Users cannot delete experiments
deny if {
    input.resource.type == "experiment"
    input.user.role == "user"
    input.action == "delete"
}

# Users cannot modify other users' projects
deny if {
    input.resource.type == "project"
    input.resource.owner_id != input.user.id
    input.user.role == "user"
    input.action in ["update", "delete"]
}

# Users cannot access other users' API keys
deny if {
    input.resource.type == "api_key"
    input.resource.owner_id != input.user.id
    input.user.role != "admin"
}

# Users cannot access other users' data
deny if {
    input.resource.type == "user"
    input.resource.id != input.user.id
    input.user.role != "admin"
    input.action in ["read", "update", "delete"]
}

# ============================================
# Version-Specific Access Rules
# ============================================

# Allow version access based on role and version
allow_version_access if {
    input.user.role == "admin"
}

allow_version_access if {
    input.user.role == "user"
    input.resource.version == "v2"
}

allow_version_access if {
    input.user.role == "viewer"
    input.resource.version == "v2"
}

# ============================================
# Promotion Rules
# ============================================

# Only admins can promote
allow_promotion if {
    input.user.role == "admin"
    input.action == "promote_version"
    input.metrics.accuracy > 0.85
    input.metrics.error_rate < 0.05
    input.metrics.cost_increase < 0.20
}

# Promotion requirements
promotion_requirements = {
    "min_accuracy": 0.85,
    "max_error_rate": 0.05,
    "max_cost_increase": 0.20,
    "required_test_coverage": 0.80,
    "required_role": "admin"
}

# ============================================
# Quarantine Rules
# ============================================

# Only admins can quarantine
allow_quarantine if {
    input.user.role == "admin"
    input.action == "quarantine"
}

# ============================================
# Analysis Execution
# ============================================

# Users can execute analysis on v2
allow_analysis if {
    input.user.role == "user"
    input.resource.version == "v2"
    input.action == "execute"
}

# Admins can execute analysis on any version
allow_analysis if {
    input.user.role == "admin"
    input.action == "execute"
}

# ============================================
# Provider Management
# ============================================

# Only admins can manage providers
allow_provider_management if {
    input.user.role == "admin"
    input.action in ["create", "update", "delete"]
}

# Users can view provider status
allow_provider_view if {
    input.action == "read"
}

# ============================================
# Experiment Management
# ============================================

# Only admins can create experiments
allow_experiment_create if {
    input.user.role == "admin"
    input.action == "create"
}

# Only admins can update experiments
allow_experiment_update if {
    input.user.role == "admin"
    input.action == "update"
}

# Only admins can delete experiments
allow_experiment_delete if {
    input.user.role == "admin"
    input.action == "delete"
}

# Users and admins can read experiments
allow_experiment_read if {
    input.user.role in ["user", "admin"]
    input.action == "read"
}
