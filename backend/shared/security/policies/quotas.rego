# Quota and Cost Control Policy
# Defines quota limits and cost control rules

package code_review.quotas

import future.keywords.if
import future.keywords.in

# ============================================
# Quota Definitions
# ============================================

# Default quotas by role
default_quotas = {
    "admin": {
        "daily_requests": 10000,
        "monthly_requests": 300000,
        "daily_cost_limit": 1000,
        "monthly_cost_limit": 30000,
        "concurrent_analyses": 100
    },
    "user": {
        "daily_requests": 1000,
        "monthly_requests": 30000,
        "daily_cost_limit": 100,
        "monthly_cost_limit": 3000,
        "concurrent_analyses": 10
    },
    "viewer": {
        "daily_requests": 100,
        "monthly_requests": 3000,
        "daily_cost_limit": 10,
        "monthly_cost_limit": 300,
        "concurrent_analyses": 1
    }
}

# Get user quota
user_quota = quota if {
    quota := default_quotas[input.user.role]
}

# ============================================
# Quota Enforcement
# ============================================

# Allow execution if within quota
allow_execution if {
    input.user.role == "admin"
}

allow_execution if {
    input.user.role in ["user", "viewer"]
    input.usage.daily_requests < default_quotas[input.user.role].daily_requests
    input.usage.daily_cost < default_quotas[input.user.role].daily_cost_limit
    input.usage.concurrent_analyses < default_quotas[input.user.role].concurrent_analyses
}

# ============================================
# Cost Control
# ============================================

# Warn if approaching daily cost limit
warn_cost_limit if {
    input.usage.daily_cost > (default_quotas[input.user.role].daily_cost_limit * 0.8)
}

# Warn if approaching monthly cost limit
warn_monthly_cost_limit if {
    input.usage.monthly_cost > (default_quotas[input.user.role].monthly_cost_limit * 0.8)
}

# Block if exceeds daily cost limit
deny_cost_limit if {
    input.usage.daily_cost >= default_quotas[input.user.role].daily_cost_limit
}

# Block if exceeds monthly cost limit
deny_monthly_cost_limit if {
    input.usage.monthly_cost >= default_quotas[input.user.role].monthly_cost_limit
}

# ============================================
# Request Rate Limiting
# ============================================

# Warn if approaching daily request limit
warn_request_limit if {
    input.usage.daily_requests > (default_quotas[input.user.role].daily_requests * 0.8)
}

# Warn if approaching monthly request limit
warn_monthly_request_limit if {
    input.usage.monthly_requests > (default_quotas[input.user.role].monthly_requests * 0.8)
}

# Block if exceeds daily request limit
deny_request_limit if {
    input.usage.daily_requests >= default_quotas[input.user.role].daily_requests
}

# Block if exceeds monthly request limit
deny_monthly_request_limit if {
    input.usage.monthly_requests >= default_quotas[input.user.role].monthly_requests
}

# ============================================
# Concurrent Analysis Limits
# ============================================

# Block if exceeds concurrent analysis limit
deny_concurrent_limit if {
    input.usage.concurrent_analyses >= default_quotas[input.user.role].concurrent_analyses
}

# ============================================
# Analysis Cost Estimation
# ============================================

# Estimate cost based on complexity
estimated_cost = cost if {
    complexity := input.analysis.complexity
    model_cost := input.analysis.model_cost_per_1k_tokens
    estimated_tokens := complexity * 1000
    cost := (estimated_tokens / 1000) * model_cost
}

# Check if estimated cost is within budget
allow_based_on_cost if {
    estimated_cost := estimated_cost
    remaining_daily_budget := default_quotas[input.user.role].daily_cost_limit - input.usage.daily_cost
    estimated_cost < remaining_daily_budget
}

# ============================================
# Quota Status
# ============================================

# Get quota status for user
quota_status = status if {
    quota := default_quotas[input.user.role]
    status := {
        "daily_requests": {
            "used": input.usage.daily_requests,
            "limit": quota.daily_requests,
            "remaining": quota.daily_requests - input.usage.daily_requests,
            "percentage": (input.usage.daily_requests / quota.daily_requests) * 100
        },
        "monthly_requests": {
            "used": input.usage.monthly_requests,
            "limit": quota.monthly_requests,
            "remaining": quota.monthly_requests - input.usage.monthly_requests,
            "percentage": (input.usage.monthly_requests / quota.monthly_requests) * 100
        },
        "daily_cost": {
            "used": input.usage.daily_cost,
            "limit": quota.daily_cost_limit,
            "remaining": quota.daily_cost_limit - input.usage.daily_cost,
            "percentage": (input.usage.daily_cost / quota.daily_cost_limit) * 100
        },
        "monthly_cost": {
            "used": input.usage.monthly_cost,
            "limit": quota.monthly_cost_limit,
            "remaining": quota.monthly_cost_limit - input.usage.monthly_cost,
            "percentage": (input.usage.monthly_cost / quota.monthly_cost_limit) * 100
        },
        "concurrent_analyses": {
            "used": input.usage.concurrent_analyses,
            "limit": quota.concurrent_analyses,
            "remaining": quota.concurrent_analyses - input.usage.concurrent_analyses
        }
    }
}
