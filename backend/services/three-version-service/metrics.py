"""
Prometheus Metrics for Three-Version Evolution Service

Exposes metrics for monitoring the evolution cycle:
- Cycle status and phase
- AI instance health per version
- Error/fix rates
- Promotion/degradation counts
- Quarantine statistics
"""

from prometheus_client import Counter, Gauge, Histogram, Info, REGISTRY
from typing import Optional

# =============================================================================
# Cycle Metrics
# =============================================================================

EVOLUTION_CYCLE_STATUS = Gauge(
    'evolution_cycle_running',
    'Whether the evolution cycle is running (1=running, 0=stopped)'
)

EVOLUTION_CYCLE_PHASE = Info(
    'evolution_cycle_phase',
    'Current phase of the evolution cycle'
)

EVOLUTION_CYCLE_TOTAL = Counter(
    'evolution_cycles_total',
    'Total number of evolution cycles completed'
)

EVOLUTION_PHASE_DURATION = Histogram(
    'evolution_phase_duration_seconds',
    'Duration of each evolution phase',
    ['phase'],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 1800, 3600]
)

# =============================================================================
# Experiment Metrics
# =============================================================================

V1_EXPERIMENTS_TOTAL = Counter(
    'v1_experiments_total',
    'Total number of V1 experiments',
    ['status']  # started, completed, failed
)

V1_EXPERIMENTS_ACTIVE = Gauge(
    'v1_experiments_active',
    'Number of active V1 experiments'
)

# =============================================================================
# Error/Fix Metrics
# =============================================================================

V1_ERRORS_TOTAL = Counter(
    'v1_errors_total',
    'Total number of V1 errors reported',
    ['error_type']  # compatibility, performance, security, accuracy, stability
)

V2_FIXES_TOTAL = Counter(
    'v2_fixes_total',
    'Total number of V2 fixes generated',
    ['status']  # pending, applied, verified, failed
)

V2_FIX_SUCCESS_RATE = Gauge(
    'v2_fix_success_rate',
    'Success rate of V2 fixes (0-1)'
)

FIX_TEMPLATES_LEARNED = Gauge(
    'fix_templates_learned_total',
    'Number of fix templates learned from successful fixes'
)

ERROR_PATTERNS_RECORDED = Gauge(
    'error_patterns_recorded_total',
    'Number of error patterns recorded for learning'
)

# =============================================================================
# Promotion/Degradation Metrics
# =============================================================================

PROMOTIONS_TOTAL = Counter(
    'promotions_total',
    'Total number of promotions (V1 to V2)',
    ['status']  # requested, approved, completed, failed
)

DEGRADATIONS_TOTAL = Counter(
    'degradations_total',
    'Total number of degradations (V2 to V3)',
    ['status']  # triggered, completed
)

REEVALUATIONS_TOTAL = Counter(
    'reevaluations_total',
    'Total number of re-evaluations (V3 to V1)',
    ['status']  # requested, approved, rejected
)

PENDING_PROMOTIONS = Gauge(
    'pending_promotions',
    'Number of technologies pending promotion'
)

PENDING_DEGRADATIONS = Gauge(
    'pending_degradations',
    'Number of technologies pending degradation'
)

PENDING_REEVALUATIONS = Gauge(
    'pending_reevaluations',
    'Number of technologies pending re-evaluation'
)

# =============================================================================
# Quarantine Metrics
# =============================================================================

QUARANTINE_TOTAL = Gauge(
    'quarantine_technologies_total',
    'Total number of quarantined technologies'
)

PERMANENT_EXCLUSIONS = Gauge(
    'permanent_exclusions_total',
    'Number of permanently excluded technologies'
)

TEMPORARY_EXCLUSIONS = Gauge(
    'temporary_exclusions_total',
    'Number of temporarily excluded technologies'
)

QUARANTINE_BY_REASON = Gauge(
    'quarantine_by_reason',
    'Number of quarantined technologies by reason',
    ['reason']  # poor_performance, poor_accuracy, high_error_rate, security, incompatible, deprecated
)

EXCLUSION_RULES = Gauge(
    'exclusion_rules_total',
    'Number of learned exclusion rules'
)

FAILURE_PATTERNS = Gauge(
    'failure_patterns_recorded',
    'Number of recorded failure patterns'
)

# =============================================================================
# AI Instance Metrics
# =============================================================================

AI_INSTANCE_STATUS = Gauge(
    'ai_instance_status',
    'Status of AI instances (1=active, 0=inactive)',
    ['version', 'ai_type']  # v1/v2/v3, vc_ai/cr_ai
)

AI_INSTANCE_REQUESTS = Counter(
    'ai_instance_requests_total',
    'Total number of requests to AI instances',
    ['version', 'ai_type']
)

AI_INSTANCE_ERRORS = Counter(
    'ai_instance_errors_total',
    'Total number of errors from AI instances',
    ['version', 'ai_type']
)

AI_INSTANCE_LATENCY = Histogram(
    'ai_instance_latency_seconds',
    'Latency of AI instance requests',
    ['version', 'ai_type'],
    buckets=[0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 30.0]
)

AI_INSTANCE_ERROR_RATE = Gauge(
    'ai_instance_error_rate',
    'Error rate of AI instances (0-1)',
    ['version', 'ai_type']
)

# =============================================================================
# Service Info
# =============================================================================

SERVICE_INFO = Info(
    'three_version_service',
    'Information about the three-version service'
)

SERVICE_INFO.info({
    'version': '1.1.0',
    'name': 'three-version-evolution-service',
})


# =============================================================================
# Metric Update Functions
# =============================================================================

def update_cycle_status(running: bool, phase: Optional[str] = None):
    """Update cycle status metrics."""
    EVOLUTION_CYCLE_STATUS.set(1 if running else 0)
    if phase:
        EVOLUTION_CYCLE_PHASE.info({'phase': phase})


def record_experiment(status: str):
    """Record an experiment event."""
    V1_EXPERIMENTS_TOTAL.labels(status=status).inc()


def record_error(error_type: str):
    """Record a V1 error."""
    V1_ERRORS_TOTAL.labels(error_type=error_type).inc()


def record_fix(status: str):
    """Record a V2 fix event."""
    V2_FIXES_TOTAL.labels(status=status).inc()


def update_fix_success_rate(rate: float):
    """Update the fix success rate."""
    V2_FIX_SUCCESS_RATE.set(rate)


def record_promotion(status: str):
    """Record a promotion event."""
    PROMOTIONS_TOTAL.labels(status=status).inc()


def record_degradation(status: str):
    """Record a degradation event."""
    DEGRADATIONS_TOTAL.labels(status=status).inc()


def record_reevaluation(status: str):
    """Record a re-evaluation event."""
    REEVALUATIONS_TOTAL.labels(status=status).inc()


def update_pending_counts(promotions: int, degradations: int, reevaluations: int):
    """Update pending operation counts."""
    PENDING_PROMOTIONS.set(promotions)
    PENDING_DEGRADATIONS.set(degradations)
    PENDING_REEVALUATIONS.set(reevaluations)


def update_quarantine_stats(
    total: int,
    permanent: int,
    temporary: int,
    by_reason: dict,
    rules: int,
    patterns: int
):
    """Update quarantine statistics."""
    QUARANTINE_TOTAL.set(total)
    PERMANENT_EXCLUSIONS.set(permanent)
    TEMPORARY_EXCLUSIONS.set(temporary)
    
    for reason, count in by_reason.items():
        QUARANTINE_BY_REASON.labels(reason=reason).set(count)
    
    EXCLUSION_RULES.set(rules)
    FAILURE_PATTERNS.set(patterns)


def update_ai_status(version: str, ai_type: str, active: bool):
    """Update AI instance status."""
    AI_INSTANCE_STATUS.labels(version=version, ai_type=ai_type).set(1 if active else 0)


def record_ai_request(version: str, ai_type: str, latency_seconds: float, error: bool = False):
    """Record an AI request."""
    AI_INSTANCE_REQUESTS.labels(version=version, ai_type=ai_type).inc()
    AI_INSTANCE_LATENCY.labels(version=version, ai_type=ai_type).observe(latency_seconds)
    
    if error:
        AI_INSTANCE_ERRORS.labels(version=version, ai_type=ai_type).inc()


def update_ai_error_rate(version: str, ai_type: str, rate: float):
    """Update AI instance error rate."""
    AI_INSTANCE_ERROR_RATE.labels(version=version, ai_type=ai_type).set(rate)


def update_feedback_stats(templates: int, patterns: int):
    """Update feedback learning stats."""
    FIX_TEMPLATES_LEARNED.set(templates)
    ERROR_PATTERNS_RECORDED.set(patterns)


def cycle_completed():
    """Record a completed cycle."""
    EVOLUTION_CYCLE_TOTAL.inc()


# =============================================================================
# Metrics Collection Function
# =============================================================================

def collect_metrics_from_status(status: dict):
    """Collect all metrics from cycle status."""
    # Cycle status
    update_cycle_status(
        status.get('running', False),
        status.get('spiral_status', {}).get('current_cycle', {}).get('phase')
    )
    
    # Pending counts
    pending = status.get('spiral_status', {}).get('pending', {})
    update_pending_counts(
        pending.get('promotions', 0),
        pending.get('degradations', 0),
        pending.get('reevaluations', 0)
    )
    
    # Feedback stats
    feedback = status.get('spiral_status', {}).get('feedback_stats', {})
    if feedback:
        update_fix_success_rate(feedback.get('fix_success_rate', 0))
        update_feedback_stats(
            feedback.get('fix_templates_learned', 0),
            feedback.get('error_patterns_recorded', 0)
        )
    
    # Quarantine stats
    quarantine = status.get('spiral_status', {}).get('quarantine_stats', {})
    if quarantine:
        update_quarantine_stats(
            quarantine.get('total_quarantined', 0),
            quarantine.get('permanent_exclusions', 0),
            quarantine.get('temporary_exclusions', 0),
            quarantine.get('by_reason', {}),
            quarantine.get('exclusion_rules', 0),
            quarantine.get('failure_patterns_recorded', 0)
        )
    
    # AI status
    ai_status = status.get('spiral_status', {}).get('ai_status', {})
    for version, version_status in ai_status.items():
        if 'vc_ai' in version_status:
            update_ai_status(
                version, 'vc_ai',
                version_status['vc_ai'].get('status') == 'active'
            )
            update_ai_error_rate(
                version, 'vc_ai',
                version_status['vc_ai'].get('error_rate', 0)
            )
        if 'cr_ai' in version_status:
            update_ai_status(
                version, 'cr_ai',
                version_status['cr_ai'].get('status') == 'active'
            )
            update_ai_error_rate(
                version, 'cr_ai',
                version_status['cr_ai'].get('error_rate', 0)
            )
