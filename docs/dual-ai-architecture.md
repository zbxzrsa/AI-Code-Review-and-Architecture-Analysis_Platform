# Dual AI Model Architecture

## Overview

The platform implements a sophisticated dual AI model architecture with two specialized services:

1. **Version Control AI** - Admin-only evaluation and promotion decisions
2. **Code Review AI** - User-facing code analysis and recommendations

This separation ensures that production stability (V2) is protected while enabling powerful experimentation capabilities.

---

## Version Control AI Service

### Purpose

Automated experiment evaluation using advanced statistical methods and machine learning techniques to make promotion decisions from V1 to V2.

### Responsibilities

#### 1. Automated Experiment Evaluation

- Comprehensive metrics analysis
- Statistical significance testing
- Baseline comparison
- Confidence scoring

#### 2. Statistical Significance Testing

```
Tests Performed:
├── T-test for accuracy improvements
├── Chi-square test for error rate changes
├── Effect size calculation
└── Confidence interval computation
```

#### 3. Cross-Version Comparative Analysis (A/B Testing)

- Control vs. treatment group comparison
- Conversion rate analysis
- Confidence intervals
- Winner determination

#### 4. Regression Detection

```
Regression Types:
├── Accuracy regression (< 95% of baseline)
├── Latency regression (> 120% of baseline)
├── Cost regression (> 120% of baseline)
├── Error rate regression (> 120% of baseline)
└── Security regression (new vulnerabilities)
```

#### 5. Cost-Benefit Analysis

- Current vs. proposed cost per request
- Accuracy improvement value
- Latency improvement value
- ROI calculation
- Break-even analysis

#### 6. Audit Trail Generation

- Structured reports with cryptographic signatures
- S3 storage with integrity verification
- Complete decision reasoning
- Recommendations for improvement

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│           Version Control AI Service                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Event Listener                                            │
│  └─→ experiment.completed events                           │
│                                                             │
│  Evaluation Pipeline                                       │
│  ├─→ Statistical Tests                                     │
│  ├─→ Regression Detection                                  │
│  ├─→ Cost-Benefit Analysis                                 │
│  ├─→ A/B Testing Analysis                                  │
│  └─→ Decision Making                                       │
│                                                             │
│  Output Generation                                         │
│  ├─→ Structured Report                                     │
│  ├─→ Cryptographic Signature                               │
│  ├─→ S3 Storage                                            │
│  └─→ OPA Policy Enforcement                                │
│                                                             │
│  Promotion Decision                                        │
│  ├─→ PROMOTE (to V2)                                       │
│  ├─→ REJECT (to V3)                                        │
│  ├─→ MANUAL_REVIEW (requires admin)                        │
│  └─→ INSUFFICIENT_DATA (more testing needed)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deployment

**Kubernetes Namespace**: `platform-v1-exp`

**Resource Configuration**:

- CPU: 1000m request, 4 limit
- Memory: 2Gi request, 8Gi limit
- GPU: 1x NVIDIA GPU (for intensive evaluation)

**Replicas**: 2 (for redundancy)

**Node Affinity**: GPU-accelerated nodes

**Scaling**: Manual (admin-controlled)

### Promotion Decision Logic

```python
if critical_regressions:
    decision = REJECT
elif insufficient_statistical_significance:
    decision = INSUFFICIENT_DATA
elif cost_increase > 20%:
    decision = MANUAL_REVIEW
elif ab_test_winner != treatment:
    decision = REJECT
else:
    decision = PROMOTE
```

### Event-Driven Integration

```
V1 Experimentation
    ↓
experiment.completed event
    ↓
Version Control AI (listens)
    ↓
Evaluation Pipeline
    ↓
promotion.approved / promotion.rejected event
    ↓
V2 Production / V3 Quarantine
```

### Report Structure

```json
{
  "id": "report_uuid",
  "experiment_id": "exp_uuid",
  "timestamp": "2025-12-02T12:00:00Z",
  "decision": "promote",
  "statistical_tests": [
    {
      "test_name": "t-test",
      "p_value": 0.032,
      "confidence_level": 0.95,
      "is_significant": true,
      "effect_size": 0.15,
      "sample_size": 1000
    }
  ],
  "regressions": [],
  "cost_benefit": {
    "current_cost": 0.03,
    "proposed_cost": 0.035,
    "cost_increase_pct": 16.7,
    "accuracy_improvement": 0.02,
    "roi_months": 2.5,
    "recommendation": "REVIEW"
  },
  "ab_test_results": {
    "winner": "treatment",
    "confidence_interval": [0.015, 0.025]
  },
  "overall_confidence": 0.95,
  "reasoning": "All statistical tests show significant improvements...",
  "recommendations": [
    "Proceed with promotion to V2",
    "Monitor metrics closely for first 24 hours"
  ],
  "signature": "sha256_hash_for_integrity"
}
```

---

## Code Review AI Service

### Purpose

User-facing code analysis service providing comprehensive code review with multiple analysis dimensions.

### Responsibilities

#### 1. Security Vulnerability Scanning (SAST)

- Code injection detection (eval, exec)
- SQL injection vulnerabilities
- XSS vulnerabilities
- Authentication/authorization issues
- Cryptography weaknesses
- CWE/OWASP mapping

#### 2. Code Quality and Style Analysis

- PEP 8 compliance (Python)
- Naming conventions
- Code duplication detection
- Complexity metrics
- Maintainability scoring

#### 3. Performance Bottleneck Detection

- Nested loop identification
- Inefficient algorithms
- Memory concerns
- Caching opportunities
- Complexity scoring (0-100)

#### 4. Architecture Dependency Analysis

- Dependency graph generation
- Circular dependency detection
- Design pattern identification
- Anti-pattern detection
- Modularity scoring
- Coupling analysis

#### 5. Test Generation and Coverage Recommendations

- Uncovered line identification
- Test template generation
- Edge case suggestions
- Coverage percentage calculation
- Test type recommendations

#### 6. Documentation and Comment Generation

- Missing docstring detection
- Parameter documentation
- Return value documentation
- Example generation
- Comment suggestions

#### 7. Intelligent Patch Generation

- Automatic fix suggestions
- Code transformation examples
- Breaking change warnings
- Confidence scoring

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│            Code Review AI Service                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  User Request                                              │
│  ├─→ Code snippet                                          │
│  ├─→ Language                                              │
│  ├─→ User API keys (optional)                              │
│  └─→ Feature flags                                         │
│                                                             │
│  Model Selection                                           │
│  ├─→ User-provided keys (priority)                         │
│  ├─→ Fallback chain:                                       │
│  │   ├─→ OpenAI GPT-4                                      │
│  │   ├─→ Anthropic Claude-3                                │
│  │   └─→ HuggingFace Local                                 │
│  └─→ Health check & availability                           │
│                                                             │
│  Parallel Analysis (async)                                 │
│  ├─→ Security scanning                                     │
│  ├─→ Code quality analysis                                 │
│  ├─→ Performance analysis                                  │
│  ├─→ Architecture analysis                                 │
│  ├─→ Test recommendations                                  │
│  └─→ Patch generation                                      │
│                                                             │
│  Feature Flag Application                                  │
│  ├─→ SAST scanning (always on)                             │
│  ├─→ Performance analysis (80% rollout)                    │
│  ├─→ Patch generation (gradual rollout)                    │
│  └─→ Test generation (canary: beta users)                  │
│                                                             │
│  Result Aggregation                                        │
│  ├─→ Overall score calculation                             │
│  ├─→ Severity ranking                                      │
│  └─→ Confidence scoring                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Deployment

**Kubernetes Namespace**: `platform-v2-stable`

**Resource Configuration**:

- CPU: 500m request, 2 limit
- Memory: 1Gi request, 4Gi limit
- No GPU required

**Replicas**: 3-50 (with HPA)

**Scaling Triggers**:

- CPU: 70% utilization
- Memory: 80% utilization

**Scale Up**: 100% increase per 30 seconds
**Scale Down**: 50% decrease per 60 seconds

### User-Provided API Keys

Users can provide their own API keys for model selection:

```python
user_api_keys = {
    "openai": "sk-user-key",
    "anthropic": "sk-ant-user-key",
    "huggingface": "hf_user_token"
}

result = await code_review_ai.analyze_code(
    code=code_snippet,
    language="python",
    user_api_keys=user_api_keys
)
```

### Multi-Model Routing with Fallback

```
Request arrives
    ↓
Check user API keys
    ├─→ If provided, try in order
    └─→ If fails, continue to default chain
    ↓
Try default routing chain:
    1. OpenAI GPT-4 (primary)
    2. Anthropic Claude-3 (secondary)
    3. HuggingFace Local (fallback)
    ↓
Return result from first available model
```

### Feature Flags for Gradual Rollouts

```yaml
Flags:
  code-review-ai-sast:
    enabled: true
    strategy: all_users

  code-review-ai-performance-analysis:
    enabled: true
    strategy: percentage
    percentage: 80%

  code-review-ai-patch-generation:
    enabled: false
    strategy: gradual

  code-review-ai-test-generation:
    enabled: false
    strategy: canary
    allowed_users: [admin@example.com, beta-tester@example.com]
```

### Result Structure

```json
{
  "id": "review_uuid",
  "timestamp": "2025-12-02T12:00:00Z",
  "code_language": "python",
  "code_length": 450,
  "security_vulnerabilities": [
    {
      "id": "vuln_1",
      "severity": "critical",
      "type": "Code Injection",
      "line": 15,
      "description": "Use of eval is dangerous",
      "remediation": "Use ast.literal_eval",
      "cwe_id": "CWE-95",
      "owasp_category": "A03:2021 – Injection",
      "confidence": 0.95
    }
  ],
  "code_issues": [
    {
      "id": "issue_1",
      "category": "style",
      "severity": "low",
      "line": 1,
      "description": "Line too long",
      "suggestion": "Keep lines under 100 characters",
      "confidence": 0.9
    }
  ],
  "performance_analysis": {
    "bottlenecks": ["Nested loop detected"],
    "optimization_opportunities": ["Use list comprehension"],
    "estimated_improvement": 15.0,
    "complexity_score": 72.0
  },
  "architecture_analysis": {
    "dependencies": ["requests", "numpy"],
    "design_patterns": ["Factory Pattern"],
    "anti_patterns": ["God Class"],
    "modularity_score": 78.0,
    "coupling_score": 35.0
  },
  "test_recommendations": {
    "uncovered_lines": [10, 15, 20],
    "coverage_percentage": 75.0,
    "recommended_tests": ["Test error handling"],
    "edge_cases": ["Empty input", "None values"]
  },
  "patch_suggestions": [
    {
      "id": "patch_1",
      "issue_id": "vuln_1",
      "original_code": "result = eval(user_input)",
      "patched_code": "result = ast.literal_eval(user_input)",
      "explanation": "Use ast.literal_eval for safer evaluation",
      "confidence": 0.95,
      "breaking_changes": false
    }
  ],
  "documentation_suggestions": ["Add module-level docstring"],
  "comment_suggestions": ["Explain complex algorithm on line 15"],
  "model_used": "openai-gpt4",
  "analysis_time_ms": 2850,
  "overall_score": 72.5,
  "feature_flags_applied": [
    "code-review-ai-sast",
    "code-review-ai-performance-analysis"
  ]
}
```

---

## Event-Driven Architecture

### Event Types

```python
EventType:
  EXPERIMENT_CREATED
  EXPERIMENT_STARTED
  EXPERIMENT_COMPLETED      ← Version Control AI listens
  EXPERIMENT_FAILED
  PROMOTION_REQUESTED
  PROMOTION_APPROVED        ← Triggers V2 deployment
  PROMOTION_REJECTED        ← Triggers V3 quarantine
  QUARANTINE_REQUESTED
  CODE_REVIEW_REQUESTED
  CODE_REVIEW_COMPLETED
```

### Event Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Event Flow                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Experiment Completed                                   │
│     V1 Experimentation → emit experiment.completed         │
│                                                             │
│  2. Event Bus Routes                                       │
│     Event Bus → Version Control AI (listener)              │
│                                                             │
│  3. Evaluation Pipeline                                    │
│     Version Control AI → Statistical Tests                 │
│                       → Regression Detection               │
│                       → Cost-Benefit Analysis              │
│                       → Decision Making                    │
│                                                             │
│  4. Emit Promotion Decision                                │
│     Version Control AI → emit promotion.approved/rejected  │
│                                                             │
│  5. Route to Destination                                   │
│     If approved → V2 Production (deployment)               │
│     If rejected → V3 Quarantine (archival)                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Correlation IDs

All events include correlation IDs for tracing:

```
User Request
    ↓ (correlation_id: abc123)
Code Review Request
    ↓ (correlation_id: abc123)
Code Review Completed
    ↓ (correlation_id: abc123)
Result Stored
```

---

## Feature Flags

### Rollout Strategies

#### 1. All Users

Feature is enabled for all users immediately.

#### 2. Percentage-Based

Feature is enabled for a percentage of users (deterministic hash-based).

```python
user_hash = hash(user_id) % 100
is_enabled = user_hash < rollout_percentage
```

#### 3. User List

Feature is enabled only for specific users (canary deployment).

#### 4. Gradual

Feature is enabled gradually over time (10% per hour).

```python
elapsed_hours = (now - created_at).hours
current_percentage = min(100, elapsed_hours * 10)
```

### Configuration

```python
# Enable SAST for all users
flag_service.register_flag(FeatureFlag(
    name="code-review-ai-sast",
    enabled=True,
    rollout_strategy=RolloutStrategy.ALL_USERS,
))

# Gradual rollout of patch generation
flag_service.register_flag(FeatureFlag(
    name="code-review-ai-patch-generation",
    enabled=False,
    rollout_strategy=RolloutStrategy.GRADUAL,
))

# Canary deployment for test generation
flag_service.register_flag(FeatureFlag(
    name="code-review-ai-test-generation",
    enabled=False,
    rollout_strategy=RolloutStrategy.CANARY,
    allowed_users=["admin@example.com", "beta-tester@example.com"],
))
```

---

## OPA Policy Engine Integration

### Purpose

Open Policy Agent (OPA) enforces policies for promotion decisions.

### Example Policies

```rego
# Reject if critical vulnerabilities detected
deny[msg] {
    input.decision == "promote"
    count(input.security_vulnerabilities[v | v.severity == "critical"]) > 0
    msg := "Cannot promote with critical vulnerabilities"
}

# Reject if cost increase > 25%
deny[msg] {
    input.decision == "promote"
    input.cost_benefit.cost_increase_percentage > 25
    msg := "Cost increase exceeds 25% threshold"
}

# Require manual review if accuracy improvement < 1%
warn[msg] {
    input.decision == "promote"
    input.cost_benefit.accuracy_improvement < 0.01
    msg := "Low accuracy improvement, recommend manual review"
}
```

---

## S3 Report Storage

### Structure

```
s3://platform-reports/
├── 2025/
│   ├── 12/
│   │   ├── 02/
│   │   │   ├── report_uuid_1.json
│   │   │   ├── report_uuid_1.json.sig
│   │   │   ├── report_uuid_2.json
│   │   │   └── report_uuid_2.json.sig
```

### Integrity Verification

Each report includes a cryptographic signature:

```python
signature = sha256(report_id + timestamp)
```

Verify on retrieval:

```python
stored_signature = s3.get_object(f"{report_id}.json.sig")
computed_signature = sha256(report_id + timestamp)
assert stored_signature == computed_signature
```

---

## Monitoring and Observability

### Metrics

**Version Control AI**:

- `version_control_ai_evaluations_total` - Total evaluations
- `version_control_ai_promotions_total` - Promotions to V2
- `version_control_ai_rejections_total` - Rejections to V3
- `version_control_ai_evaluation_duration_seconds` - Evaluation time

**Code Review AI**:

- `code_review_ai_analyses_total` - Total analyses
- `code_review_ai_analysis_duration_seconds` - Analysis time
- `code_review_ai_vulnerabilities_detected` - Security issues found
- `code_review_ai_model_routing_fallbacks` - Fallback usage

### Alerts

```yaml
alerts:
  - name: VersionControlAIHighRejectionRate
    condition: rejection_rate > 0.3
    severity: warning

  - name: CodeReviewAIHighLatency
    condition: p95_latency > 5000ms
    severity: critical

  - name: CodeReviewAIModelUnavailable
    condition: all_models_failed
    severity: critical
```

---

## Best Practices

### Version Control AI

1. **Always run statistical tests** - Don't rely on single metrics
2. **Set conservative thresholds** - Better to reject than promote unstable code
3. **Monitor cost carefully** - Prevent expensive model upgrades
4. **Review OPA policies regularly** - Update as requirements change
5. **Archive all reports** - Maintain audit trail for compliance

### Code Review AI

1. **Use feature flags** - Gradual rollout of new analysis types
2. **Support user API keys** - Let power users choose their models
3. **Implement fallback chains** - Ensure service availability
4. **Monitor model performance** - Track accuracy and latency per model
5. **Cache results** - Reduce redundant analyses

---

## Troubleshooting

### Version Control AI Not Evaluating

1. Check event bus is running: `kubectl get pods -n platform-v1-exp`
2. Verify event listener is registered
3. Check logs: `kubectl logs -n platform-v1-exp version-control-ai-*`
4. Verify S3 connectivity: `aws s3 ls s3://platform-reports/`

### Code Review AI Slow

1. Check model availability: `curl http://code-review-ai:8000/health/status`
2. Monitor HPA: `kubectl get hpa -n platform-v2-stable`
3. Check resource usage: `kubectl top pods -n platform-v2-stable`
4. Review model routing: Check which fallback model is being used

### Feature Flags Not Applied

1. Verify flag service is initialized
2. Check flag configuration: `GET /api/v1/feature-flags`
3. Verify user ID is correct
4. Check rollout percentage calculation

---

## Future Enhancements

- [ ] Machine learning-based threshold optimization
- [ ] Real-time A/B test result streaming
- [ ] Custom metric definitions
- [ ] Advanced regression detection (time-series analysis)
- [ ] Automated cost negotiation with providers
- [ ] Integration with external ML platforms
- [ ] Advanced patch generation with ML
- [ ] Semantic code analysis
- [ ] Multi-language support expansion
