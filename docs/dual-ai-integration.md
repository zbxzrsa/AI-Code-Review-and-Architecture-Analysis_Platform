# Dual AI Integration Guide

## Quick Integration

### 1. Version Control AI - Automatic Promotion

```python
from backend.shared.services.event_bus import get_event_bus, EventType, emit_experiment_completed
from backend.shared.services.version_control_ai import VersionControlAI

# Initialize services
event_bus = get_event_bus()
version_control_ai = VersionControlAI()

# Subscribe to experiment completion
async def on_experiment_completed(event):
    """Handle experiment completion and trigger evaluation."""
    experiment_id = event.data["experiment_id"]
    metrics = event.data["metrics"]

    # Run evaluation
    report = await version_control_ai.evaluate_experiment(
        experiment_id=experiment_id,
        experiment_metrics=metrics,
        baseline_metrics=get_baseline_metrics(),
    )

    # Emit promotion decision
    if report.decision == PromotionDecision.PROMOTE:
        await emit_promotion_approved(
            experiment_id=experiment_id,
            report_id=report.id,
            correlation_id=event.correlation_id,
        )

# Register listener
event_bus.subscribe(EventType.EXPERIMENT_COMPLETED, on_experiment_completed)

# When experiment completes, emit event
await emit_experiment_completed(
    experiment_id="exp_123",
    metrics={
        "accuracy": 0.94,
        "latency_ms": 2800,
        "cost": 0.035,
        "error_rate": 0.025,
    }
)
```

### 2. Code Review AI - User-Facing Analysis

```python
from backend.shared.services.code_review_ai import CodeReviewAI

# Initialize service
code_review_ai = CodeReviewAI()

# Analyze code
result = await code_review_ai.analyze_code(
    code="""
def process_data(user_input):
    result = eval(user_input)  # Security issue!
    return result
""",
    language="python",
    user_api_keys={
        "openai": "sk-user-key",  # Optional: user's own key
    },
    feature_flags=["code-review-ai-sast", "code-review-ai-performance-analysis"],
)

# Access results
print(f"Overall Score: {result.overall_score}/100")
print(f"Vulnerabilities: {len(result.security_vulnerabilities)}")
print(f"Issues: {len(result.code_issues)}")
print(f"Model Used: {result.model_used}")
print(f"Analysis Time: {result.analysis_time_ms}ms")

# Get specific findings
for vuln in result.security_vulnerabilities:
    print(f"[{vuln.severity}] {vuln.type}: {vuln.description}")
    print(f"  Remediation: {vuln.remediation}")

# Get patch suggestions
for patch in result.patch_suggestions:
    print(f"Patch for {patch.issue_id}:")
    print(f"  Original: {patch.original_code}")
    print(f"  Patched:  {patch.patched_code}")
```

### 3. Feature Flags - Gradual Rollouts

```python
from backend.shared.services.feature_flags import get_feature_flag_service

flag_service = get_feature_flag_service()

# Check if feature is enabled for user
if flag_service.is_enabled("code-review-ai-patch-generation", user_id="user@example.com"):
    # Include patch suggestions in response
    patches = result.patch_suggestions
else:
    patches = []

# Get all enabled features for user
enabled_features = flag_service.get_enabled_flags(user_id="user@example.com")
print(f"Enabled features: {enabled_features}")

# Update flag rollout
flag_service.update_flag(
    flag_name="code-review-ai-patch-generation",
    rollout_percentage=50.0,  # Increase from 0% to 50%
)

# Get flag statistics
stats = flag_service.get_flag_stats()
print(f"Total flags: {stats['total_flags']}")
print(f"Enabled: {stats['enabled_flags']}")
```

### 4. Event Bus - Event-Driven Architecture

```python
from backend.shared.services.event_bus import get_event_bus, EventType

event_bus = get_event_bus()

# Subscribe to multiple events
async def on_promotion_approved(event):
    print(f"Promotion approved for {event.data['experiment_id']}")
    # Trigger V2 deployment

async def on_promotion_rejected(event):
    print(f"Promotion rejected for {event.data['experiment_id']}")
    # Trigger V3 quarantine

event_bus.subscribe(EventType.PROMOTION_APPROVED, on_promotion_approved)
event_bus.subscribe(EventType.PROMOTION_REJECTED, on_promotion_rejected)

# Get event history
history = event_bus.get_event_history(
    event_type=EventType.PROMOTION_APPROVED,
    limit=10,
)

for event in history:
    print(f"{event.timestamp}: {event.type.value}")
```

---

## Integration Points

### V1 Experimentation API

**Emit experiment completion**:

```python
# In v1-experimentation/src/routers/experiments.py
from backend.shared.services.event_bus import emit_experiment_completed

@router.post("/experiments/run/{experiment_id}")
async def run_experiment(experiment_id: str, request: RunExperimentRequest):
    # ... run experiment ...

    # Emit completion event
    await emit_experiment_completed(
        experiment_id=experiment_id,
        metrics=experiment.metrics.to_dict(),
        correlation_id=request.correlation_id,
    )

    return {"status": "completed", "metrics": experiment.metrics.to_dict()}
```

### V2 Production API

**Integrate Code Review AI**:

```python
# In v2-production/src/routers/code_review.py
from backend.shared.services.code_review_ai import CodeReviewAI
from backend.shared.services.feature_flags import is_feature_enabled

code_review_ai = CodeReviewAI()

@router.post("/code-review/analyze")
async def analyze_code(request: CodeReviewRequest, user_id: str = None):
    # Get user's API keys if provided
    user_api_keys = await get_user_api_keys(user_id)

    # Get enabled features for user
    feature_flags = await get_user_feature_flags(user_id)

    # Analyze code
    result = await code_review_ai.analyze_code(
        code=request.code,
        language=request.language,
        user_api_keys=user_api_keys,
        feature_flags=feature_flags,
    )

    return result.to_dict()
```

### V1 Evaluation Endpoint

**Integrate Version Control AI**:

```python
# In v1-experimentation/src/routers/evaluation.py
from backend.shared.services.version_control_ai import VersionControlAI
from backend.shared.services.event_bus import emit_promotion_approved, emit_promotion_rejected

version_control_ai = VersionControlAI()

@router.post("/evaluation/evaluate/{experiment_id}")
async def evaluate_experiment(experiment_id: str):
    # Get experiment and metrics
    experiment = await get_experiment(experiment_id)
    baseline = await get_baseline_metrics()

    # Run evaluation
    report = await version_control_ai.evaluate_experiment(
        experiment_id=experiment_id,
        experiment_metrics=experiment.metrics.to_dict(),
        baseline_metrics=baseline,
    )

    # Emit decision event
    if report.decision == PromotionDecision.PROMOTE:
        await emit_promotion_approved(
            experiment_id=experiment_id,
            report_id=report.id,
        )
    else:
        await emit_promotion_rejected(
            experiment_id=experiment_id,
            report_id=report.id,
        )

    return report.to_dict()
```

---

## Deployment

### 1. Build Docker Images

```bash
# Version Control AI
docker build -t platform-version-control-ai:latest \
  -f backend/version-control-ai/Dockerfile \
  backend/

# Code Review AI
docker build -t platform-code-review-ai:latest \
  -f backend/code-review-ai/Dockerfile \
  backend/

# Push to registry
docker push your-registry/platform-version-control-ai:latest
docker push your-registry/platform-code-review-ai:latest
```

### 2. Deploy to Kubernetes

```bash
# Deploy Version Control AI
kubectl apply -f kubernetes/deployments/version-control-ai.yaml

# Deploy Code Review AI
kubectl apply -f kubernetes/deployments/code-review-ai.yaml

# Verify deployments
kubectl get pods -n platform-v1-exp
kubectl get pods -n platform-v2-stable
```

### 3. Configure Secrets

```bash
# Update S3 credentials
kubectl set env deployment/version-control-ai \
  -n platform-v1-exp \
  AWS_ACCESS_KEY_ID=your_key \
  AWS_SECRET_ACCESS_KEY=your_secret

# Update OPA URL
kubectl set env deployment/version-control-ai \
  -n platform-v1-exp \
  OPA_URL=http://opa-service:8181
```

---

## Testing

### Unit Tests

```python
# tests/test_version_control_ai.py
import pytest
from backend.shared.services.version_control_ai import VersionControlAI, PromotionDecision

@pytest.mark.asyncio
async def test_promote_high_accuracy():
    """Test promotion decision for high accuracy improvement."""
    vca = VersionControlAI()

    report = await vca.evaluate_experiment(
        experiment_id="exp_123",
        experiment_metrics={"accuracy": 0.96, "latency_ms": 2700, "cost": 0.03},
        baseline_metrics={"accuracy": 0.92, "latency_ms": 2800, "cost": 0.03},
    )

    assert report.decision == PromotionDecision.PROMOTE
    assert report.overall_confidence > 0.9

@pytest.mark.asyncio
async def test_reject_critical_regression():
    """Test rejection for critical regression."""
    vca = VersionControlAI()

    report = await vca.evaluate_experiment(
        experiment_id="exp_124",
        experiment_metrics={"accuracy": 0.88, "latency_ms": 2800, "cost": 0.03},
        baseline_metrics={"accuracy": 0.92, "latency_ms": 2800, "cost": 0.03},
    )

    assert report.decision == PromotionDecision.REJECT
    assert len(report.regressions) > 0
```

```python
# tests/test_code_review_ai.py
import pytest
from backend.shared.services.code_review_ai import CodeReviewAI

@pytest.mark.asyncio
async def test_detect_code_injection():
    """Test detection of code injection vulnerability."""
    cra = CodeReviewAI()

    result = await cra.analyze_code(
        code="result = eval(user_input)",
        language="python",
    )

    assert len(result.security_vulnerabilities) > 0
    assert any(v.type == "Code Injection" for v in result.security_vulnerabilities)

@pytest.mark.asyncio
async def test_feature_flag_application():
    """Test feature flag application."""
    cra = CodeReviewAI()

    result = await cra.analyze_code(
        code="def test(): pass",
        language="python",
        feature_flags=["code-review-ai-sast"],
    )

    assert "code-review-ai-sast" in result.feature_flags_applied
```

### Integration Tests

```python
# tests/test_integration.py
import pytest
from backend.shared.services.event_bus import get_event_bus, EventType, emit_experiment_completed

@pytest.mark.asyncio
async def test_end_to_end_promotion():
    """Test end-to-end promotion workflow."""
    event_bus = get_event_bus()

    # Track events
    events = []

    async def capture_event(event):
        events.append(event)

    event_bus.subscribe(EventType.PROMOTION_APPROVED, capture_event)

    # Emit experiment completion
    await emit_experiment_completed(
        experiment_id="exp_test",
        metrics={"accuracy": 0.96, "latency_ms": 2700, "cost": 0.03},
    )

    # Wait for processing
    await asyncio.sleep(1)

    # Verify promotion event was emitted
    assert len(events) > 0
    assert events[0].type == EventType.PROMOTION_APPROVED
```

---

## Monitoring

### Prometheus Queries

```promql
# Version Control AI evaluation rate
rate(version_control_ai_evaluations_total[5m])

# Promotion success rate
rate(version_control_ai_promotions_total[5m]) / rate(version_control_ai_evaluations_total[5m])

# Code Review AI average analysis time
avg(code_review_ai_analysis_duration_seconds)

# Model routing fallback rate
rate(code_review_ai_model_routing_fallbacks[5m])
```

### Grafana Dashboards

Create dashboards for:

- Version Control AI evaluation metrics
- Code Review AI analysis metrics
- Feature flag rollout progress
- Model performance comparison
- Event bus throughput

---

## Troubleshooting

### Version Control AI Not Receiving Events

```bash
# Check event bus
kubectl logs -n platform-v1-exp version-control-ai-*

# Verify event listener registration
curl http://version-control-ai:8000/api/v1/debug/event-listeners

# Check S3 connectivity
aws s3 ls s3://platform-reports/
```

### Code Review AI Slow Analysis

```bash
# Check HPA status
kubectl get hpa -n platform-v2-stable code-review-ai-hpa

# Monitor pod resources
kubectl top pods -n platform-v2-stable

# Check model availability
curl http://code-review-ai:8000/api/v1/health/models
```

### Feature Flags Not Working

```bash
# Check flag service
curl http://code-review-ai:8000/api/v1/feature-flags

# Verify user ID
curl http://code-review-ai:8000/api/v1/feature-flags?user_id=test@example.com

# Check flag configuration
kubectl exec -it code-review-ai-* -n platform-v2-stable -- \
  python -c "from backend.shared.services.feature_flags import get_feature_flag_service; print(get_feature_flag_service().list_flags())"
```

---

## Performance Tuning

### Version Control AI

- Increase GPU memory for larger datasets
- Tune statistical test thresholds
- Optimize S3 upload performance
- Cache baseline metrics

### Code Review AI

- Increase HPA max replicas for high load
- Optimize model routing chain
- Cache analysis results
- Use local models for common languages

---

## Security Considerations

1. **API Key Management**: Rotate user API keys regularly
2. **Report Integrity**: Verify cryptographic signatures
3. **Access Control**: Enforce RBAC for admin operations
4. **Audit Trail**: Log all promotion decisions
5. **Data Privacy**: Encrypt sensitive data in transit and at rest

---

## Next Steps

1. Deploy both services to Kubernetes
2. Configure event bus listeners
3. Set up monitoring and alerting
4. Run integration tests
5. Gradually roll out features using feature flags
6. Monitor metrics and adjust thresholds
7. Collect feedback and iterate
