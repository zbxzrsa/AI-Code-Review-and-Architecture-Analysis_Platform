# AI Orchestration Layer Documentation

## Overview

Comprehensive AI orchestration system with four specialized services:

1. **AI Orchestrator** - Unified routing and multi-expert coordination
2. **Version Control Service** - Experiment lifecycle and promotion workflows
3. **Comparison Service** - A/B testing and statistical analysis
4. **Provider Service** - Model provider management and quota enforcement

---

## AI Orchestrator

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  AI Orchestrator                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Top Layer - Task Orchestrator                             │
│  ├─→ Complexity estimation (0-1 scale)                     │
│  ├─→ Expert selection                                      │
│  ├─→ Budget allocation                                     │
│  ├─→ Quality-latency-cost optimization                     │
│  └─→ Output validation & sanitization                      │
│                                                             │
│  Middle Layer - Expert Coordinator                         │
│  ├─→ Pipeline/parallel/backtrack execution                 │
│  ├─→ Consistency scoring                                   │
│  ├─→ Confidence aggregation (Borda count)                  │
│  └─→ Retry logic with exponential backoff                  │
│                                                             │
│  Bottom Layer - Expert Agents (7 types)                    │
│  ├─→ Security Expert (SAST + LLM review)                   │
│  ├─→ Quality Expert (complexity + code smells)             │
│  ├─→ Performance Expert (profiling analysis)               │
│  ├─→ Dependency Expert (graph traversal)                   │
│  ├─→ Test Expert (coverage gaps)                           │
│  ├─→ Documentation Expert (missing docs)                   │
│  └─→ Meta Expert (output validation)                       │
│                                                             │
│  Provider Router                                           │
│  ├─→ Complexity-based routing                              │
│  ├─→ Health-based routing                                  │
│  ├─→ Cost-based routing                                    │
│  ├─→ Latency-based routing                                 │
│  └─→ Success-rate-based routing                            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Complexity Estimation

**Score Calculation**:

```
base_score = min(code_length / 10000, 1.0)
language_multiplier = {
  python: 0.8,
  javascript: 0.9,
  java: 1.0,
  rust: 1.1,
  cpp: 1.2
}
complexity_score = base_score * language_multiplier

estimated_cost = 0.01 + (complexity_score * 0.09)
estimated_latency = 1000 + int(complexity_score * 4000)
```

**Expert Selection**:

- Simple tasks (< 0.3): Quality, Meta
- Medium tasks (0.3-0.7): Quality, Dependency, Meta
- Complex tasks (> 0.7): All experts

### Execution Modes

#### 1. Pipeline Mode

- Sequential expert execution
- Each expert uses previous results
- Lower latency, higher accuracy

#### 2. Parallel Mode

- Concurrent expert execution
- Independent analysis
- Higher latency, faster overall

#### 3. Backtrack Mode

- Retry with exponential backoff
- Fallback on failure
- Most reliable

### Prompt Engineering

**System Prompt Template**:

```
You are a {{expert_type}} for code review.
Language: {{language}}
Framework: {{framework}}

Context:
{{code_context}}

Task: {{specific_task}}

Output Format (JSON):
{
  "findings": [
    {
      "type": "string",
      "severity": "critical|high|medium|low",
      "line_range": [start, end],
      "description": "string",
      "recommendation": "string",
      "confidence": 0.0-1.0
    }
  ],
  "summary": "string",
  "execution_time_ms": number
}

Constraints:
- Maximum 15 findings per analysis
- Prioritize actionable recommendations
- Include code snippets in recommendations
```

### Provider Management

**Supported Providers**:

- OpenAI (gpt-4-turbo, gpt-3.5-turbo)
- Anthropic (claude-3-opus, claude-3-sonnet)
- HuggingFace (CodeLlama, StarCoder)
- Local (vLLM, TGI)
- User-provided (encrypted API keys)

**Routing Strategies**:

1. **Complexity-based**: Simple → cheap, Complex → powerful
2. **Health-based**: Skip unhealthy providers
3. **Cost-based**: Minimize cost within quality threshold
4. **Latency-based**: Prefer faster models for interactive requests
5. **Success-rate-based**: Favor reliable models

**Fallback Chain Example**:

```
Primary: user-provided-gpt-4
  ↓ (on failure)
Secondary: platform-claude-3-sonnet
  ↓ (on failure)
Tertiary: platform-gpt-3.5-turbo
  ↓ (on failure)
Final: local-codellama (degraded mode)
```

---

## Version Control Service

### Experiment Lifecycle

```
1. Create
   └─→ Admin defines new model config in v1

2. Execute
   └─→ Run against test dataset (100+ samples)

3. Evaluate
   └─→ Version Control AI scores:
       - Accuracy (vs ground truth)
       - Precision/Recall
       - False positive rate
       - Latency (p50, p95, p99)
       - Cost per analysis
       - Error rate

4. Gate Check
   └─→ OPA policy evaluation:
       - accuracy > 0.85
       - error_rate < 0.05
       - cost_increase < 20%

5. Decision
   ├─→ Pass → Promote to v2
   └─→ Fail → Quarantine to v3 with reason

6. Audit
   └─→ Immutable record in audits table
```

### Database Schema

```sql
experiments:
  id (UUID, PK)
  name (String)
  version (String: v1)
  config (JSON)
  dataset_id (UUID)
  status (Enum)
  created_by (UUID)
  created_at (DateTime)

evaluations:
  id (UUID, PK)
  experiment_id (UUID, FK)
  metrics (JSON)
  ai_verdict (String: pass, fail, manual_review)
  ai_confidence (String)
  human_override (String)
  override_reason (Text)
  evaluated_by (String: ai, human)
  evaluated_at (DateTime)

promotions:
  id (UUID, PK)
  from_version_id (UUID)
  to_version_id (UUID)
  status (Enum)
  reason (Text)
  approver_id (UUID)
  promoted_at (DateTime)

blacklist:
  id (UUID, PK)
  config_hash (String, unique)
  reason (Text)
  evidence (JSON)
  quarantined_at (DateTime)
  review_status (String)
  reviewed_by (UUID)
  reviewed_at (DateTime)
```

### API Endpoints

```
POST /experiments
  - Create new v1 experiment

POST /experiments/{id}/evaluate
  - Trigger evaluation

GET /experiments/{id}/metrics
  - Get detailed metrics

POST /experiments/{id}/promote
  - Promote to v2 (requires approval)

POST /experiments/{id}/quarantine
  - Move to v3

GET /versions/{v1_id}/compare/{v2_id}
  - Detailed comparison
```

---

## Comparison Service

### Comparison Dimensions

#### Functional Metrics

- Issue detection overlap (precision, recall, F1)
- False positive/negative rates
- Consistency across runs (stability)

#### Non-Functional Metrics

- Latency distribution (p50, p95, p99)
- Throughput (requests per second)
- Error rates by category
- Resource consumption (CPU, memory, GPU)

#### Economic Metrics

- Cost per analysis
- Token usage efficiency
- ROI calculation (value vs cost)

### Statistical Testing

**Tests Performed**:

- t-test for accuracy improvements
- Mann-Whitney U for non-normal distributions
- Effect size calculation
- Confidence intervals

**Significance Levels**:

- p < 0.05: Significant
- p < 0.01: Highly significant
- p > 0.05: Not significant

### Output Format

```json
{
  "comparison_id": "uuid",
  "versions": ["v1-exp-123", "v2-stable"],
  "dataset": "test-set-500",
  "dataset_size": 500,
  "metrics": {
    "accuracy": {
      "v1": 0.87,
      "v2": 0.89,
      "change": "+2.3%",
      "significant": true
    },
    "latency_p95": {
      "v1": 2.3,
      "v2": 1.8,
      "change": "-21.7%",
      "significant": true
    },
    "cost_per_analysis": {
      "v1": 0.05,
      "v2": 0.04,
      "change": "-20%"
    }
  },
  "statistical_tests": [
    {
      "test_name": "t-test",
      "metric": "accuracy",
      "p_value": 0.032,
      "is_significant": "yes",
      "effect_size": 0.15
    }
  ],
  "recommendation": "Promote v1 to v2",
  "confidence": 0.92,
  "generated_at": "2024-12-02T10:30:00Z"
}
```

### Database Schema

```sql
comparisons:
  id (UUID, PK)
  comparison_id (String, unique)
  v1_version_id (UUID)
  v2_version_id (UUID)
  dataset_id (UUID)
  dataset_size (Integer)
  accuracy_v1, accuracy_v2 (Float)
  precision_v1, precision_v2 (Float)
  recall_v1, recall_v2 (Float)
  latency_p50_v1, latency_p50_v2 (Float)
  latency_p95_v1, latency_p95_v2 (Float)
  latency_p99_v1, latency_p99_v2 (Float)
  cost_per_analysis_v1, cost_per_analysis_v2 (Float)
  recommendation (String)
  confidence (Float)
  created_at (DateTime)

statistical_tests:
  id (UUID, PK)
  comparison_id (UUID, FK)
  test_name (String)
  metric (String)
  p_value (Float)
  is_significant (String)
  effect_size (Float)
```

---

## Provider Service

### API Key Management

**Encryption Flow**:

1. **Storage**:

   - User submits API key via HTTPS
   - Generate Data Encryption Key (DEK) from KMS
   - Encrypt API key with DEK using AES-256-GCM
   - Encrypt DEK with KMS master key
   - Store encrypted_key + encrypted_dek in database
   - Display only last 4 characters to user

2. **Retrieval**:
   - Fetch encrypted_key + encrypted_dek from database
   - Decrypt DEK using KMS
   - Decrypt API key using DEK
   - Use key for single request (never cache)
   - Securely zero out key from memory

### Quota System

**Configuration**:

```sql
user_quotas:
  user_id (UUID, unique)
  daily_limit (Integer)
  monthly_limit (Integer)
  daily_cost_limit (Float)
  monthly_cost_limit (Float)

usage_tracking:
  user_id (UUID)
  date (String: YYYY-MM-DD)
  requests_count (Integer)
  tokens_used (Integer)
  cost_usd (Float)
```

**Enforcement**:

- Redis counter: `quota:{user_id}:{date}`
- Pre-flight check before each AI request
- Graceful degradation: offer lower-cost model if quota near limit
- Admin notifications at 80%, 90%, 100% usage

### Health Monitoring

**Schedule**: Every 5 minutes

**Checks**:

- HTTP 200 response from provider
- Response time < 10s
- Valid JSON response
- No rate limit errors (429)

**Actions on Failure**:

- Mark provider as unhealthy in Redis
- Trigger fallback routing
- Alert on-call engineer
- Auto-recovery after 3 consecutive successful checks

### Database Schema

```sql
providers:
  id (UUID, PK)
  name (String, unique)
  provider_type (String)
  model_name (String)
  api_endpoint (String)
  is_active (Boolean)
  is_platform_provided (Boolean)
  cost_per_1k_tokens (Float)
  max_tokens (Integer)
  timeout_seconds (Integer)

user_providers:
  id (UUID, PK)
  user_id (UUID)
  provider_name (String)
  provider_type (String)
  model_name (String)
  encrypted_api_key (Text)
  encrypted_dek (Text)
  key_last_4_chars (String)
  is_active (Boolean)

provider_health:
  id (UUID, PK)
  provider_id (UUID)
  is_healthy (Boolean)
  last_check_at (DateTime)
  last_error (Text)
  consecutive_failures (Integer)
  response_time_ms (Float)
  success_rate (Float)

user_quotas:
  user_id (UUID, unique)
  daily_limit (Integer)
  monthly_limit (Integer)
  daily_cost_limit (Float)
  monthly_cost_limit (Float)

usage_tracking:
  user_id (UUID)
  date (String)
  requests_count (Integer)
  tokens_used (Integer)
  cost_usd (Float)

cost_alerts:
  user_id (UUID)
  alert_type (String)
  threshold_percentage (Integer)
  triggered_at (DateTime)
  acknowledged (Boolean)
```

---

## Inter-Service Communication

### Event Flow

```
User Request
    ↓
AI Orchestrator
├─→ Estimate complexity
├─→ Select experts
├─→ Route to providers
└─→ Aggregate results
    ↓
Expert Agents
├─→ Security Expert
├─→ Quality Expert
├─→ Performance Expert
├─→ Dependency Expert
├─→ Test Expert
├─→ Documentation Expert
└─→ Meta Expert (validation)
    ↓
Provider Service
├─→ Check health
├─→ Check quota
├─→ Track usage
└─→ Manage costs
    ↓
Version Control Service (for experiments)
├─→ Create experiment
├─→ Execute analysis
├─→ Evaluate results
└─→ Promote/Quarantine
    ↓
Comparison Service (for A/B testing)
├─→ Compare versions
├─→ Run statistical tests
└─→ Generate report
```

---

## Security Considerations

1. **API Key Management**: Encrypt with AES-256-GCM + KMS
2. **Quota Enforcement**: Redis-backed rate limiting
3. **Health Monitoring**: Automatic failover on provider failure
4. **Cost Control**: Budget alerts and graceful degradation
5. **Audit Trail**: Complete logging of all operations

---

## Monitoring

### Prometheus Metrics

- `orchestrator_complexity_score` - Task complexity
- `orchestrator_expert_execution_time_seconds` - Expert execution time
- `orchestrator_provider_selection_total` - Provider selection count
- `version_control_experiments_total` - Total experiments
- `version_control_promotions_total` - Successful promotions
- `comparison_statistical_tests_total` - Statistical tests run
- `provider_health_checks_total` - Health check count
- `provider_quota_violations_total` - Quota violations

### Grafana Dashboards

- Orchestrator performance
- Expert execution times
- Provider health status
- Quota usage by user
- Cost tracking
- Experiment promotion rates
- Statistical test results

---

## Future Enhancements

- [ ] Advanced prompt optimization (DSPy)
- [ ] Multi-model ensemble voting
- [ ] Adaptive expert selection based on history
- [ ] Custom expert agents
- [ ] Real-time cost prediction
- [ ] Advanced quota management (burst allowance)
- [ ] Provider performance prediction
- [ ] Automated provider selection optimization
