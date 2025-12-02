# API Reference

## Base URLs

- **V2 Production**: `http://localhost:8001/api/v1` (user-facing)
- **V1 Experimentation**: `http://localhost:8002/api/v1` (internal)
- **V3 Quarantine**: `http://localhost:8003/api/v1` (internal)

## Authentication

Currently, the API uses no authentication. In production, implement:

- JWT tokens for API access
- API keys for service-to-service communication
- OAuth 2.0 for user authentication

## V2 Production API

### Code Review Endpoints

#### POST /code-review/analyze

Analyze code and provide comprehensive review.

**Request:**

```json
{
  "code": "def hello():\n    print('Hello')",
  "language": "python",
  "focus_areas": ["security", "performance"],
  "include_architecture_analysis": true
}
```

**Response:**

```json
{
  "review_id": "review_1701505200000",
  "timestamp": "2025-12-02T12:10:00Z",
  "language": "python",
  "issues": [
    {
      "severity": "low",
      "type": "style",
      "line": 1,
      "description": "Use double quotes for strings",
      "suggestion": "Use double quotes consistently"
    }
  ],
  "suggestions": ["Add type hints", "Add docstring"],
  "architecture_insights": null,
  "security_concerns": [],
  "performance_notes": [],
  "confidence_score": 0.95,
  "analysis_time_ms": 1250.5,
  "model_used": "gpt-4"
}
```

**Status Codes:**

- `200`: Success
- `400`: Invalid request (code too long, unsupported language)
- `500`: Analysis failed

#### GET /code-review/reviews/{review_id}

Retrieve a previous code review.

**Response:**

```json
{
  "review_id": "review_1701505200000",
  "timestamp": "2025-12-02T12:10:00Z",
  "language": "python",
  "issues": [],
  "suggestions": [],
  "confidence_score": 0.95,
  "analysis_time_ms": 1250.5,
  "model_used": "gpt-4"
}
```

#### GET /code-review/reviews

List recent code reviews.

**Query Parameters:**

- `limit` (int, 1-100): Number of reviews to return (default: 10)
- `offset` (int): Pagination offset (default: 0)

**Response:**

```json
{
  "reviews": [
    {
      "review_id": "review_1701505200000",
      "timestamp": "2025-12-02T12:10:00Z",
      "language": "python"
    }
  ],
  "total": 42,
  "limit": 10,
  "offset": 0
}
```

### Health Endpoints

#### GET /health/status

Get overall service health status.

**Response:**

```json
{
  "status": "healthy",
  "version": "v2-production",
  "timestamp": "2025-12-02T12:10:00Z",
  "checks": {
    "database": "ok",
    "ai_models": "ok",
    "cache": "ok"
  }
}
```

#### GET /health/slo

Get SLO compliance status.

**Response:**

```json
{
  "slo_status": "compliant",
  "metrics": {
    "response_time_p95_ms": 2500,
    "slo_threshold_ms": 3000,
    "error_rate": 0.008,
    "slo_threshold": 0.02,
    "uptime_percentage": 99.95
  }
}
```

### Metrics Endpoints

#### GET /metrics/performance

Get performance metrics.

**Response:**

```json
{
  "timestamp": "2025-12-02T12:10:00Z",
  "period": "last_24h",
  "metrics": {
    "total_requests": 45230,
    "successful_reviews": 44890,
    "failed_reviews": 340,
    "average_response_time_ms": 1850,
    "p95_response_time_ms": 2800,
    "p99_response_time_ms": 2950,
    "error_rate": 0.0075,
    "throughput_rps": 0.52
  }
}
```

#### GET /metrics/slo-compliance

Get SLO compliance metrics.

**Response:**

```json
{
  "timestamp": "2025-12-02T12:10:00Z",
  "period": "last_30d",
  "slo_targets": {
    "response_time_p95_ms": 3000,
    "error_rate": 0.02,
    "uptime_percentage": 99.9
  },
  "actual_metrics": {
    "response_time_p95_ms": 2650,
    "error_rate": 0.0082,
    "uptime_percentage": 99.98
  },
  "compliance": {
    "response_time": "compliant",
    "error_rate": "compliant",
    "uptime": "compliant"
  }
}
```

#### GET /metrics/models

Get AI model performance metrics.

**Response:**

```json
{
  "timestamp": "2025-12-02T12:10:00Z",
  "models": {
    "gpt-4": {
      "requests": 22500,
      "average_latency_ms": 1800,
      "accuracy": 0.96,
      "cost": 675.0
    },
    "claude-3-opus": {
      "requests": 22390,
      "average_latency_ms": 1900,
      "accuracy": 0.94,
      "cost": 336.85
    }
  }
}
```

## V1 Experimentation API

### Experiment Management

#### POST /experiments/create

Create a new experiment.

**Request:**

```json
{
  "name": "Test GPT-4 with new prompt",
  "description": "Testing improved prompt template",
  "primary_model": "gpt-4",
  "secondary_model": "claude-3-opus-20240229",
  "prompt_template": "Review this code: {code}",
  "routing_strategy": "primary",
  "tags": ["gpt-4", "prompt-v2"]
}
```

**Response:**

```json
{
  "id": "exp_uuid",
  "name": "Test GPT-4 with new prompt",
  "description": "Testing improved prompt template",
  "status": "pending",
  "primary_model": "gpt-4",
  "secondary_model": "claude-3-opus-20240229",
  "routing_strategy": "primary",
  "created_at": "2025-12-02T12:10:00Z",
  "started_at": null,
  "completed_at": null,
  "promotion_status": "pending_evaluation",
  "metrics": null
}
```

#### POST /experiments/run/{experiment_id}

Run an experiment with provided code samples.

**Request:**

```json
{
  "code_samples": ["def test(): pass", "class MyClass: pass"],
  "language": "python"
}
```

**Response:**

```json
{
  "experiment_id": "exp_uuid",
  "status": "completed",
  "promotion_status": "passed",
  "metrics": {
    "accuracy": 0.94,
    "latency_ms": 2800,
    "cost": 15.5,
    "error_rate": 0.025,
    "throughput": 100,
    "user_satisfaction": 4.2,
    "false_positives": 2,
    "false_negatives": 3,
    "timestamp": "2025-12-02T12:10:00Z"
  },
  "duration_ms": 2850
}
```

#### GET /experiments/{experiment_id}

Get experiment details.

**Response:**

```json
{
  "id": "exp_uuid",
  "name": "Test GPT-4 with new prompt",
  "status": "completed",
  "primary_model": "gpt-4",
  "secondary_model": "claude-3-opus-20240229",
  "routing_strategy": "primary",
  "created_at": "2025-12-02T12:10:00Z",
  "started_at": "2025-12-02T12:10:05Z",
  "completed_at": "2025-12-02T12:10:10Z",
  "promotion_status": "passed",
  "metrics": {
    "accuracy": 0.94,
    "latency_ms": 2800,
    "cost": 15.5,
    "error_rate": 0.025
  }
}
```

#### GET /experiments

List experiments.

**Query Parameters:**

- `status` (string): Filter by status (pending, running, completed, failed)
- `limit` (int, 1-100): Number of experiments to return (default: 10)
- `offset` (int): Pagination offset (default: 0)

**Response:**

```json
{
  "experiments": [
    {
      "id": "exp_uuid",
      "name": "Test GPT-4 with new prompt",
      "status": "completed",
      "promotion_status": "passed",
      "created_at": "2025-12-02T12:10:00Z"
    }
  ],
  "total": 15,
  "limit": 10,
  "offset": 0
}
```

### Evaluation Endpoints

#### POST /evaluation/promote/{experiment_id}

Promote an experiment to V2 production.

**Query Parameters:**

- `force` (boolean): Bypass evaluation checks (default: false)

**Response:**

```json
{
  "experiment_id": "exp_uuid",
  "promotion_status": "pending",
  "message": "Promotion to V2 initiated",
  "timestamp": "2025-12-02T12:10:00Z"
}
```

#### POST /evaluation/quarantine/{experiment_id}

Quarantine an experiment that failed evaluation.

**Request:**

```json
{
  "reason": "Accuracy below threshold",
  "impact_analysis": {
    "affected_models": ["gpt-4"],
    "recommendation": "Adjust prompt template"
  }
}
```

**Response:**

```json
{
  "experiment_id": "exp_uuid",
  "quarantine_status": "quarantined",
  "reason": "Accuracy below threshold",
  "timestamp": "2025-12-02T12:10:00Z"
}
```

#### GET /evaluation/thresholds

Get current evaluation thresholds for promotion.

**Response:**

```json
{
  "accuracy_threshold": 0.95,
  "latency_threshold_ms": 3000,
  "error_rate_threshold": 0.02,
  "description": "Experiments must meet ALL thresholds to be promoted to V2"
}
```

#### GET /evaluation/status/{experiment_id}

Get evaluation status of an experiment.

**Response:**

```json
{
  "experiment_id": "exp_uuid",
  "evaluation_status": "pending",
  "metrics": {
    "accuracy": 0.94,
    "latency_ms": 2800,
    "error_rate": 0.025
  },
  "recommendation": "pending_analysis"
}
```

## V3 Quarantine API

### Quarantine Records

#### GET /quarantine/records

List all quarantined experiments.

**Query Parameters:**

- `limit` (int, 1-100): Number of records to return (default: 10)
- `offset` (int): Pagination offset (default: 0)

**Response:**

```json
{
  "records": [
    {
      "id": "quar_uuid",
      "experiment_id": "exp_uuid",
      "reason": "Accuracy below threshold",
      "quarantined_at": "2025-12-02T12:10:00Z",
      "can_re_evaluate": true
    }
  ],
  "total": 12,
  "limit": 10,
  "offset": 0
}
```

#### GET /quarantine/records/{record_id}

Get detailed quarantine record.

**Response:**

```json
{
  "id": "quar_uuid",
  "experiment_id": "exp_uuid",
  "reason": "Accuracy below threshold",
  "failure_analysis": {
    "accuracy": 0.92,
    "latency_ms": 3500,
    "error_rate": 0.035
  },
  "metrics_at_failure": {
    "accuracy": 0.92,
    "latency_ms": 3500,
    "cost": 20.0,
    "error_rate": 0.035
  },
  "quarantined_at": "2025-12-02T12:10:00Z",
  "quarantined_by": "system",
  "can_re_evaluate": true,
  "impact_analysis": {},
  "related_experiments": []
}
```

#### POST /quarantine/records/{experiment_id}

Create a new quarantine record.

**Request:**

```json
{
  "reason": "Accuracy below threshold",
  "failure_analysis": {
    "accuracy": 0.92,
    "latency_ms": 3500
  },
  "impact_analysis": {
    "affected_models": ["gpt-4"]
  }
}
```

**Response:**

```json
{
  "id": "quar_uuid",
  "experiment_id": "exp_uuid",
  "status": "quarantined",
  "timestamp": "2025-12-02T12:10:00Z"
}
```

#### POST /quarantine/records/{record_id}/request-re-evaluation

Request re-evaluation of a quarantined experiment.

**Request:**

```json
{
  "notes": "Fixed the prompt template, ready to retry"
}
```

**Response:**

```json
{
  "record_id": "quar_uuid",
  "re_evaluation_status": "requested",
  "timestamp": "2025-12-02T12:10:00Z"
}
```

#### GET /quarantine/statistics

Get statistics about quarantined experiments.

**Response:**

```json
{
  "total_quarantined": 12,
  "can_re_evaluate": 8,
  "permanently_blacklisted": 4,
  "re_evaluation_pending": 2,
  "quarantine_reasons": {
    "low_accuracy": 6,
    "high_latency": 3,
    "high_error_rate": 2,
    "other": 1
  }
}
```

## Error Responses

All endpoints return errors in the following format:

```json
{
  "detail": "Error message",
  "error_id": "1701505200000"
}
```

### Common Status Codes

- `200`: Success
- `400`: Bad request (invalid parameters)
- `404`: Not found
- `500`: Internal server error
- `503`: Service unavailable

## Rate Limiting

Currently not implemented. In production, implement:

- 100 requests per minute per API key
- 10 concurrent requests per API key
- Exponential backoff for retries

## Pagination

List endpoints support pagination with:

- `limit`: Number of items (1-100, default 10)
- `offset`: Starting position (default 0)

Example: `/code-review/reviews?limit=20&offset=40`

## Sorting

Currently not implemented. In production, support:

- `sort_by`: Field to sort by
- `sort_order`: "asc" or "desc"

Example: `/experiments?sort_by=created_at&sort_order=desc`
