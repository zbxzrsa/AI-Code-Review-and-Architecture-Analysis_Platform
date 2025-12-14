# 消息主题与 Schema 说明（Kafka 示例）

## 主题命名（前缀隔离）
- tech.monitor.events
- tech.baseline.report
- tech.candidate
- v1.sandbox.run
- v1.review.findings
- v1.validation.report
- ai.sign.v1 | ai.sign.v3 | ai.sign.v2
- release.candidate
- release.preprod.report
- release.rollout
- metrics.stream | logs.stream | feedback.stream
- release.rollback
- tech.baseline.update

## 版本与兼容性
- 使用 Schema Registry（Avro/JSON），开启兼容策略：向后兼容/完全兼容
- CI 在合并前校验 Schema 变更；破坏性变更需双轨与迁移期

## 示例 Schema 片段（JSON Schema）
```json
{
  "$id": "tech.candidate",
  "type": "object",
  "required": ["id", "name", "source", "metrics", "submittedAt"],
  "properties": {
    "id": {"type": "string"},
    "name": {"type": "string"},
    "source": {"type": "string"},
    "rationale": {"type": "string"},
    "metrics": {
      "type": "object",
      "properties": {
        "latencyImprovementPct": {"type": "number"},
        "throughputImprovementPct": {"type": "number"},
        "resourceDeltaPct": {"type": "number"}
      }
    },
    "submittedAt": {"type": "string", "format": "date-time"}
  }
}
```

