# API 参考文档

## 概述

本文档提供了 AI 代码审查平台所有 API 端点的详细说明。

**基础 URL**: `https://api.coderev.example.com/v1`

**认证方式**: Bearer Token (JWT)

---

## 认证 API

### 登录

```http
POST /api/auth/login
```

**请求体**:

```json
{
  "email": "user@example.com",
  "password": "your-password"
}
```

**响应**:

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "refresh_token": "eyJhbGciOiJSUzI1NiIs...",
  "token_type": "Bearer",
  "expires_in": 900,
  "user": {
    "id": "user-123",
    "email": "user@example.com",
    "role": "user"
  }
}
```

### 刷新令牌

```http
POST /api/auth/refresh
```

**请求头**:

```
Authorization: Bearer {refresh_token}
```

**响应**:

```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIs...",
  "expires_in": 900
}
```

### 登出

```http
POST /api/auth/logout
```

**请求头**:

```
Authorization: Bearer {access_token}
```

---

## 代码分析 API

### 提交代码分析

```http
POST /api/analyze/code
```

**请求头**:

```
Authorization: Bearer {access_token}
Content-Type: application/json
```

**请求体**:

```json
{
  "code": "def process(data):\n    eval(data)\n    return data",
  "language": "python",
  "rules": ["security", "quality", "performance"],
  "context": {
    "file_path": "src/utils.py",
    "project_id": "proj-123"
  }
}
```

**响应**:

```json
{
  "analysis_id": "analysis-456",
  "status": "completed",
  "issues": [
    {
      "id": "issue-1",
      "type": "security",
      "severity": "critical",
      "message": "使用 eval() 存在代码注入风险",
      "line": 2,
      "column": 5,
      "rule_id": "S001",
      "suggestion": "使用 ast.literal_eval() 或 json.loads() 替代"
    }
  ],
  "metrics": {
    "total_issues": 1,
    "critical": 1,
    "high": 0,
    "medium": 0,
    "low": 0,
    "analysis_time_ms": 250
  }
}
```

### 获取分析结果

```http
GET /api/analyze/{analysis_id}/results
```

**响应**:

```json
{
  "analysis_id": "analysis-456",
  "status": "completed",
  "issues": [...],
  "metrics": {...},
  "created_at": "2024-01-15T10:30:00Z",
  "completed_at": "2024-01-15T10:30:01Z"
}
```

### 应用修复建议

```http
POST /api/fixes/apply
```

**请求体**:

```json
{
  "analysis_id": "analysis-456",
  "issue_id": "issue-1",
  "fix_type": "suggested"
}
```

**响应**:

```json
{
  "success": true,
  "fixed_code": "def process(data):\n    import ast\n    ast.literal_eval(data)\n    return data",
  "diff": "..."
}
```

---

## 三版本控制 API

### 获取版本状态

```http
GET /api/three-version/status
```

**响应**:

```json
{
  "cycle_running": true,
  "current_phase": "monitoring",
  "v1": {
    "status": "experimenting",
    "active_experiments": 3,
    "avg_accuracy": 0.82,
    "pending_promotion": 1
  },
  "v2": {
    "status": "production",
    "requests_24h": 15000,
    "accuracy": 0.91,
    "error_rate": 0.015,
    "latency_p95_ms": 2500
  },
  "v3": {
    "status": "quarantine",
    "archived_count": 5,
    "pending_reevaluation": 1
  }
}
```

### 提升实验到生产

```http
POST /api/three-version/promote
```

**请求体**:

```json
{
  "experiment_id": "exp-123",
  "reason": "准确率达到阈值，错误率符合SLO",
  "metrics": {
    "accuracy": 0.88,
    "error_rate": 0.02,
    "latency_p95_ms": 2800
  }
}
```

**响应**:

```json
{
  "success": true,
  "promotion_id": "promo-789",
  "previous_version": "v2-old",
  "new_version": "v2-new",
  "promoted_at": "2024-01-15T10:30:00Z"
}
```

### 降级到隔离区

```http
POST /api/three-version/demote
```

**请求体**:

```json
{
  "version_id": "v2-current",
  "reason": "错误率超过阈值",
  "evidence": {
    "error_rate": 0.05,
    "incidents": ["INC-001", "INC-002"]
  }
}
```

### 重新评估隔离版本

```http
POST /api/three-version/reevaluate
```

**请求体**:

```json
{
  "quarantine_id": "q-456",
  "new_config": {
    "temperature": 0.5,
    "max_tokens": 2048
  }
}
```

---

## 审计日志 API

### 查询审计日志

```http
GET /api/admin/audit/logs
```

**查询参数**:
| 参数 | 类型 | 描述 |
|------|------|------|
| entity | string | 实体类型 (version, user, experiment) |
| action | string | 操作类型 (create, update, delete, promote) |
| actor_id | string | 操作者 ID |
| from_ts | datetime | 开始时间 |
| to_ts | datetime | 结束时间 |
| limit | integer | 返回数量限制 (默认 100) |

**响应**:

```json
{
  "logs": [
    {
      "id": "log-123",
      "entity": "version",
      "action": "promote",
      "actor_id": "admin-001",
      "resource_id": "exp-123",
      "status": "success",
      "timestamp": "2024-01-15T10:30:00Z",
      "payload": {
        "from": "v1",
        "to": "v2"
      }
    }
  ],
  "total": 150,
  "has_more": true
}
```

### 验证审计日志完整性

```http
POST /api/admin/audit/verify
```

**请求体**:

```json
{
  "from_ts": "2024-01-01T00:00:00Z",
  "to_ts": "2024-01-31T23:59:59Z",
  "entity": "version"
}
```

**响应**:

```json
{
  "valid": true,
  "verified_count": 1500,
  "tampered_count": 0,
  "broken_chains": [],
  "verification_time_ms": 850
}
```

### 导出审计日志

```http
GET /api/admin/audit/export
```

**查询参数**:
| 参数 | 类型 | 描述 |
|------|------|------|
| format | string | 导出格式 (json, csv) |
| from_ts | datetime | 开始时间 |
| to_ts | datetime | 结束时间 |

---

## 多重签名 API

### 获取待审批请求

```http
GET /api/admin/multisig/pending
```

**响应**:

```json
{
  "requests": [
    {
      "request_id": "req-123",
      "operation_type": "version_promotion",
      "requester_id": "admin-001",
      "requester_email": "admin1@example.com",
      "signatures": 1,
      "required_signatures": 2,
      "created_at": "2024-01-15T10:00:00Z",
      "expires_at": "2024-01-16T10:00:00Z",
      "payload_hash": "abc123..."
    }
  ]
}
```

### 签名审批请求

```http
POST /api/admin/multisig/sign/{request_id}
```

**请求体**:

```json
{
  "signature": "base64_encoded_signature",
  "comment": "已审核，同意提升"
}
```

### 拒绝审批请求

```http
POST /api/admin/multisig/reject/{request_id}
```

**请求体**:

```json
{
  "reason": "当前时间不适合进行版本提升"
}
```

---

## 用户管理 API (管理员)

### 获取用户列表

```http
GET /api/admin/users
```

**查询参数**:
| 参数 | 类型 | 描述 |
|------|------|------|
| status | string | 状态筛选 (active, suspended) |
| role | string | 角色筛选 |
| search | string | 搜索邮箱 |
| page | integer | 页码 |
| limit | integer | 每页数量 |

### 暂停用户

```http
POST /api/admin/users/{user_id}/suspend
```

**请求体**:

```json
{
  "reason": "违反服务条款"
}
```

### 重新激活用户

```http
POST /api/admin/users/{user_id}/reactivate
```

### 重置用户密码

```http
POST /api/admin/users/{user_id}/reset-password
```

**响应**:

```json
{
  "temporary_password": "TempPass123!",
  "expires_at": "2024-01-16T10:00:00Z"
}
```

---

## 错误响应

所有 API 在发生错误时返回统一格式:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "请求参数无效",
    "details": [
      {
        "field": "email",
        "message": "邮箱格式不正确"
      }
    ]
  },
  "request_id": "req-abc123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### 错误码列表

| 错误码              | HTTP 状态码 | 描述             |
| ------------------- | ----------- | ---------------- |
| UNAUTHORIZED        | 401         | 未认证或令牌过期 |
| FORBIDDEN           | 403         | 权限不足         |
| NOT_FOUND           | 404         | 资源不存在       |
| VALIDATION_ERROR    | 400         | 请求参数无效     |
| RATE_LIMITED        | 429         | 请求频率超限     |
| INTERNAL_ERROR      | 500         | 服务器内部错误   |
| SERVICE_UNAVAILABLE | 503         | 服务暂时不可用   |

---

## 速率限制

| 端点类型   | 限制         |
| ---------- | ------------ |
| 认证相关   | 10 次/分钟   |
| 代码分析   | 100 次/小时  |
| 一般 API   | 1000 次/小时 |
| 管理员 API | 500 次/小时  |

响应头:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1705318800
```

---

## SDK 示例

### Python SDK

```python
from coderev import Client

client = Client(api_key="your-api-key")

# 分析代码
result = client.analyze(
    code="def foo(): pass",
    language="python",
    rules=["security", "quality"]
)

for issue in result.issues:
    print(f"{issue.severity}: {issue.message}")
```

### JavaScript SDK

```javascript
import { CodeRevClient } from "@coderev/sdk";

const client = new CodeRevClient({ apiKey: "your-api-key" });

const result = await client.analyze({
  code: "function foo() {}",
  language: "javascript",
  rules: ["security", "quality"],
});

result.issues.forEach((issue) => {
  console.log(`${issue.severity}: ${issue.message}`);
});
```
