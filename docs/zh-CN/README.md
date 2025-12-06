# AI 代码审查与架构分析平台

## 概述

这是一个基于人工智能的代码审查平台，采用三版本自进化循环架构。该平台能够自动检测代码中的安全漏洞、性能问题和代码质量问题，并提供智能修复建议。

## 核心特性

### 🔄 三版本自进化架构

- **V1 实验区** - 新 AI 模型的测试场，配额放宽，支持快速迭代
- **V2 生产区** - 面向用户的稳定 API，严格 SLO 执行 (p95 < 3 秒, 错误率 < 2%)
- **V3 隔离区** - 失败实验的只读存档，支持重新评估

### 🤖 双 AI 架构

- **VC-AI (版本控制 AI)** - 仅管理员可访问，负责版本管理和模型迭代
- **CR-AI (代码审查 AI)** - 用户访问 V2 版本，提供代码审查服务

### 🔐 安全特性

- 区块链审计日志
- 分布式验证
- 机器学习异常检测
- 多重签名审批
- RSA-PSS 加密签名
- 链式哈希验证

### 📊 监控与可观测性

- Prometheus 指标收集
- Grafana 仪表板
- Loki 日志聚合
- Tempo 分布式追踪
- OpenTelemetry 集成

---

## 快速开始

### 前置要求

- Docker 和 Docker Compose
- Node.js 18+
- Python 3.10+
- PostgreSQL 16
- Redis 7

### 安装步骤

1. **克隆仓库**

```bash
git clone https://github.com/your-org/ai-code-review-platform.git
cd ai-code-review-platform
```

2. **配置环境变量**

```bash
cp .env.example .env
# 编辑 .env 文件，配置必要的API密钥和数据库连接
```

3. **启动服务**

```bash
# 使用 Docker Compose
docker-compose up -d

# 或使用 Makefile
make dev
```

4. **访问应用**

- 前端: http://localhost:3000
- API: http://localhost:8000
- Grafana: http://localhost:3001
- Prometheus: http://localhost:9090

---

## 项目结构

```
├── ai_core/                 # AI核心模块
│   ├── three_version_cycle/ # 三版本循环引擎
│   ├── continuous_learning/ # 持续学习模块
│   └── foundation_model/    # 基础模型
├── backend/
│   ├── app/                 # 主应用
│   ├── services/            # 微服务
│   └── shared/              # 共享代码
├── frontend/                # React前端
├── database/                # 数据库Schema
├── kubernetes/              # K8s部署配置
├── monitoring/              # 监控配置
└── docs/                    # 文档
```

---

## 核心功能

### 代码审查

```python
# 示例：提交代码进行审查
response = await client.analyze_code(
    code=source_code,
    language="python",
    rules=["security", "performance", "quality"]
)

# 获取审查结果
for issue in response.issues:
    print(f"{issue.severity}: {issue.message} (行 {issue.line})")
```

### 版本管理

```python
# 将V1实验提升到V2生产
await version_control.promote(
    experiment_id="exp-123",
    reason="准确率达到阈值",
    approver_id="admin-001"
)
```

### 审计日志

```python
# 记录审计事件
await audit_logger.log_event(
    entity="version",
    action="promote",
    actor_id=admin_id,
    payload={"from": "v1", "to": "v2"}
)

# 验证日志完整性
result = await audit_logger.verify_integrity()
```

---

## API 文档

### 认证

```bash
# 登录获取令牌
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'
```

### 代码分析

```bash
# 提交代码分析请求
curl -X POST http://localhost:8000/api/analyze/code \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"code": "...", "language": "python"}'
```

### 版本控制

```bash
# 获取版本状态
curl http://localhost:8000/api/three-version/status \
  -H "Authorization: Bearer $TOKEN"
```

---

## 部署指南

### Docker 部署

```bash
# 生产环境部署
docker-compose -f docker-compose.yml up -d

# 离线部署
docker-compose -f docker-compose-offline.yml up -d
```

### Kubernetes 部署

```bash
# 使用 Helm
helm install coderev ./charts/coderev-platform \
  -f values-production.yaml \
  --namespace coderev \
  --create-namespace

# 验证部署
kubectl get pods -n coderev
```

---

## 配置说明

### 环境变量

| 变量名              | 描述                  | 默认值                   |
| ------------------- | --------------------- | ------------------------ |
| `DATABASE_URL`      | PostgreSQL 连接字符串 | -                        |
| `REDIS_URL`         | Redis 连接字符串      | `redis://localhost:6379` |
| `OPENAI_API_KEY`    | OpenAI API 密钥       | -                        |
| `ANTHROPIC_API_KEY` | Anthropic API 密钥    | -                        |
| `MOCK_MODE`         | 模拟模式（开发用）    | `false`                  |
| `LOG_LEVEL`         | 日志级别              | `INFO`                   |

### AI 模型配置

```yaml
ai_models:
  primary:
    provider: openai
    model: gpt-4-turbo
    max_tokens: 4096
    temperature: 0.7
  fallback:
    provider: anthropic
    model: claude-3-opus
```

---

## 安全指南

### 最佳实践

1. **API 密钥管理**

   - 使用环境变量或密钥管理服务
   - 定期轮换密钥
   - 不要在代码中硬编码

2. **访问控制**

   - 实施最小权限原则
   - 使用 RBAC 进行权限管理
   - 定期审计访问日志

3. **数据保护**
   - 传输中加密 (TLS 1.3)
   - 静态数据加密 (AES-256)
   - 定期备份

### 多重签名

敏感操作需要多人审批：

```python
# 创建审批请求
request = await multisig.create_request(
    operation_type=OperationType.VERSION_PROMOTION,
    requester_id="admin-001",
    payload={"experiment_id": "exp-123"}
)

# 添加签名
await multisig.add_signature(
    request_id=request.request_id,
    signer_id="admin-002",
    signature=signature_bytes
)
```

---

## 故障排除

### 常见问题

**Q: 服务启动失败**

```bash
# 检查日志
docker-compose logs -f service-name

# 检查健康状态
curl http://localhost:8000/health
```

**Q: 数据库连接失败**

```bash
# 检查PostgreSQL状态
docker-compose ps postgres

# 测试连接
psql $DATABASE_URL -c "SELECT 1"
```

**Q: AI 分析超时**

- 检查 AI 提供商状态
- 调整超时设置
- 切换到备用模型

---

## 贡献指南

### 开发流程

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加新功能'`)
4. 推送分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

### 代码规范

- Python: 使用 `black`, `ruff`, `isort`
- TypeScript: 使用 ESLint + Prettier
- 提交信息遵循 Conventional Commits

### 测试要求

- 单元测试覆盖率 > 95%
- E2E 测试覆盖核心流程
- 通过所有 CI 检查

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](../../LICENSE) 文件。

---

## 联系我们

- 问题反馈: GitHub Issues
- 文档: https://docs.coderev.example.com
- 邮箱: support@coderev.example.com
