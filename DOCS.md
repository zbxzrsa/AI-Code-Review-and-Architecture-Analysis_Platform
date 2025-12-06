# 项目文档索引 / Documentation Index

## 📁 整理后的文档结构

**整理前**: 47 个 .md 文件（大量冗余）  
**整理后**: 27 个 .md 文件（精简有序）  
**删除文件**: 20+ 个重复/临时文件

---

## 🏠 根目录文档

| 文件                               | 说明         |
| ---------------------------------- | ------------ |
| [README.md](README.md)             | 项目主介绍   |
| [QUICKSTART.md](QUICKSTART.md)     | 快速启动指南 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 贡献指南     |
| [CHANGELOG.md](CHANGELOG.md)       | 版本历史     |
| [DOCS.md](DOCS.md)                 | 本文档索引   |

---

## 📚 技术文档 (`docs/`)

### 架构设计

| 文件                                                                | 说明             |
| ------------------------------------------------------------------- | ---------------- |
| [architecture.md](docs/architecture.md)                             | 系统架构概览     |
| [three-version-architecture.md](docs/three-version-architecture.md) | 三版本自演化架构 |
| [dual-ai-architecture.md](docs/dual-ai-architecture.md)             | 双 AI 协作架构   |
| [microservices.md](docs/microservices.md)                           | 微服务设计       |

### API & 集成

| 文件                                            | 说明         |
| ----------------------------------------------- | ------------ |
| [api-reference.md](docs/api-reference.md)       | API 参考文档 |
| [api-gateway.md](docs/api-gateway.md)           | API 网关配置 |
| [ai-orchestration.md](docs/ai-orchestration.md) | AI 编排系统  |
| [opa-integration.md](docs/opa-integration.md)   | OPA 策略集成 |

### 数据存储

| 文件                                                | 说明                  |
| --------------------------------------------------- | --------------------- |
| [database-design.md](docs/database-design.md)       | PostgreSQL 数据库设计 |
| [redis-caching.md](docs/redis-caching.md)           | Redis 缓存策略        |
| [neo4j-graph-design.md](docs/neo4j-graph-design.md) | Neo4j 图数据库设计    |

### 部署运维

| 文件                                                      | 说明            |
| --------------------------------------------------------- | --------------- |
| [deployment.md](docs/deployment.md)                       | 部署指南        |
| [docker-services.md](docs/docker-services.md)             | Docker 服务配置 |
| [kubernetes-deployment.md](docs/kubernetes-deployment.md) | Kubernetes 部署 |
| [ci-cd-advanced.md](docs/ci-cd-advanced.md)               | CI/CD 流水线    |
| [operations.md](docs/operations.md)                       | 运维手册        |

### 安全 & 日志

| 文件                                      | 说明     |
| ----------------------------------------- | -------- |
| [security.md](docs/security.md)           | 安全配置 |
| [audit-logging.md](docs/audit-logging.md) | 审计日志 |

### 前端 & 测试

| 文件                                        | 说明       |
| ------------------------------------------- | ---------- |
| [frontend-stack.md](docs/frontend-stack.md) | 前端技术栈 |
| [i18n-guide.md](docs/i18n-guide.md)         | 国际化指南 |
| [testing-guide.md](docs/testing-guide.md)   | 测试指南   |

### 三版本演化

| 文件                                                            | 说明     |
| --------------------------------------------------------------- | -------- |
| [three-version-quickstart.md](docs/three-version-quickstart.md) | 快速开始 |
| [three-version-evolution.md](docs/three-version-evolution.md)   | 演化机制 |

---

## 📂 子目录文档

### `docs/deployment/`

- [private-offline-deployment.md](docs/deployment/private-offline-deployment.md) - 私有化离线部署

### `docs/operations/`

- [promotion-rollback-procedures.md](docs/operations/promotion-rollback-procedures.md) - 升级回滚流程

### `frontend/docs/`

- [OPTIMIZATION_GUIDE.md](frontend/docs/OPTIMIZATION_GUIDE.md) - 前端优化指南
- [SECURITY.md](frontend/docs/SECURITY.md) - 前端安全

### 其他 README

- `backend/services/three-version-service/README.md` - 三版本服务
- `charts/coderev-platform/README.md` - Helm Chart
- `data/common-patterns/README.md` - 代码模式库
- `database/migrations/README.md` - 数据库迁移

---

## 🗑️ 已删除的冗余文件

### 临时会话报告（已删除）

- ~~SESSION_SUMMARY.md~~
- ~~COMPLETE_SESSION_SUMMARY.md~~
- ~~FINAL_SESSION_REPORT.md~~
- ~~CODE_QUALITY_FIXES_SESSION.md~~
- ~~REFACTORING_PROGRESS.md~~
- ~~IMPLEMENTATION_SUMMARY.md~~

### 重复报告文件（已删除）

- ~~BUG_FIXES_BATCH_2.md~~
- ~~CODE_QUALITY_FIXES.md~~
- ~~CODE_REVIEW_OPTIMIZATION_REPORT.md~~
- ~~OPTIMIZATION_REPORT.md~~
- ~~PERFORMANCE_QUICK_WINS_IMPLEMENTATION.md~~
- ~~QUICK_WINS.md~~
- ~~QUICK_WINS_BATCH_2.md~~
- ~~SECURITY_FIXES_REPORT.md~~
- ~~PROJECT_AUDIT_REPORT.md~~
- ~~ENTERPRISE_AUDIT_REPORT.md~~

### docs/ 中的重复文件（已删除）

- ~~docs/BUGS_AND_OPTIMIZATIONS.md~~
- ~~docs/CODE_REVIEW_REPORT.md~~
- ~~docs/COMPLETION_REPORT.md~~
- ~~docs/IMPLEMENTATION_COMPLETE.md~~
- ~~docs/IMPLEMENTATION_STATUS.md~~
- ~~docs/PROJECT_OPTIMIZATION_REPORT.md~~
- ~~docs/GRANULAR_CODE_AUDIT.md~~
- ~~docs/THREE_VERSION_IMPLEMENTATION_TRACKER.md~~

### docs/summaries/ 整个目录（已删除）

- 23 个重复的摘要文件

---

## 📊 整理统计

| 指标           | 整理前 | 整理后 | 减少     |
| -------------- | ------ | ------ | -------- |
| **文件数量**   | 47     | 27     | **43%**  |
| **根目录文件** | 24     | 5      | **79%**  |
| **重复内容**   | 20+    | 0      | **100%** |
| **临时报告**   | 6      | 0      | **100%** |

---

## 🎯 文档使用指南

### 新用户

1. 阅读 [README.md](README.md)
2. 按照 [QUICKSTART.md](QUICKSTART.md) 启动项目
3. 深入阅读 [docs/architecture.md](docs/architecture.md)

### 开发者

1. [docs/api-reference.md](docs/api-reference.md) - API 开发
2. [docs/testing-guide.md](docs/testing-guide.md) - 编写测试
3. [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献代码

### 运维人员

1. [docs/deployment.md](docs/deployment.md) - 部署
2. [docs/operations.md](docs/operations.md) - 运维
3. [docs/kubernetes-deployment.md](docs/kubernetes-deployment.md) - K8s

---

**文档已整理完毕！** ✅
