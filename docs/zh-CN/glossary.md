# 术语表 (Glossary)

本文档提供了 AI 代码审查平台中使用的技术术语和专有名词的中英文对照及解释。

---

## A

### Access Token (访问令牌)

用于 API 认证的短期有效令牌，通常有效期为 15 分钟。

### Anomaly Detection (异常检测)

使用机器学习算法自动识别异常行为或数据模式的技术。

### Anchor Point (锚点)

区块链审计中，将一批审计日志的 Merkle 根存储到区块链的操作。

### API Gateway (API 网关)

作为所有 API 请求入口的服务，负责路由、认证、限流等功能。

### Audit Log (审计日志)

记录系统中所有重要操作的不可篡改日志。

---

## B

### Blockchain (区块链)

分布式账本技术，用于存储审计日志的 Merkle 根以确保不可篡改性。

### Bearer Token (持有者令牌)

一种认证方式，在 HTTP 头中以"Bearer {token}"格式传递。

---

## C

### CI/CD (持续集成/持续部署)

Continuous Integration / Continuous Deployment，自动化构建、测试和部署的流程。

### Consensus (共识)

分布式验证中，多个节点对验证结果达成一致的过程。

### Container (容器)

轻量级的应用打包和运行环境，如 Docker 容器。

### CR-AI (代码审查 AI)

Code Review AI，负责执行代码审查的 AI 组件。

---

## D

### Degradation (降级)

将不合格的版本从 V2 生产区移至 V3 隔离区的操作。

### Distributed Verification (分布式验证)

跨多个节点进行审计日志验证的机制。

---

## E

### E2E Testing (端到端测试)

End-to-End Testing，模拟真实用户场景的完整流程测试。

### Ensemble (集成)

多个 AI 模型组合使用以提高准确性的技术。

### Experiment (实验)

在 V1 实验区中运行的新模型或配置测试。

---

## F

### Fallback (故障转移)

当主服务不可用时，自动切换到备用服务的机制。

### Feature Extraction (特征提取)

从原始数据中提取用于机器学习的特征向量。

---

## G

### gRPC

Google 开发的高性能远程过程调用框架。

### Grafana

开源的指标可视化和监控平台。

---

## H

### HPA (水平 Pod 自动扩缩)

Horizontal Pod Autoscaler，Kubernetes 中根据负载自动调整 Pod 数量的组件。

### Health Check (健康检查)

定期检测服务是否正常运行的机制。

---

## I

### Immutable (不可变)

一旦创建就不能修改的特性，审计日志具有此特性。

### Isolation Forest (隔离森林)

一种基于树的异常检测算法。

---

## J

### JWT (JSON Web Token)

一种紧凑的、URL 安全的令牌格式，用于在各方之间安全传输信息。

---

## K

### Kubernetes (K8s)

开源的容器编排平台，用于自动化部署、扩展和管理容器化应用。

---

## L

### Latency (延迟)

从发送请求到收到响应的时间。

### LOF (局部离群因子)

Local Outlier Factor，一种基于密度的异常检测算法。

### Loki

开源的日志聚合系统。

---

## M

### Merkle Tree (默克尔树)

一种哈希树数据结构，用于高效验证大量数据的完整性。

### Microservices (微服务)

将应用拆分为小型、独立服务的架构风格。

### Multi-signature (多重签名)

需要多个签名者批准才能执行敏感操作的安全机制。

---

## N

### Namespace (命名空间)

Kubernetes 中用于隔离资源的逻辑分区。

### Network Policy (网络策略)

Kubernetes 中控制 Pod 间网络流量的规则。

---

## O

### OPA (开放策略代理)

Open Policy Agent，一种通用的策略引擎。

### OpenTelemetry

开源的可观测性框架，用于收集追踪、指标和日志。

---

## P

### P95/P99 (95/99 分位数)

表示 95%或 99%的请求在此时间内完成的延迟指标。

### Pod

Kubernetes 中的最小部署单元，包含一个或多个容器。

### Prometheus

开源的监控和告警系统。

### Promotion (提升)

将实验从 V1 实验区提升到 V2 生产区的操作。

---

## Q

### Quarantine (隔离)

将不合格版本移至 V3 隔离区进行分析和可能的重新评估。

### Quota (配额)

对用户或服务使用量的限制。

---

## R

### RBAC (基于角色的访问控制)

Role-Based Access Control，根据用户角色授予权限的机制。

### Refresh Token (刷新令牌)

用于获取新访问令牌的长期有效令牌。

### Re-evaluation (重新评估)

对 V3 隔离区中的版本进行重新测试和评估的过程。

### RSA-PSS

一种带有概率签名方案的 RSA 数字签名算法。

---

## S

### SLO (服务等级目标)

Service Level Objective，定义服务质量目标的指标。

### SHA256

一种产生 256 位哈希值的加密哈希函数。

### Shadow Traffic (影子流量)

将生产流量复制到实验环境进行测试，不影响用户。

---

## T

### Tempo

开源的分布式追踪后端。

### Three-Version Cycle (三版本循环)

本平台的核心架构，包括 V1 实验、V2 生产和 V3 隔离三个阶段。

---

## U

### Unit Test (单元测试)

对代码最小单元（如函数）进行的测试。

---

## V

### V1 Experiment Zone (V1 实验区)

用于测试新 AI 模型和配置的环境。

### V2 Production Zone (V2 生产区)

面向用户的稳定生产环境。

### V3 Quarantine Zone (V3 隔离区)

存放失败实验进行分析的只读环境。

### VC-AI (版本控制 AI)

Version Control AI，负责管理版本循环的 AI 组件。

---

## W

### Webhook

应用之间通过 HTTP 回调传递事件通知的机制。

### Weight (权重)

在多重签名中，签名者的投票权重。

---

## 缩略语对照表

| 缩写    | 英文全称                                      | 中文                    |
| ------- | --------------------------------------------- | ----------------------- |
| API     | Application Programming Interface             | 应用程序接口            |
| CI/CD   | Continuous Integration/Continuous Deployment  | 持续集成/持续部署       |
| CPU     | Central Processing Unit                       | 中央处理器              |
| DNS     | Domain Name System                            | 域名系统                |
| GPU     | Graphics Processing Unit                      | 图形处理器              |
| HTTP    | HyperText Transfer Protocol                   | 超文本传输协议          |
| JSON    | JavaScript Object Notation                    | JavaScript 对象表示法   |
| JWT     | JSON Web Token                                | JSON 网络令牌           |
| K8s     | Kubernetes                                    | Kubernetes 容器编排平台 |
| ML      | Machine Learning                              | 机器学习                |
| REST    | Representational State Transfer               | 表述性状态转移          |
| SDK     | Software Development Kit                      | 软件开发工具包          |
| SQL     | Structured Query Language                     | 结构化查询语言          |
| SSL/TLS | Secure Sockets Layer/Transport Layer Security | 安全套接字层/传输层安全 |
| URL     | Uniform Resource Locator                      | 统一资源定位符          |
