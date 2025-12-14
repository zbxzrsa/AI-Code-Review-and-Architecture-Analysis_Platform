# infra 基线说明

- 作用：提供三版本命名空间、网络隔离、Kafka/Schema Registry、可观测性与发布控制（Argo Rollouts）最小可运行骨架。
- 栈：Kubernetes + Kustomize/Helm；Kafka + Schema Registry；Prometheus/Grafana + Loki + Tempo；ArgoCD + Argo Rollouts。
- 安全：mTLS、NetworkPolicy、命名空间级配额与 RBAC；策略由 Kyverno/OPA 提供。
- 使用：根据环境覆盖对应 values/patches，优先保持 API 兼容与隔离。

