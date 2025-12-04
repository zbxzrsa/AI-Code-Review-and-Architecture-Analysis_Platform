# Conftest Policy - Kubernetes Deployment Validation
# Run with: conftest test kubernetes/ --policy policies/kubernetes/

package main

import future.keywords.if
import future.keywords.in

# ============================================================
# Resource Limits Required
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.limits
    msg := sprintf("Container '%s' in Deployment '%s' must have resource limits", [container.name, input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not container.resources.requests
    msg := sprintf("Container '%s' in Deployment '%s' must have resource requests", [container.name, input.metadata.name])
}

# ============================================================
# Security Context Required for Production
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.runAsNonRoot
    msg := sprintf("Container '%s' in V2 production must run as non-root", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    container.securityContext.privileged == true
    msg := sprintf("Container '%s' in V2 production cannot be privileged", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem
    msg := sprintf("Container '%s' in V2 production should have read-only root filesystem", [container.name])
}

# ============================================================
# Priority Class Required
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    not input.spec.template.spec.priorityClassName
    msg := sprintf("Deployment '%s' in V2 must have a priorityClassName", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    input.spec.template.spec.priorityClassName != "production-critical"
    msg := sprintf("Deployment '%s' in V2 must use 'production-critical' priorityClassName", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v1-exp"
    input.spec.template.spec.priorityClassName
    input.spec.template.spec.priorityClassName != "experiment-priority"
    msg := sprintf("Deployment '%s' in V1 must use 'experiment-priority' priorityClassName", [input.metadata.name])
}

# ============================================================
# Image Registry Validation
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    not startswith(container.image, "gcr.io/")
    msg := sprintf("Container '%s' in V2 must use images from gcr.io registry", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    contains(container.image, ":latest")
    msg := sprintf("Container '%s' should not use ':latest' tag", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    container := input.spec.template.spec.containers[_]
    not contains(container.image, ":")
    msg := sprintf("Container '%s' must specify an image tag", [container.name])
}

# ============================================================
# Labels Required
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.version
    msg := sprintf("Deployment '%s' must have 'version' label", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.app
    msg := sprintf("Deployment '%s' must have 'app' label", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    not input.metadata.labels.environment
    msg := sprintf("Deployment '%s' must have 'environment' label", [input.metadata.name])
}

# ============================================================
# Probes Required for Production
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    not container.livenessProbe
    msg := sprintf("Container '%s' in V2 must have a livenessProbe", [container.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    container := input.spec.template.spec.containers[_]
    not container.readinessProbe
    msg := sprintf("Container '%s' in V2 must have a readinessProbe", [container.name])
}

# ============================================================
# Replica Count Validation
# ============================================================

warn[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    input.spec.replicas < 3
    msg := sprintf("Deployment '%s' in V2 should have at least 3 replicas for HA", [input.metadata.name])
}

warn[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v1-exp"
    input.spec.replicas > 10
    msg := sprintf("Deployment '%s' in V1 has high replica count (%d), ensure cost is justified", [input.metadata.name, input.spec.replicas])
}

# ============================================================
# Network Policy Namespace Validation
# ============================================================

deny[msg] {
    input.kind == "NetworkPolicy"
    input.metadata.namespace == "platform-v1-exp"
    some egress in input.spec.egress
    some to in egress.to
    to.namespaceSelector.matchLabels.version == "v2"
    msg := "V1 NetworkPolicy cannot allow egress to V2 namespace"
}

deny[msg] {
    input.kind == "NetworkPolicy"
    input.metadata.namespace == "platform-v3-legacy"
    some egress in input.spec.egress
    some to in egress.to
    to.namespaceSelector.matchLabels.version == "v2"
    msg := "V3 NetworkPolicy cannot allow egress to V2 namespace"
}

# ============================================================
# Service Account Validation
# ============================================================

deny[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    not input.spec.template.spec.serviceAccountName
    msg := sprintf("Deployment '%s' in V2 must specify a serviceAccountName", [input.metadata.name])
}

deny[msg] {
    input.kind == "Deployment"
    input.spec.template.spec.serviceAccountName == "default"
    msg := sprintf("Deployment '%s' should not use 'default' serviceAccount", [input.metadata.name])
}

# ============================================================
# Pod Disruption Budget Required for Production
# ============================================================

warn[msg] {
    input.kind == "Deployment"
    input.metadata.namespace == "platform-v2-stable"
    not has_pdb(input.metadata.name)
    msg := sprintf("Deployment '%s' in V2 should have a PodDisruptionBudget", [input.metadata.name])
}

# Helper function - would check for PDB existence
has_pdb(deployment_name) = false

# ============================================================
# HPA Validation
# ============================================================

deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.metadata.namespace == "platform-v2-stable"
    input.spec.minReplicas < 3
    msg := sprintf("HPA '%s' in V2 must have minReplicas >= 3", [input.metadata.name])
}

deny[msg] {
    input.kind == "HorizontalPodAutoscaler"
    input.metadata.namespace == "platform-v3-legacy"
    input.spec.maxReplicas > 5
    msg := sprintf("HPA '%s' in V3 should have maxReplicas <= 5 to control costs", [input.metadata.name])
}

# ============================================================
# Secret Validation
# ============================================================

deny[msg] {
    input.kind == "Secret"
    input.type == "Opaque"
    input.metadata.namespace in ["platform-v1-exp", "platform-v2-stable", "platform-v3-legacy"]
    msg := sprintf("Use SealedSecret or ExternalSecret instead of plain Secret '%s'", [input.metadata.name])
}

# ============================================================
# ConfigMap Validation
# ============================================================

warn[msg] {
    input.kind == "ConfigMap"
    data_value := input.data[key]
    contains(lower(key), "password")
    msg := sprintf("ConfigMap '%s' key '%s' may contain sensitive data - use Secret instead", [input.metadata.name, key])
}

warn[msg] {
    input.kind == "ConfigMap"
    data_value := input.data[key]
    contains(lower(key), "api_key")
    msg := sprintf("ConfigMap '%s' key '%s' may contain sensitive data - use Secret instead", [input.metadata.name, key])
}
