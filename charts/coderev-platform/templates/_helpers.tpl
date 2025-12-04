{{/*
Expand the name of the chart.
*/}}
{{- define "coderev.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "coderev.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "coderev.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "coderev.labels" -}}
helm.sh/chart: {{ include "coderev.chart" . }}
{{ include "coderev.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "coderev.selectorLabels" -}}
app.kubernetes.io/name: {{ include "coderev.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "coderev.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "coderev.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Version-specific labels
*/}}
{{- define "coderev.versionLabels" -}}
{{- $version := .version -}}
version: {{ $version }}
environment: {{ if eq $version "v1" }}experiment{{ else if eq $version "v2" }}production{{ else }}legacy{{ end }}
tier: platform
{{- end }}

{{/*
Get namespace for version
*/}}
{{- define "coderev.namespace" -}}
{{- $version := .version -}}
{{- if eq $version "v1" }}
{{- .root.Values.versions.v1.namespace }}
{{- else if eq $version "v2" }}
{{- .root.Values.versions.v2.namespace }}
{{- else }}
{{- .root.Values.versions.v3.namespace }}
{{- end }}
{{- end }}

{{/*
Get replicas for version
*/}}
{{- define "coderev.replicas" -}}
{{- $version := .version -}}
{{- if eq $version "v1" }}
{{- .root.Values.versions.v1.replicas }}
{{- else if eq $version "v2" }}
{{- .root.Values.versions.v2.replicas }}
{{- else }}
{{- .root.Values.versions.v3.replicas }}
{{- end }}
{{- end }}

{{/*
Get resources for version
*/}}
{{- define "coderev.resources" -}}
{{- $version := .version -}}
{{- if eq $version "v1" }}
{{- toYaml .root.Values.versions.v1.resources }}
{{- else if eq $version "v2" }}
{{- toYaml .root.Values.versions.v2.resources }}
{{- else }}
{{- toYaml .root.Values.versions.v3.resources }}
{{- end }}
{{- end }}

{{/*
Image name with registry
*/}}
{{- define "coderev.image" -}}
{{- $registry := .root.Values.global.imageRegistry -}}
{{- $repository := .image.repository -}}
{{- $tag := .image.tag | default .root.Chart.AppVersion -}}
{{- printf "%s/%s:%s" $registry $repository $tag }}
{{- end }}

{{/*
Database URL
*/}}
{{- define "coderev.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
{{- printf "postgresql://%s@%s-postgresql:5432/%s" .Values.postgresql.auth.username (include "coderev.fullname" .) .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.externalDatabase.url }}
{{- end }}
{{- end }}

{{/*
Redis URL
*/}}
{{- define "coderev.redisUrl" -}}
{{- if .Values.redis.enabled }}
{{- printf "redis://%s-redis-master:6379" (include "coderev.fullname" .) }}
{{- else }}
{{- .Values.externalRedis.url }}
{{- end }}
{{- end }}
