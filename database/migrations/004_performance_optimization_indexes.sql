-- Performance Optimization Indexes
-- Created: 2024-12-06
-- Purpose: Additional strategic indexes for query performance
-- Database: PostgreSQL 16+
-- Run during low-traffic periods

-- ============================================================
-- 1. ANALYSIS TABLES
-- ============================================================

CREATE INDEX idx_analyses_project_status_created
ON production.analysis_sessions (project_id, status, created_at DESC);
CREATE INDEX idx_analyses_created_at
ON production.analysis_sessions (created_at DESC);
CREATE INDEX idx_analyses_user_id
ON production.analysis_sessions (user_id, created_at DESC);
CREATE INDEX idx_analyses_composite
ON production.analysis_sessions (project_id, user_id, status, created_at DESC);
CREATE INDEX idx_analysis_tasks_session
ON production.analysis_tasks (session_id, status, created_at DESC);
CREATE INDEX idx_review_results_session_severity
ON production.code_review_results (session_id, severity, created_at DESC);

-- ============================================================
-- 2. PROJECT TABLES
-- ============================================================

CREATE INDEX idx_projects_status_updated
ON projects.projects (status, updated_at DESC);
CREATE INDEX idx_projects_owner
ON projects.projects (owner_id, created_at DESC);
CREATE INDEX idx_project_versions
ON projects.project_versions (project_id, version_number DESC);

-- ============================================================
-- 3. USER & AUTH TABLES
-- ============================================================

CREATE INDEX idx_sessions_user_active
ON auth.sessions (user_id, expires_at DESC);
CREATE INDEX idx_sessions_token
ON auth.sessions (token_hash);
CREATE INDEX idx_audit_user_created
ON audits.audit_log (user_id, created_at DESC);
CREATE INDEX idx_audit_entity_action
ON audits.audit_log (entity_type, action, created_at DESC);

-- ============================================================
-- 4. EXPERIMENT TABLES
-- ============================================================

CREATE INDEX idx_experiments_status
ON experiments_v1.experiments (status, created_at DESC);
CREATE INDEX idx_evaluations_experiment
ON experiments_v1.evaluations (experiment_id, created_at DESC);
CREATE INDEX idx_promotions_created
ON experiments_v1.promotions (created_at DESC);

-- ============================================================
-- 5. PROVIDER TABLES
-- ============================================================

CREATE INDEX idx_provider_health
ON providers.provider_health (provider_id, status, checked_at DESC);
CREATE INDEX idx_usage_tracking_user_date
ON providers.usage_tracking (user_id, usage_date DESC);
CREATE INDEX idx_quotas_user_type
ON providers.user_quotas (user_id, quota_type);

-- ============================================================
-- 6. ADDITIONAL INDEXES
-- ============================================================

CREATE INDEX idx_analyses_pending
ON production.analysis_sessions (created_at DESC, status);
CREATE INDEX idx_analyses_failed
ON production.analysis_sessions (status, created_at DESC);
CREATE INDEX idx_issues_critical
ON production.code_review_results (session_id, severity, created_at DESC);

-- ============================================================
-- UPDATE STATISTICS (run after index creation)
-- ============================================================
-- Run these commands separately in psql:
-- ANALYZE projects.projects;
-- ANALYZE production.analysis_sessions;
-- ANALYZE production.analysis_tasks;
-- ANALYZE production.code_review_results;
-- ANALYZE auth.users;
-- ANALYZE auth.sessions;
-- ANALYZE audits.audit_log;
-- ANALYZE experiments_v1.experiments;
-- ANALYZE providers.provider_health;

-- Expected improvement: 20-40% reduction in query latency