-- @dialect: postgresql
-- Migration: Performance Indexes
-- Version: 002a
-- Note: Run these commands individually (CONCURRENTLY cannot be in transaction)
-- PostgreSQL 9.5+ required

-- Experiments V1 indexes
CREATE INDEX idx_experiments_v1_status_created
ON experiments_v1.experiments(status, created_at DESC);
CREATE INDEX idx_experiments_v1_user_status
ON experiments_v1.experiments(user_id, status);

-- Production indexes
CREATE INDEX idx_production_user_project
ON production.analysis_sessions(user_id, project_id);
CREATE INDEX idx_production_created_at
ON production.analysis_sessions(created_at DESC);

-- Audit log indexes
CREATE INDEX idx_audit_timestamp
ON audits.audit_log(timestamp DESC);
CREATE INDEX idx_audit_user_action
ON audits.audit_log(user_id, action, timestamp DESC);
CREATE INDEX idx_audit_entity
ON audits.audit_log(entity_type, entity_id);

-- Quarantine indexes
CREATE INDEX idx_quarantine_status
ON quarantine.quarantined_records(status, quarantined_at DESC);

-- Provider indexes
CREATE INDEX idx_provider_health_status
ON providers.provider_health(provider_id, status);
CREATE INDEX idx_provider_usage_date
ON providers.usage_tracking(provider_id, date DESC);