-- @dialect: postgresql
-- Migration: System Tables
-- Version: 002c
-- Note: This file uses PostgreSQL syntax. IDE warnings can be ignored.

-- Create system schema
CREATE SCHEMA system;

-- Retention Policies Table
CREATE TABLE system.retention_policies (
  id SERIAL PRIMARY KEY
  , table_schema VARCHAR(64) NOT NULL
  , table_name VARCHAR(128) NOT NULL
  , retention_days INTEGER NOT NULL
  , partition_column VARCHAR(64) DEFAULT 'created_at'
  , enabled BOOLEAN DEFAULT true
  , last_cleanup_at TIMESTAMP WITH TIME ZONE
  , created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  , UNIQUE(table_schema, table_name)
);

-- Default retention policies
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('audits', 'audit_log', 2555);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('experiments_v1', 'experiments', 365);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('experiments_v1', 'evaluations', 365);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('production', 'analysis_sessions', 90);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('production', 'analysis_tasks', 90);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('quarantine', 'quarantined_records', 730);
INSERT INTO
  system.retention_policies (table_schema, table_name, retention_days)
VALUES
  ('providers', 'usage_tracking', 365);

-- Encryption Keys Table
CREATE TABLE system.encryption_keys (
  id SERIAL PRIMARY KEY
  , key_id UUID NOT NULL UNIQUE
  , version INTEGER NOT NULL DEFAULT 1
  , algorithm VARCHAR(32) NOT NULL DEFAULT 'aes-256-gcm'
  , status VARCHAR(16) NOT NULL DEFAULT 'active'
  , created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  , rotated_at TIMESTAMP WITH TIME ZONE
  , retired_at TIMESTAMP WITH TIME ZONE
  , CHECK (status IN ('active', 'rotating', 'retired'))
);

-- Connection Pool Settings Table
CREATE TABLE system.connection_pool_settings (
  id SERIAL PRIMARY KEY
  , pool_name VARCHAR(64) NOT NULL UNIQUE
  , min_connections INTEGER NOT NULL DEFAULT 5
  , max_connections INTEGER NOT NULL DEFAULT 20
  , connection_timeout_ms INTEGER NOT NULL DEFAULT 10000
  , idle_timeout_ms INTEGER NOT NULL DEFAULT 600000
  , max_lifetime_ms INTEGER NOT NULL DEFAULT 1800000
  , updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
  , CHECK (min_connections > 0)
  , CHECK (max_connections >= min_connections)
  , CHECK (max_connections <= 100)
);

-- Default pool settings
INSERT INTO
  system.connection_pool_settings (pool_name, min_connections, max_connections)
VALUES
  ('v2_production', 10, 50);
INSERT INTO
  system.connection_pool_settings (pool_name, min_connections, max_connections)
VALUES
  ('v1_experimentation', 5, 20);
INSERT INTO
  system.connection_pool_settings (pool_name, min_connections, max_connections)
VALUES
  ('v3_quarantine', 2, 10);
INSERT INTO
  system.connection_pool_settings (pool_name, min_connections, max_connections)
VALUES
  ('audit_logging', 3, 15);