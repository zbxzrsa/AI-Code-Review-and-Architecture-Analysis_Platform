-- @dialect: postgresql
-- Migration: Database Roles and Permissions
-- Version: 002d
-- Note: This file uses PostgreSQL syntax. IDE warnings can be ignored.

-- Create roles
CREATE ROLE app_readonly;
CREATE ROLE app_readwrite;
CREATE ROLE app_admin;

-- Readonly permissions
GRANT
  USAGE
ON SCHEMA production
TO
  app_readonly;
GRANT
  USAGE
ON SCHEMA experiments_v1
TO
  app_readonly;
GRANT
  USAGE
ON SCHEMA quarantine
TO
  app_readonly;

GRANT
  SELECT
ON ALL TABLES IN SCHEMA production
TO
  app_readonly;
GRANT
  SELECT
ON ALL TABLES IN SCHEMA experiments_v1
TO
  app_readonly;
GRANT
  SELECT
ON ALL TABLES IN SCHEMA quarantine
TO
  app_readonly;
GRANT
  SELECT
ON ALL TABLES IN SCHEMA audits
TO
  app_readonly;

-- Readwrite permissions
GRANT
  USAGE
ON SCHEMA production
TO
  app_readwrite;
GRANT
  USAGE
ON SCHEMA experiments_v1
TO
  app_readwrite;

GRANT
  SELECT
  , INSERT
, UPDATE
ON ALL TABLES IN SCHEMA production
TO
  app_readwrite;
GRANT
  SELECT
  , INSERT
, UPDATE
ON ALL TABLES IN SCHEMA experiments_v1
TO
  app_readwrite;
GRANT
  USAGE
ON ALL SEQUENCES IN SCHEMA production
TO
  app_readwrite;
GRANT
  USAGE
ON ALL SEQUENCES IN SCHEMA experiments_v1
TO
  app_readwrite;

-- Admin permissions (inherits readonly and readwrite)
GRANT
  app_readonly
TO
  app_admin;
GRANT
  app_readwrite
TO
  app_admin;

GRANT
  ALL
ON SCHEMA production
TO
  app_admin;
GRANT
  ALL
ON SCHEMA experiments_v1
TO
  app_admin;
GRANT
  ALL
ON SCHEMA quarantine
TO
  app_admin;
GRANT
  ALL
ON SCHEMA audits
TO
  app_admin;
GRANT
  ALL
ON SCHEMA providers
TO
  app_admin;
GRANT
  ALL
ON SCHEMA projects
TO
  app_admin;
GRANT
  ALL
ON SCHEMA auth
TO
  app_admin;

GRANT
  ALL
ON ALL TABLES IN SCHEMA production
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA experiments_v1
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA quarantine
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA audits
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA providers
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA projects
TO
  app_admin;
GRANT
  ALL
ON ALL TABLES IN SCHEMA auth
TO
  app_admin;

GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA production
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA experiments_v1
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA quarantine
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA audits
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA providers
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA projects
TO
  app_admin;
GRANT
  ALL
ON ALL SEQUENCES IN SCHEMA auth
TO
  app_admin;