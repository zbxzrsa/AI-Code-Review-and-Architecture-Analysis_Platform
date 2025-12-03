# Database Migrations

## Migration 002: Security and Performance

This migration has been split into multiple files for better IDE compatibility and easier maintenance.

### Files

| File                     | Description                       | IDE Compatible         |
| ------------------------ | --------------------------------- | ---------------------- |
| `002a_indexes.sql`       | Performance indexes               | ✅ Yes                 |
| `002b_foreign_keys.sql`  | Foreign key constraints           | ✅ Yes                 |
| `002c_system_tables.sql` | System configuration tables       | ✅ Yes                 |
| `002d_roles_grants.sql`  | Database roles and permissions    | ✅ Yes                 |
| `002e_functions.pgsql`   | PostgreSQL functions and triggers | ⚠️ PostgreSQL-specific |

### Execution Order

```bash
# Run in order using psql
psql -d your_database -f 002a_indexes.sql
psql -d your_database -f 002b_foreign_keys.sql
psql -d your_database -f 002c_system_tables.sql
psql -d your_database -f 002d_roles_grants.sql
psql -d your_database -f 002e_functions.pgsql
```

### IDE Compatibility Notes

The `002e_functions.pgsql` file uses PostgreSQL-specific syntax:

- Dollar-quoted strings (`$func$...$func$`)
- PL/pgSQL functions
- Triggers

Most SQL linters don't support these PostgreSQL extensions. The `.pgsql` extension helps IDEs identify it as PostgreSQL-specific code.

### What Each File Does

#### 002a_indexes.sql

Creates 11 performance indexes for frequently queried columns:

- Experiments status and timestamps
- Production analysis sessions
- Audit log lookups
- Provider health tracking

#### 002b_foreign_keys.sql

Adds referential integrity constraints:

- Experiments → Users
- Evaluations → Experiments
- Analysis Sessions → Users, Projects
- Quarantine → Experiments

#### 002c_system_tables.sql

Creates system configuration tables:

- `system.retention_policies` - Data retention rules
- `system.encryption_keys` - Key management
- `system.connection_pool_settings` - Pool configuration

#### 002d_roles_grants.sql

Creates database roles with appropriate permissions:

- `app_readonly` - SELECT only
- `app_readwrite` - SELECT, INSERT, UPDATE
- `app_admin` - Full access

#### 002e_functions.pgsql

PostgreSQL-specific code:

- Encryption/decryption functions using pgcrypto
- Audit logging trigger function
- Triggers on sensitive tables

### Rollback

To rollback this migration:

```sql
-- Drop triggers
DROP TRIGGER IF EXISTS audit_users_changes ON auth.users;
DROP TRIGGER IF EXISTS audit_api_keys_changes ON providers.user_providers;
DROP TRIGGER IF EXISTS audit_promotions_changes ON experiments_v1.promotions;

-- Drop functions
DROP FUNCTION IF EXISTS audits.log_sensitive_operation();
DROP FUNCTION IF EXISTS system.encrypt_data(TEXT, UUID);
DROP FUNCTION IF EXISTS system.decrypt_data(BYTEA, UUID);

-- Drop roles (careful - may affect existing users)
-- REVOKE ALL FROM app_admin, app_readwrite, app_readonly;
-- DROP ROLE IF EXISTS app_admin, app_readwrite, app_readonly;

-- Drop system tables
DROP TABLE IF EXISTS system.connection_pool_settings;
DROP TABLE IF EXISTS system.encryption_keys;
DROP TABLE IF EXISTS system.retention_policies;
DROP SCHEMA IF EXISTS system;

-- Drop foreign keys
ALTER TABLE experiments_v1.experiments DROP CONSTRAINT IF EXISTS fk_experiments_user;
ALTER TABLE experiments_v1.evaluations DROP CONSTRAINT IF EXISTS fk_evaluations_experiment;
ALTER TABLE experiments_v1.promotions DROP CONSTRAINT IF EXISTS fk_promotions_experiment;
ALTER TABLE production.analysis_sessions DROP CONSTRAINT IF EXISTS fk_sessions_user;
ALTER TABLE production.analysis_sessions DROP CONSTRAINT IF EXISTS fk_sessions_project;
ALTER TABLE quarantine.quarantined_records DROP CONSTRAINT IF EXISTS fk_quarantine_experiment;

-- Drop indexes
DROP INDEX IF EXISTS experiments_v1.idx_experiments_v1_status_created;
DROP INDEX IF EXISTS experiments_v1.idx_experiments_v1_user_status;
DROP INDEX IF EXISTS production.idx_production_user_project;
DROP INDEX IF EXISTS production.idx_production_created_at;
DROP INDEX IF EXISTS audits.idx_audit_timestamp;
DROP INDEX IF EXISTS audits.idx_audit_user_action;
DROP INDEX IF EXISTS audits.idx_audit_entity;
DROP INDEX IF EXISTS quarantine.idx_quarantine_status;
DROP INDEX IF EXISTS providers.idx_provider_health_status;
DROP INDEX IF EXISTS providers.idx_provider_usage_date;
```
