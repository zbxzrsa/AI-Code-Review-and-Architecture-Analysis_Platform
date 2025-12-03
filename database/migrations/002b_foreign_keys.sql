-- @dialect: postgresql
-- Migration: Foreign Key Constraints
-- Version: 002b
-- Note: This file uses PostgreSQL syntax. IDE warnings can be ignored.

-- Experiments -> Users
ALTER TABLE experiments_v1.experiments
ADD CONSTRAINT fk_experiments_user FOREIGN KEY (user_id) REFERENCES auth.users(id)
ON DELETE SET NULL;

-- Evaluations -> Experiments
ALTER TABLE experiments_v1.evaluations
ADD CONSTRAINT fk_evaluations_experiment FOREIGN KEY (experiment_id) REFERENCES experiments_v1.experiments(id)
ON DELETE CASCADE;

-- Promotions -> Experiments
ALTER TABLE experiments_v1.promotions
ADD CONSTRAINT fk_promotions_experiment FOREIGN KEY (experiment_id) REFERENCES experiments_v1.experiments(id)
ON DELETE CASCADE;

-- Analysis Sessions -> Users
ALTER TABLE production.analysis_sessions
ADD CONSTRAINT fk_sessions_user FOREIGN KEY (user_id) REFERENCES auth.users(id)
ON DELETE SET NULL;

-- Analysis Sessions -> Projects
ALTER TABLE production.analysis_sessions
ADD CONSTRAINT fk_sessions_project FOREIGN KEY (project_id) REFERENCES projects.projects(id)
ON DELETE CASCADE;

-- Quarantine -> Experiments
ALTER TABLE quarantine.quarantined_records
ADD CONSTRAINT fk_quarantine_experiment FOREIGN KEY (source_experiment_id) REFERENCES experiments_v1.experiments(id)
ON DELETE SET NULL;