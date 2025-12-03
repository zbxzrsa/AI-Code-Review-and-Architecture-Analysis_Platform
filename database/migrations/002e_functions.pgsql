-- Migration: PostgreSQL Functions and Triggers
-- Version: 002e
-- Note: This file uses PostgreSQL-specific syntax (plpgsql, dollar quoting)
-- Execute with: psql -f 002e_functions.pgsql

-- Enable pgcrypto extension
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Encrypt data function
CREATE OR REPLACE FUNCTION system.encrypt_data(plaintext TEXT, key_id UUID)
RETURNS BYTEA AS $func$
DECLARE
    encryption_key BYTEA;
BEGIN
    encryption_key := digest(key_id::TEXT, 'sha256');
    RETURN pgp_sym_encrypt(plaintext, encode(encryption_key, 'hex'));
END;
$func$ LANGUAGE plpgsql SECURITY DEFINER;

-- Decrypt data function
CREATE OR REPLACE FUNCTION system.decrypt_data(ciphertext BYTEA, key_id UUID)
RETURNS TEXT AS $func$
DECLARE
    encryption_key BYTEA;
BEGIN
    encryption_key := digest(key_id::TEXT, 'sha256');
    RETURN pgp_sym_decrypt(ciphertext, encode(encryption_key, 'hex'));
END;
$func$ LANGUAGE plpgsql SECURITY DEFINER;

-- Audit trigger function
CREATE OR REPLACE FUNCTION audits.log_sensitive_operation()
RETURNS TRIGGER AS $func$
DECLARE
    v_old_data JSONB;
    v_new_data JSONB;
BEGIN
    IF TG_OP = 'DELETE' THEN
        v_old_data := to_jsonb(OLD);
        v_new_data := NULL;
    ELSIF TG_OP = 'UPDATE' THEN
        v_old_data := to_jsonb(OLD);
        v_new_data := to_jsonb(NEW);
    ELSE
        v_old_data := NULL;
        v_new_data := to_jsonb(NEW);
    END IF;

    -- Mask sensitive fields
    IF v_old_data ? 'password_hash' THEN
        v_old_data := v_old_data - 'password_hash';
    END IF;
    IF v_new_data ? 'password_hash' THEN
        v_new_data := v_new_data - 'password_hash';
    END IF;
    IF v_old_data ? 'api_key' THEN
        v_old_data := jsonb_set(v_old_data, '{api_key}', '"[REDACTED]"');
    END IF;
    IF v_new_data ? 'api_key' THEN
        v_new_data := jsonb_set(v_new_data, '{api_key}', '"[REDACTED]"');
    END IF;

    INSERT INTO audits.audit_log (entity_type, entity_id, action, old_data, new_data, user_id, timestamp)
    VALUES (
        TG_TABLE_SCHEMA || '.' || TG_TABLE_NAME,
        COALESCE(NEW.id::TEXT, OLD.id::TEXT),
        TG_OP,
        v_old_data,
        v_new_data,
        current_setting('app.current_user_id', true),
        CURRENT_TIMESTAMP
    );

    RETURN COALESCE(NEW, OLD);
END;
$func$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit triggers
DROP TRIGGER IF EXISTS audit_users_changes ON auth.users;
CREATE TRIGGER audit_users_changes 
AFTER INSERT OR UPDATE OR DELETE ON auth.users
FOR EACH ROW EXECUTE FUNCTION audits.log_sensitive_operation();

DROP TRIGGER IF EXISTS audit_api_keys_changes ON providers.user_providers;
CREATE TRIGGER audit_api_keys_changes 
AFTER INSERT OR UPDATE OR DELETE ON providers.user_providers
FOR EACH ROW EXECUTE FUNCTION audits.log_sensitive_operation();

DROP TRIGGER IF EXISTS audit_promotions_changes ON experiments_v1.promotions;
CREATE TRIGGER audit_promotions_changes 
AFTER INSERT OR UPDATE OR DELETE ON experiments_v1.promotions
FOR EACH ROW EXECUTE FUNCTION audits.log_sensitive_operation();
