-- ============================================
-- Schema: audits
-- Immutable audit trail with cryptographic chaining
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS audits;

  -- Audit log table (partitioned by month)
  CREATE TABLE
  IF
    NOT EXISTS audits.audit_log (
      id BIGSERIAL
      , entity VARCHAR(100) NOT NULL
      , action VARCHAR(50) NOT NULL
      , actor_id UUID REFERENCES auth.users(id)
      ON DELETE SET NULL
      , payload JSONB NOT NULL
      , signature TEXT NOT NULL
      , -- Cryptographic signature (SHA256)
        prev_hash TEXT
      , -- Hash of previous log entry (chain)
        ts TIMESTAMPTZ NOT NULL DEFAULT NOW()
      , PRIMARY KEY (id, ts)
    )
    PARTITION BY RANGE (ts);

    -- Create monthly partitions for 2024 and 2025
    CREATE TABLE
    IF
      NOT EXISTS audits.audit_log_2024_12
      PARTITION OF audits.audit_log
      FOR
      VALUES
      FROM
        ('2024-12-01')
      TO
        ('2025-01-01');

      CREATE TABLE
      IF
        NOT EXISTS audits.audit_log_2025_01
        PARTITION OF audits.audit_log
        FOR
        VALUES
        FROM
          ('2025-01-01')
        TO
          ('2025-02-01');

        CREATE TABLE
        IF
          NOT EXISTS audits.audit_log_2025_02
          PARTITION OF audits.audit_log
          FOR
          VALUES
          FROM
            ('2025-02-01')
          TO
            ('2025-03-01');

          CREATE TABLE
          IF
            NOT EXISTS audits.audit_log_2025_03
            PARTITION OF audits.audit_log
            FOR
            VALUES
            FROM
              ('2025-03-01')
            TO
              ('2025-04-01');

            CREATE TABLE
            IF
              NOT EXISTS audits.audit_log_2025_04
              PARTITION OF audits.audit_log
              FOR
              VALUES
              FROM
                ('2025-04-01')
              TO
                ('2025-05-01');

              CREATE TABLE
              IF
                NOT EXISTS audits.audit_log_2025_05
                PARTITION OF audits.audit_log
                FOR
                VALUES
                FROM
                  ('2025-05-01')
                TO
                  ('2025-06-01');

                CREATE TABLE
                IF
                  NOT EXISTS audits.audit_log_2025_06
                  PARTITION OF audits.audit_log
                  FOR
                  VALUES
                  FROM
                    ('2025-06-01')
                  TO
                    ('2025-07-01');

                  CREATE TABLE
                  IF
                    NOT EXISTS audits.audit_log_2025_07
                    PARTITION OF audits.audit_log
                    FOR
                    VALUES
                    FROM
                      ('2025-07-01')
                    TO
                      ('2025-08-01');

                    CREATE TABLE
                    IF
                      NOT EXISTS audits.audit_log_2025_08
                      PARTITION OF audits.audit_log
                      FOR
                      VALUES
                      FROM
                        ('2025-08-01')
                      TO
                        ('2025-09-01');

                      CREATE TABLE
                      IF
                        NOT EXISTS audits.audit_log_2025_09
                        PARTITION OF audits.audit_log
                        FOR
                        VALUES
                        FROM
                          ('2025-09-01')
                        TO
                          ('2025-10-01');

                        CREATE TABLE
                        IF
                          NOT EXISTS audits.audit_log_2025_10
                          PARTITION OF audits.audit_log
                          FOR
                          VALUES
                          FROM
                            ('2025-10-01')
                          TO
                            ('2025-11-01');

                          CREATE TABLE
                          IF
                            NOT EXISTS audits.audit_log_2025_11
                            PARTITION OF audits.audit_log
                            FOR
                            VALUES
                            FROM
                              ('2025-11-01')
                            TO
                              ('2025-12-01');

                            CREATE TABLE
                            IF
                              NOT EXISTS audits.audit_log_2025_12
                              PARTITION OF audits.audit_log
                              FOR
                              VALUES
                              FROM
                                ('2025-12-01')
                              TO
                                ('2026-01-01');

                              -- Indexes on audit log
                              CREATE INDEX
                              IF
                                NOT EXISTS idx_audit_entity
                                ON audits.audit_log(entity);
                                CREATE INDEX
                                IF
                                  NOT EXISTS idx_audit_actor
                                  ON audits.audit_log(actor_id);
                                  CREATE INDEX
                                  IF
                                    NOT EXISTS idx_audit_ts
                                    ON audits.audit_log(ts);
                                    CREATE INDEX
                                    IF
                                      NOT EXISTS idx_audit_action
                                      ON audits.audit_log(action);

                                      -- Audit log chain validation function
                                      CREATE OR REPLACE FUNCTION audits.validate_audit_chain()
                                      RETURNS TRIGGER AS $ $
                                      DECLARE prev_record RECORD;
                                      BEGIN
                                        -- Get the previous log entry
                                        SELECT
                                          signature
                                        INTO
                                          prev_record
                                        FROM
                                          audits.audit_log
                                        WHERE
                                          ts <
                                          NEW.ts
                                        ORDER BY
                                          ts DESC
                                          , id DESC
                                        LIMIT
                                          1;

                                        -- If there's a previous record and prev_hash doesn't match, raise error
                                        IF
                                          prev_record.signature IS NOT NULL
                                          AND
                                          NEW.prev_hash != prev_record.signature
                                        THEN
                                          RAISE EXCEPTION 'Audit chain broken: invalid prev_hash';
                                        END IF;

                                        RETURN
                                        NEW;
                                      END;
                                      $ $
                                      LANGUAGE plpgsql;

                                      -- Trigger for audit chain validation
                                      CREATE TRIGGER validate_audit_chain_trigger
                                      BEFORE INSERT
                                      ON audits.audit_log
                                      FOR EACH ROW EXECUTE FUNCTION audits.validate_audit_chain();

                                      -- Audit log immutability function
                                      CREATE OR REPLACE FUNCTION audits.prevent_audit_modification()
                                      RETURNS TRIGGER AS $ $
                                      BEGIN
                                        RAISE EXCEPTION 'Audit logs are immutable and cannot be modified or deleted';
                                      END;
                                      $ $
                                      LANGUAGE plpgsql;

                                      -- Triggers to prevent modification
                                      CREATE TRIGGER prevent_audit_update
                                      BEFORE UPDATE
                                      ON audits.audit_log
                                      FOR EACH ROW EXECUTE FUNCTION audits.prevent_audit_modification();

                                      CREATE TRIGGER prevent_audit_delete
                                      BEFORE DELETE
                                      ON audits.audit_log
                                      FOR EACH ROW EXECUTE FUNCTION audits.prevent_audit_modification();

                                      -- Audit log statistics view
                                      CREATE OR REPLACE VIEW audits.statistics AS
                                      SELECT
                                        entity
                                        , action
                                        , COUNT(*) as count
                                        , COUNT(DISTINCT actor_id) as unique_actors
                                        , MIN(ts) as first_occurrence
                                        , MAX(ts) as last_occurrence
                                        , DATE_TRUNC('day', ts) as day
                                      FROM
                                        audits.audit_log
                                      GROUP BY
                                        entity
                                        , action
                                        , DATE_TRUNC('day', ts);

                                      -- Grant permissions (read-only for audit logs)
                                      GRANT
                                        USAGE
                                      ON SCHEMA audits
                                      TO
                                        PUBLIC;
                                      GRANT
                                        SELECT
                                      ON ALL TABLES IN SCHEMA audits
                                      TO
                                        PUBLIC;
                                      GRANT
                                        SELECT
                                      ON ALL VIEWS IN SCHEMA audits
                                      TO
                                        PUBLIC;