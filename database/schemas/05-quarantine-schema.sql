-- ============================================
-- Schema: quarantine
-- V3 Quarantine zone data (read-only archive)
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS quarantine;

  -- Quarantine records table
  CREATE TABLE
  IF
    NOT EXISTS quarantine.quarantine_records (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , experiment_id UUID NOT NULL
      , config_hash VARCHAR(64) UNIQUE NOT NULL
      , config JSONB NOT NULL
      , reason TEXT NOT NULL
      , evidence JSONB NOT NULL
      , failure_metrics JSONB
      , quarantined_at TIMESTAMPTZ DEFAULT NOW()
      , quarantined_by UUID REFERENCES auth.users(id)
      , review_status VARCHAR(50) NOT NULL CHECK (
        review_status IN ('quarantined', 'under_review', 'approved_retry')
      )
      , reviewed_at TIMESTAMPTZ
      , reviewed_by UUID REFERENCES auth.users(id)
      , review_notes TEXT
      , created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_quarantine_config_hash
      ON quarantine.quarantine_records(config_hash);
      CREATE INDEX
      IF
        NOT EXISTS idx_quarantine_review_status
        ON quarantine.quarantine_records(review_status);
        CREATE INDEX
        IF
          NOT EXISTS idx_quarantine_quarantined_at
          ON quarantine.quarantine_records(quarantined_at);

          -- Re-evaluation requests table
          CREATE TABLE
          IF
            NOT EXISTS quarantine.reevaluation_requests (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid()
              , quarantine_id UUID NOT NULL REFERENCES quarantine.quarantine_records(id)
              ON DELETE CASCADE
              , requested_by UUID NOT NULL REFERENCES auth.users(id)
              , reason TEXT NOT NULL
              , status VARCHAR(50) NOT NULL CHECK (
                status IN ('pending', 'approved', 'rejected', 'completed')
              )
              , approved_by UUID REFERENCES auth.users(id)
              , approved_at TIMESTAMPTZ
              , completed_at TIMESTAMPTZ
              , result JSONB
              , created_at TIMESTAMPTZ DEFAULT NOW()
              , updated_at TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX
            IF
              NOT EXISTS idx_reevaluation_quarantine
              ON quarantine.reevaluation_requests(quarantine_id);
              CREATE INDEX
              IF
                NOT EXISTS idx_reevaluation_status
                ON quarantine.reevaluation_requests(status);
                CREATE INDEX
                IF
                  NOT EXISTS idx_reevaluation_requested_by
                  ON quarantine.reevaluation_requests(requested_by);

                  -- Quarantine statistics view
                  CREATE OR REPLACE VIEW quarantine.statistics AS
                  SELECT
                    COUNT(*) as total_quarantined
                    , COUNT(
                      CASE
                        WHEN review_status = 'quarantined' THEN 1
                      END
                    ) as pending_review
                    , COUNT(
                      CASE
                        WHEN review_status = 'under_review' THEN 1
                      END
                    ) as under_review
                    , COUNT(
                      CASE
                        WHEN review_status = 'approved_retry' THEN 1
                      END
                    ) as approved_retry
                    , DATE_TRUNC('month', quarantined_at) as month
                  FROM
                    quarantine.quarantine_records
                  GROUP BY
                    DATE_TRUNC('month', quarantined_at);

                  -- Grant permissions (read-only for quarantine)
                  GRANT
                    USAGE
                  ON SCHEMA quarantine
                  TO
                    PUBLIC;
                  GRANT
                    SELECT
                  ON ALL TABLES IN SCHEMA quarantine
                  TO
                    PUBLIC;
                  GRANT
                    SELECT
                  ON ALL VIEWS IN SCHEMA quarantine
                  TO
                    PUBLIC;