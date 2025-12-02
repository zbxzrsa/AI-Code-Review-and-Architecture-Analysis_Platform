-- Initialize PostgreSQL schemas for all three versions

-- ============================================================================
-- V2 Production Schema
-- ============================================================================
CREATE SCHEMA
IF
  NOT EXISTS production;

  CREATE TABLE
  IF
    NOT EXISTS production.code_reviews (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , code_snippet TEXT NOT NULL
      , language VARCHAR(50) NOT NULL
      , issues JSONB DEFAULT '[]': :jsonb
      , suggestions TEXT [] DEFAULT ARRAY []: :TEXT []
      , architecture_insights JSONB DEFAULT '{}': :jsonb
      , security_concerns JSONB DEFAULT '[]': :jsonb
      , performance_notes TEXT [] DEFAULT ARRAY []: :TEXT []
      , confidence_score DECIMAL(3, 2) DEFAULT 0.0
      , analysis_time_ms DECIMAL(10, 2) DEFAULT 0.0
      , model_used VARCHAR(100)
      , created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      , updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
      , INDEX idx_created_at (created_at DESC)
      , INDEX idx_language (
        language
      )
    );

    CREATE TABLE
    IF
      NOT EXISTS production.slo_metrics (
        id UUID PRIMARY KEY DEFAULT gen_random_uuid()
        , metric_name VARCHAR(100) NOT NULL
        , value DECIMAL(10, 2) NOT NULL
        , threshold DECIMAL(10, 2)
        , compliant BOOLEAN DEFAULT true
        , recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        , INDEX idx_metric_name (metric_name)
        , INDEX idx_recorded_at (recorded_at DESC)
      );

      -- ============================================================================
      -- V1 Experimentation Schema
      -- ============================================================================
      CREATE SCHEMA
      IF
        NOT EXISTS experiments_v1;

        CREATE TABLE
        IF
          NOT EXISTS experiments_v1.experiments (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
            , name VARCHAR(255) NOT NULL
            , description TEXT
            , status VARCHAR(50) DEFAULT 'pending'
            , primary_model VARCHAR(100) NOT NULL
            , secondary_model VARCHAR(100)
            , prompt_template TEXT NOT NULL
            , routing_strategy VARCHAR(50) DEFAULT 'primary'
            , created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            , started_at TIMESTAMP
            , completed_at TIMESTAMP
            , promotion_status VARCHAR(50) DEFAULT 'pending_evaluation'
            , promotion_decision_at TIMESTAMP
            , promotion_reason TEXT
            , tags TEXT [] DEFAULT ARRAY []: :TEXT []
            , notes TEXT
            , created_by VARCHAR(100)
            , INDEX idx_status (status)
            , INDEX idx_promotion_status (promotion_status)
            , INDEX idx_created_at (created_at DESC)
          );

          CREATE TABLE
          IF
            NOT EXISTS experiments_v1.experiment_metrics (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid()
              , experiment_id UUID NOT NULL REFERENCES experiments_v1.experiments(id)
              ON DELETE CASCADE
              , accuracy DECIMAL(3, 2)
              , latency_ms DECIMAL(10, 2)
              , cost DECIMAL(10, 2)
              , error_rate DECIMAL(3, 2)
              , throughput INTEGER
              , user_satisfaction DECIMAL(2, 1)
              , false_positives INTEGER
              , false_negatives INTEGER
              , recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
              , INDEX idx_experiment_id (experiment_id)
              , INDEX idx_recorded_at (recorded_at DESC)
            );

            CREATE TABLE
            IF
              NOT EXISTS experiments_v1.code_analyses (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                , experiment_id UUID NOT NULL REFERENCES experiments_v1.experiments(id)
                ON DELETE CASCADE
                , code_snippet TEXT NOT NULL
                , language VARCHAR(50)
                , issues JSONB DEFAULT '[]': :jsonb
                , suggestions TEXT [] DEFAULT ARRAY []: :TEXT []
                , architecture_insights JSONB DEFAULT '{}': :jsonb
                , security_concerns JSONB DEFAULT '[]': :jsonb
                , performance_notes TEXT [] DEFAULT ARRAY []: :TEXT []
                , confidence_score DECIMAL(3, 2)
                , analysis_time_ms DECIMAL(10, 2)
                , model_used VARCHAR(100)
                , created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                , INDEX idx_experiment_id (experiment_id)
                , INDEX idx_created_at (created_at DESC)
              );

              -- ============================================================================
              -- V3 Quarantine Schema (Read-Only)
              -- ============================================================================
              CREATE SCHEMA
              IF
                NOT EXISTS quarantine;

                CREATE TABLE
                IF
                  NOT EXISTS quarantine.quarantine_records (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                    , experiment_id UUID NOT NULL
                    , reason TEXT NOT NULL
                    , failure_analysis JSONB DEFAULT '{}': :jsonb
                    , metrics_at_failure JSONB
                    , quarantined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    , quarantined_by VARCHAR(100)
                    , can_re_evaluate BOOLEAN DEFAULT true
                    , re_evaluation_notes TEXT
                    , re_evaluation_requested_at TIMESTAMP
                    , re_evaluation_requested_by VARCHAR(100)
                    , impact_analysis JSONB DEFAULT '{}': :jsonb
                    , related_experiments UUID [] DEFAULT ARRAY []: :UUID []
                    , INDEX idx_experiment_id (experiment_id)
                    , INDEX idx_quarantined_at (quarantined_at DESC)
                    , INDEX idx_can_re_evaluate (can_re_evaluate)
                  );

                  -- ============================================================================
                  -- Shared Audit Log Schema
                  -- ============================================================================
                  CREATE SCHEMA
                  IF
                    NOT EXISTS
                    audit;

                    CREATE TABLE
                    IF
                      NOT EXISTS
                      audit.event_log (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                        , event_type VARCHAR(100) NOT NULL
                        , version VARCHAR(10) NOT NULL
                        , entity_type VARCHAR(100)
                        , entity_id UUID
                        , user_id VARCHAR(100)
                        , action VARCHAR(50)
                        , details JSONB DEFAULT '{}': :jsonb
                        , timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        , INDEX idx_event_type (event_type)
                        , INDEX idx_version (
                          version
                        )
                        , INDEX idx_timestamp (timestamp DESC)
                      );

                      -- ============================================================================
                      -- Create Indexes for Performance
                      -- ============================================================================
                      CREATE INDEX
                      IF
                        NOT EXISTS idx_production_code_reviews_created_at
                        ON production.code_reviews(created_at DESC);

                        CREATE INDEX
                        IF
                          NOT EXISTS idx_experiments_v1_status
                          ON experiments_v1.experiments(status);

                          CREATE INDEX
                          IF
                            NOT EXISTS idx_experiments_v1_promotion_status
                            ON experiments_v1.experiments(promotion_status);

                            CREATE INDEX
                            IF
                              NOT EXISTS idx_quarantine_records_quarantined_at
                              ON quarantine.quarantine_records(quarantined_at DESC);

                              CREATE INDEX
                              IF
                                NOT EXISTS idx_audit_event_log_timestamp
                                ON
                                audit.event_log(timestamp DESC);

                                -- ============================================================================
                                -- Create Views for Analytics
                                -- ============================================================================
                                CREATE OR REPLACE VIEW experiments_v1.promotion_readiness AS
                                SELECT
                                  e.id
                                  , e.name
                                  , e.status
                                  , m.accuracy
                                  , m.latency_ms
                                  , m.error_rate
                                  , CASE
                                    WHEN
                                      m.accuracy >= 0.95
                                      AND m.latency_ms <= 3000
                                      AND m.error_rate <= 0.02
                                    THEN 'READY'
                                    ELSE 'NOT_READY'
                                  END as promotion_status
                                  , e.created_at
                                FROM
                                  experiments_v1.experiments e
                                  LEFT JOIN experiments_v1.experiment_metrics m
                                ON e.id = m.experiment_id
                                WHERE
                                  e.status = 'completed'
                                ORDER BY
                                  e.created_at DESC;

                                CREATE OR REPLACE VIEW quarantine.quarantine_summary AS
                                SELECT
                                  COUNT(*) as total_quarantined
                                  , SUM(
                                    CASE
                                      WHEN can_re_evaluate THEN 1
                                      ELSE 0
                                    END
                                  ) as can_re_evaluate
                                  , SUM(
                                    CASE
                                      WHEN NOT can_re_evaluate THEN 1
                                      ELSE 0
                                    END
                                  ) as permanently_blacklisted
                                  , SUM(
                                    CASE
                                      WHEN re_evaluation_requested_at IS NOT NULL THEN 1
                                      ELSE 0
                                    END
                                  ) as re_evaluation_pending
                                FROM
                                  quarantine.quarantine_records;

                                -- ============================================================================
                                -- Grant Permissions
                                -- ============================================================================
                                -- V2 Production user (full access to production schema)
                                GRANT
                                  ALL PRIVILEGES
                                ON SCHEMA production
                                TO
                                  platform_user;
                                GRANT
                                  ALL PRIVILEGES
                                ON ALL TABLES IN SCHEMA production
                                TO
                                  platform_user;
                                GRANT
                                  ALL PRIVILEGES
                                ON ALL SEQUENCES IN SCHEMA production
                                TO
                                  platform_user;

                                -- V1 Experimentation user (full access to experiments_v1 schema)
                                GRANT
                                  ALL PRIVILEGES
                                ON SCHEMA experiments_v1
                                TO
                                  platform_user;
                                GRANT
                                  ALL PRIVILEGES
                                ON ALL TABLES IN SCHEMA experiments_v1
                                TO
                                  platform_user;
                                GRANT
                                  ALL PRIVILEGES
                                ON ALL SEQUENCES IN SCHEMA experiments_v1
                                TO
                                  platform_user;

                                -- V3 Quarantine user (read-only access to quarantine schema)
                                GRANT
                                  USAGE
                                ON SCHEMA quarantine
                                TO
                                  platform_readonly;
                                GRANT
                                  SELECT
                                ON ALL TABLES IN SCHEMA quarantine
                                TO
                                  platform_readonly;

                                -- Audit log (append-only for all users)
                                GRANT
                                  USAGE
                                ON SCHEMA
                                audit
                                TO
                                  platform_user;
                                GRANT
                                  SELECT
                                  , INSERT
                                ON ALL TABLES IN SCHEMA
                                audit
                                TO
                                  platform_user;