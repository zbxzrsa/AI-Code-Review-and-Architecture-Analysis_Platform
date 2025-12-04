-- Three-Version Database Schemas
-- V1 (experiments_v1), V2 (production), V3 (quarantine)
-- Flyway migration V001

-- ============================================================
-- SCHEMA: experiments_v1 (V1 Experimentation Zone)
-- Purpose: Rapid trial, shadow traffic, new models/prompts
-- ============================================================

CREATE SCHEMA
IF
  NOT EXISTS experiments_v1;

  -- Shadow traffic analysis results
  CREATE TABLE experiments_v1.shadow_analyses (
    id BIGSERIAL PRIMARY KEY
    , request_id UUID NOT NULL
    , original_request_id UUID
    , -- Links to V2 production request
    -- Version information
    model_version VARCHAR(100) NOT NULL
    , prompt_version VARCHAR(100) NOT NULL
    , routing_policy_version VARCHAR(100) NOT NULL
    , -- Input
      code_hash VARCHAR(64) NOT NULL
    , language VARCHAR(50)
    , file_count INT
    , total_lines INT
    , complexity_score DECIMAL(5, 4)
    , -- Output
      analysis_result JSONB
    , confidence_score DECIMAL(5, 4)
    , issues_found INT DEFAULT 0
    , -- Metrics
      latency_ms INT NOT NULL
    , token_count_input INT
    , token_count_output INT
    , cost_usd DECIMAL(10, 6)
    , -- Comparison with baseline (V2)
      baseline_request_id UUID
    , accuracy_score DECIMAL(5, 4)
    , diff_score DECIMAL(5, 4)
    , -- Difference from baseline
    -- Status
    status VARCHAR(20) DEFAULT 'completed'
    , error_message TEXT
    , created_at TIMESTAMPTZ DEFAULT NOW()
    , -- Indexes
      CONSTRAINT shadow_analyses_status_check CHECK (status IN ('pending', 'completed', 'error'))
  );

  CREATE INDEX idx_shadow_analyses_request
  ON experiments_v1.shadow_analyses(request_id);
  CREATE INDEX idx_shadow_analyses_original
  ON experiments_v1.shadow_analyses(original_request_id);
  CREATE INDEX idx_shadow_analyses_model
  ON experiments_v1.shadow_analyses(model_version);
  CREATE INDEX idx_shadow_analyses_created
  ON experiments_v1.shadow_analyses(created_at);
  CREATE INDEX idx_shadow_analyses_accuracy
  ON experiments_v1.shadow_analyses(accuracy_score);

  -- Experiment configurations being tested
  CREATE TABLE experiments_v1.experiment_configs (
    id BIGSERIAL PRIMARY KEY
    , experiment_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
    , name VARCHAR(200) NOT NULL
    , description TEXT
    , -- Configuration
      model_version VARCHAR(100) NOT NULL
    , prompt_version VARCHAR(100) NOT NULL
    , routing_policy_version VARCHAR(100) NOT NULL
    , parameters JSONB DEFAULT '{}'
    , -- Lifecycle
      state VARCHAR(30) DEFAULT 'created'
    , created_at TIMESTAMPTZ DEFAULT NOW()
    , started_at TIMESTAMPTZ
    , completed_at TIMESTAMPTZ
    , -- Metadata
      created_by VARCHAR(100)
    , tags TEXT []
    , CONSTRAINT experiment_state_check CHECK (
      state IN (
        'created'
        , 'running'
        , 'evaluating'
        , 'passed'
        , 'failed'
        , 'promoted'
        , 'downgraded'
      )
    )
  );

  CREATE INDEX idx_experiment_configs_state
  ON experiments_v1.experiment_configs(state);
  CREATE INDEX idx_experiment_configs_created
  ON experiments_v1.experiment_configs(created_at);

  -- Evaluation results for experiments
  CREATE TABLE experiments_v1.evaluation_results (
    id BIGSERIAL PRIMARY KEY
    , experiment_id UUID NOT NULL REFERENCES experiments_v1.experiment_configs(experiment_id)
    , evaluation_window_start TIMESTAMPTZ NOT NULL
    , evaluation_window_end TIMESTAMPTZ NOT NULL
    , -- Metrics
      total_requests INT NOT NULL
    , p50_latency_ms DECIMAL(10, 2)
    , p95_latency_ms DECIMAL(10, 2)
    , p99_latency_ms DECIMAL(10, 2)
    , error_rate DECIMAL(5, 4)
    , -- Accuracy metrics
      accuracy_score DECIMAL(5, 4)
    , accuracy_delta DECIMAL(5, 4)
    , -- vs baseline
      precision_score DECIMAL(5, 4)
    , recall_score DECIMAL(5, 4)
    , f1_score DECIMAL(5, 4)
    , -- Security metrics
      security_pass_rate DECIMAL(5, 4)
    , security_issues_found INT DEFAULT 0
    , -- Cost metrics
      total_cost_usd DECIMAL(10, 4)
    , avg_cost_per_request DECIMAL(10, 6)
    , cost_delta DECIMAL(5, 4)
    , -- vs baseline
    -- Statistical tests
    accuracy_p_value DECIMAL(10, 8)
    , latency_p_value DECIMAL(10, 8)
    , cost_p_value DECIMAL(10, 8)
    , statistical_significance BOOLEAN DEFAULT FALSE
    , -- Decision
      passed_thresholds BOOLEAN DEFAULT FALSE
    , failed_checks JSONB DEFAULT '[]'
    , created_at TIMESTAMPTZ DEFAULT NOW()
  );

  CREATE INDEX idx_evaluation_results_experiment
  ON experiments_v1.evaluation_results(experiment_id);
  CREATE INDEX idx_evaluation_results_window
  ON experiments_v1.evaluation_results(evaluation_window_start, evaluation_window_end);


  -- ============================================================
  -- SCHEMA: production (V2 Stable Zone)
  -- Purpose: User-facing, strong SLOs, ground-truth baseline
  -- ============================================================

  CREATE SCHEMA
  IF
    NOT EXISTS production;

    -- Production analysis requests
    CREATE TABLE production.analysis_sessions (
      id BIGSERIAL PRIMARY KEY
      , session_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
      , user_id UUID NOT NULL
      , project_id UUID
      , -- Request details
        request_type VARCHAR(50) NOT NULL
      , code_hash VARCHAR(64) NOT NULL
      , language VARCHAR(50)
      , framework VARCHAR(100)
      , file_count INT DEFAULT 1
      , total_lines INT
      , -- Version tracking (baseline)
        model_version VARCHAR(100) NOT NULL
      , prompt_version VARCHAR(100) NOT NULL
      , routing_policy_version VARCHAR(100) NOT NULL
      , -- Response
        analysis_result JSONB
      , issues_found INT DEFAULT 0
      , severity_counts JSONB DEFAULT '{"critical": 0, "high": 0, "medium": 0, "low": 0}'
      , -- Metrics (for SLO tracking)
        latency_ms INT NOT NULL
      , token_count_input INT
      , token_count_output INT
      , cost_usd DECIMAL(10, 6)
      , -- Quality scores
        confidence_score DECIMAL(5, 4)
      , user_feedback_score INT
      , -- 1-5 rating
        feedback_comment TEXT
      , -- Status
        status VARCHAR(20) DEFAULT 'completed'
      , error_message TEXT
      , -- Compliance
        compliance_labels TEXT []
      , audit_required BOOLEAN DEFAULT FALSE
      , created_at TIMESTAMPTZ DEFAULT NOW()
      , completed_at TIMESTAMPTZ
      , CONSTRAINT analysis_status_check CHECK (
        status IN ('pending', 'processing', 'completed', 'error')
      )
    );

    CREATE INDEX idx_production_sessions_user
    ON production.analysis_sessions(user_id);
    CREATE INDEX idx_production_sessions_project
    ON production.analysis_sessions(project_id);
    CREATE INDEX idx_production_sessions_created
    ON production.analysis_sessions(created_at);
    CREATE INDEX idx_production_sessions_status
    ON production.analysis_sessions(status);
    CREATE INDEX idx_production_sessions_model
    ON production.analysis_sessions(model_version);

    -- SLO tracking
    CREATE TABLE production.slo_metrics (
      id BIGSERIAL PRIMARY KEY
      , metric_window_start TIMESTAMPTZ NOT NULL
      , metric_window_end TIMESTAMPTZ NOT NULL
      , -- Volume
        total_requests INT NOT NULL
      , successful_requests INT NOT NULL
      , failed_requests INT NOT NULL
      , -- Latency SLOs
        p50_latency_ms DECIMAL(10, 2)
      , p95_latency_ms DECIMAL(10, 2)
      , p99_latency_ms DECIMAL(10, 2)
      , p95_slo_target_ms INT DEFAULT 3000
      , p95_slo_met BOOLEAN
      , -- Error rate SLO
        error_rate DECIMAL(5, 4)
      , error_rate_slo_target DECIMAL(5, 4) DEFAULT 0.02
      , error_rate_slo_met BOOLEAN
      , -- Availability
        availability DECIMAL(5, 4)
      , availability_slo_target DECIMAL(5, 4) DEFAULT 0.999
      , availability_slo_met BOOLEAN
      , -- Quality
        avg_accuracy DECIMAL(5, 4)
      , avg_user_rating DECIMAL(3, 2)
      , -- Overall
        all_slos_met BOOLEAN
      , created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX idx_slo_metrics_window
    ON production.slo_metrics(metric_window_start);
    CREATE INDEX idx_slo_metrics_slo_met
    ON production.slo_metrics(all_slos_met);

    -- Baseline versions (ground truth)
    CREATE TABLE production.baseline_versions (
      id SERIAL PRIMARY KEY
      , version_id VARCHAR(100) UNIQUE NOT NULL
      , model_version VARCHAR(100) NOT NULL
      , prompt_version VARCHAR(100) NOT NULL
      , routing_policy_version VARCHAR(100) NOT NULL
      , -- Performance baseline
        baseline_accuracy DECIMAL(5, 4)
      , baseline_p95_latency_ms INT
      , baseline_cost_per_request DECIMAL(10, 6)
      , -- Status
        is_active BOOLEAN DEFAULT TRUE
      , promoted_at TIMESTAMPTZ DEFAULT NOW()
      , retired_at TIMESTAMPTZ
      , -- Audit
        promoted_by VARCHAR(100)
      , promotion_reason TEXT
      , created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX idx_baseline_active
    ON production.baseline_versions(is_active);


    -- ============================================================
    -- SCHEMA: quarantine (V3 Legacy/Isolated Zone)
    -- Purpose: Recovery, re-evaluation, parameter repair
    -- ============================================================

    CREATE SCHEMA
    IF
      NOT EXISTS quarantine;

      -- Quarantined versions
      CREATE TABLE quarantine.quarantined_versions (
        id BIGSERIAL PRIMARY KEY
        , quarantine_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
        , -- Original experiment
          original_experiment_id UUID NOT NULL
        , -- Version info
          model_version VARCHAR(100) NOT NULL
        , prompt_version VARCHAR(100) NOT NULL
        , routing_policy_version VARCHAR(100) NOT NULL
        , parameters JSONB DEFAULT '{}'
        , -- Failure details
          quarantine_reason TEXT NOT NULL
        , failed_metrics JSONB NOT NULL
        , failure_timestamp TIMESTAMPTZ NOT NULL
        , -- Recovery tracking
          recovery_attempts INT DEFAULT 0
        , last_recovery_attempt TIMESTAMPTZ
        , recovery_status VARCHAR(30) DEFAULT 'pending'
        , -- Re-evaluation results
          reevaluation_results JSONB
        , -- Status
          status VARCHAR(30) DEFAULT 'quarantined'
        , created_at TIMESTAMPTZ DEFAULT NOW()
        , updated_at TIMESTAMPTZ DEFAULT NOW()
        , CONSTRAINT quarantine_status_check CHECK (
          status IN (
            'quarantined'
            , 're-evaluating'
            , 'recovered'
            , 'archived'
            , 'permanently_failed'
          )
        )
        , CONSTRAINT recovery_status_check CHECK (
          recovery_status IN ('pending', 'in_progress', 'passed', 'failed')
        )
      );

      CREATE INDEX idx_quarantine_status
      ON quarantine.quarantined_versions(status);
      CREATE INDEX idx_quarantine_recovery
      ON quarantine.quarantined_versions(recovery_status);
      CREATE INDEX idx_quarantine_created
      ON quarantine.quarantined_versions(created_at);

      -- Re-evaluation requests
      CREATE TABLE quarantine.reevaluation_requests (
        id BIGSERIAL PRIMARY KEY
        , request_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
        , quarantine_id UUID NOT NULL REFERENCES quarantine.quarantined_versions(quarantine_id)
        , -- Modified parameters for retry
          modified_parameters JSONB
        , modification_reason TEXT
        , -- Evaluation config
          evaluation_type VARCHAR(50) DEFAULT 'gold_set'
        , gold_set_id VARCHAR(100)
        , -- Results
          status VARCHAR(30) DEFAULT 'pending'
        , evaluation_results JSONB
        , passed BOOLEAN
        , -- Audit
          requested_by VARCHAR(100)
        , approved_by VARCHAR(100)
        , created_at TIMESTAMPTZ DEFAULT NOW()
        , completed_at TIMESTAMPTZ
        , CONSTRAINT reevaluation_status_check CHECK (
          status IN (
            'pending'
            , 'approved'
            , 'running'
            , 'completed'
            , 'rejected'
          )
        )
      );

      CREATE INDEX idx_reevaluation_quarantine
      ON quarantine.reevaluation_requests(quarantine_id);
      CREATE INDEX idx_reevaluation_status
      ON quarantine.reevaluation_requests(status);

      -- Low-cost shadow analysis (for comparison/recovery testing)
      CREATE TABLE quarantine.shadow_analyses (
        id BIGSERIAL PRIMARY KEY
        , request_id UUID NOT NULL
        , quarantine_id UUID REFERENCES quarantine.quarantined_versions(quarantine_id)
        , -- Analysis details
          code_hash VARCHAR(64) NOT NULL
        , analysis_result JSONB
        , -- Metrics (minimal tracking)
          latency_ms INT
        , accuracy_score DECIMAL(5, 4)
        , created_at TIMESTAMPTZ DEFAULT NOW()
      );

      CREATE INDEX idx_quarantine_shadow_quarantine
      ON quarantine.shadow_analyses(quarantine_id);


      -- ============================================================
      -- SCHEMA: lifecycle (Cross-version lifecycle tracking)
      -- ============================================================

      CREATE SCHEMA
      IF
        NOT EXISTS lifecycle;

        -- Version lifecycle events (audit trail)
        CREATE TABLE lifecycle.version_events (
          id BIGSERIAL PRIMARY KEY
          , event_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
          , -- Version identification
            version_id VARCHAR(100) NOT NULL
          , experiment_id UUID
          , -- Event details
            event_type VARCHAR(50) NOT NULL
          , from_state VARCHAR(30)
          , to_state VARCHAR(30)
          , -- Context
            triggered_by VARCHAR(100)
          , trigger_reason TEXT
          , metrics_snapshot JSONB
          , -- Audit
            opa_decision_id VARCHAR(100)
          , opa_policy_version VARCHAR(50)
          , created_at TIMESTAMPTZ DEFAULT NOW()
          , CONSTRAINT event_type_check CHECK (
            event_type IN (
              'created'
              , 'shadow_started'
              , 'shadow_completed'
              , 'promotion_requested'
              , 'promotion_approved'
              , 'promotion_denied'
              , 'gray_scale_started'
              , 'gray_scale_advanced'
              , 'gray_scale_completed'
              , 'rollback_triggered'
              , 'downgrade_triggered'
              , 'quarantined'
              , 'recovery_started'
              , 'recovery_completed'
              , 'recovery_failed'
              , 'archived'
              , 'deleted'
            )
          )
        );

        CREATE INDEX idx_lifecycle_events_version
        ON lifecycle.version_events(version_id);
        CREATE INDEX idx_lifecycle_events_type
        ON lifecycle.version_events(event_type);
        CREATE INDEX idx_lifecycle_events_created
        ON lifecycle.version_events(created_at);
        CREATE INDEX idx_lifecycle_events_experiment
        ON lifecycle.version_events(experiment_id);

        -- Promotion history
        CREATE TABLE lifecycle.promotions (
          id BIGSERIAL PRIMARY KEY
          , promotion_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
          , -- Version being promoted
            version_id VARCHAR(100) NOT NULL
          , experiment_id UUID
          , -- Promotion details
            from_environment VARCHAR(20) NOT NULL
          , to_environment VARCHAR(20) NOT NULL
          , promotion_type VARCHAR(30) NOT NULL
          , -- shadow_to_gray, gray_advance, full_rollout
          -- Metrics at promotion
          metrics_snapshot JSONB NOT NULL
          , -- Approval
            approved_by VARCHAR(100)
          , approval_method VARCHAR(30)
          , -- automatic, manual, opa
            opa_decision JSONB
          , -- Status
            status VARCHAR(30) DEFAULT 'pending'
          , created_at TIMESTAMPTZ DEFAULT NOW()
          , completed_at TIMESTAMPTZ
          , CONSTRAINT promotion_status_check CHECK (
            status IN (
              'pending'
              , 'approved'
              , 'in_progress'
              , 'completed'
              , 'rolled_back'
              , 'failed'
            )
          )
        );

        CREATE INDEX idx_promotions_version
        ON lifecycle.promotions(version_id);
        CREATE INDEX idx_promotions_status
        ON lifecycle.promotions(status);
        CREATE INDEX idx_promotions_created
        ON lifecycle.promotions(created_at);

        -- Gold set test results
        CREATE TABLE lifecycle.gold_set_results (
          id BIGSERIAL PRIMARY KEY
          , result_id UUID UNIQUE NOT NULL DEFAULT gen_random_uuid()
          , version_id VARCHAR(100) NOT NULL
          , gold_set_id VARCHAR(100) NOT NULL
          , -- Test categories
            category VARCHAR(50) NOT NULL
          , -- security, injection, long_context, multilingual, etc.
          -- Results
          total_tests INT NOT NULL
          , passed_tests INT NOT NULL
          , failed_tests INT NOT NULL
          , pass_rate DECIMAL(5, 4)
          , -- Details
            failed_test_ids TEXT []
          , detailed_results JSONB
          , created_at TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX idx_gold_set_version
        ON lifecycle.gold_set_results(version_id);
        CREATE INDEX idx_gold_set_category
        ON lifecycle.gold_set_results(category);


        -- ============================================================
        -- CDC Event Triggers (for Kafka/NATS event bus)
        -- ============================================================

        -- Function to emit events
        CREATE OR REPLACE FUNCTION lifecycle.emit_event()
        RETURNS TRIGGER AS $ $
        BEGIN
          -- Insert into event log for CDC pickup
          INSERT INTO
            lifecycle.version_events (
              version_id
              , experiment_id
              , event_type
              , from_state
              , to_state
              , metrics_snapshot
              , created_at
            )
          VALUES
            (
              COALESCE(
                NEW.version_id
                , NEW.experiment_id: :text
                , NEW.id: :text
              )
              , NEW.experiment_id
              , TG_ARGV [0]
              , OLD.state
              , NEW.state
              , to_jsonb(
                NEW
              )
              , NOW()
            );

          -- Notify for real-time CDC
          PERFORM pg_notify(
            'lifecycle_events'
            , json_build_object(
              'event_type'
              , TG_ARGV [0]
              , 'table'
              , TG_TABLE_NAME
              , 'operation'
              , TG_OP
              , 'data'
              , to_jsonb(
                NEW
              )
            ): :text
          );

          RETURN
          NEW;
        END;
        $ $
        LANGUAGE plpgsql;

        -- Trigger on experiment state changes
        CREATE TRIGGER experiment_state_change
        AFTER UPDATE OF state
        ON experiments_v1.experiment_configs
        FOR EACH ROW
      WHEN (
        OLD.state IS DISTINCT FROM
        NEW.state
      ) EXECUTE FUNCTION lifecycle.emit_event('state_changed');

      -- Trigger on quarantine status changes
      CREATE TRIGGER quarantine_status_change
      AFTER UPDATE OF status
      ON quarantine.quarantined_versions
      FOR EACH ROW
      WHEN (
        OLD.status IS DISTINCT FROM
        NEW.status
      ) EXECUTE FUNCTION lifecycle.emit_event('quarantine_status_changed');


      -- ============================================================
      -- Views for Cross-Schema Queries
      -- ============================================================

      -- Current system status view
      CREATE OR REPLACE VIEW lifecycle.system_status AS
      SELECT
        'v1_experiments' as zone
        , state
        , COUNT(*) as count
      FROM
        experiments_v1.experiment_configs
      GROUP BY
        state
      UNION ALL
      SELECT
        'v2_production' as zone
        , CASE
          WHEN is_active THEN 'active'
          ELSE 'retired'
        END as state
        , COUNT(*) as count
      FROM
        production.baseline_versions
      GROUP BY
        is_active
      UNION ALL
      SELECT
        'v3_quarantine' as zone
        , status as state
        , COUNT(*) as count
      FROM
        quarantine.quarantined_versions
      GROUP BY
        status;

      -- Recent promotions view
      CREATE OR REPLACE VIEW lifecycle.recent_promotions AS
      SELECT
        p.promotion_id
        , p.version_id
        , p.from_environment
        , p.to_environment
        , p.promotion_type
        , p.status
        , p.approved_by
        , p.created_at
        , p.completed_at
        , p.metrics_snapshot - > > 'accuracy_delta' as accuracy_delta
        , p.metrics_snapshot - > > 'p95_latency_ms' as p95_latency_ms
      FROM
        lifecycle.promotions p
      ORDER BY
        p.created_at DESC
      LIMIT
        100;


      -- ============================================================
      -- Permissions
      -- ============================================================

      -- V1 Experiment service account
      GRANT
        USAGE
      ON SCHEMA experiments_v1
      TO
        v1_service;
      GRANT
        SELECT
        , INSERT
      , UPDATE
      ON ALL TABLES IN SCHEMA experiments_v1
      TO
        v1_service;
      GRANT
        USAGE
      , SELECT
      ON ALL SEQUENCES IN SCHEMA experiments_v1
      TO
        v1_service;

      -- V2 Production service account (read-heavy)
      GRANT
        USAGE
      ON SCHEMA production
      TO
        v2_service;
      GRANT
        SELECT
        , INSERT
      ON ALL TABLES IN SCHEMA production
      TO
        v2_service;
      GRANT
        UPDATE
      ON production.analysis_sessions
      TO
        v2_service;
      GRANT
        USAGE
      , SELECT
      ON ALL SEQUENCES IN SCHEMA production
      TO
        v2_service;

      -- V3 Quarantine service account (minimal)
      GRANT
        USAGE
      ON SCHEMA quarantine
      TO
        v3_service;
      GRANT
        SELECT
        , INSERT
      , UPDATE
      ON ALL TABLES IN SCHEMA quarantine
      TO
        v3_service;
      GRANT
        USAGE
      , SELECT
      ON ALL SEQUENCES IN SCHEMA quarantine
      TO
        v3_service;

      -- Lifecycle controller (cross-schema access)
      GRANT
        USAGE
      ON SCHEMA experiments_v1
      , production
      , quarantine
      , lifecycle
      TO
        lifecycle_controller;
      GRANT
        SELECT
      ON ALL TABLES IN SCHEMA experiments_v1
      , production
      , quarantine
      TO
        lifecycle_controller;
      GRANT
        SELECT
        , INSERT
      , UPDATE
      ON ALL TABLES IN SCHEMA lifecycle
      TO
        lifecycle_controller;
      GRANT
        USAGE
      , SELECT
      ON ALL SEQUENCES IN SCHEMA lifecycle
      TO
        lifecycle_controller;