-- ============================================
-- Schema: experiments_v1
-- V1 Experimentation zone data
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS experiments_v1;

  -- Experiments table
  CREATE TABLE
  IF
    NOT EXISTS experiments_v1.experiments (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , name VARCHAR(255) NOT NULL
      , description TEXT
      , version VARCHAR(10) DEFAULT 'v1' CHECK (
        version = 'v1'
      )
      , config JSONB NOT NULL
      , dataset_id UUID NOT NULL
      , status VARCHAR(20) NOT NULL CHECK (
        status IN (
          'created'
          , 'running'
          , 'completed'
          , 'failed'
          , 'evaluating'
          , 'promoted'
          , 'quarantined'
        )
      )
      , created_by UUID NOT NULL REFERENCES auth.users(id)
      , created_at TIMESTAMPTZ DEFAULT NOW()
      , updated_at TIMESTAMPTZ DEFAULT NOW()
      , started_at TIMESTAMPTZ
      , completed_at TIMESTAMPTZ
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_experiments_status
      ON experiments_v1.experiments(status);
      CREATE INDEX
      IF
        NOT EXISTS idx_experiments_created_by
        ON experiments_v1.experiments(created_by);
        CREATE INDEX
        IF
          NOT EXISTS idx_experiments_dataset
          ON experiments_v1.experiments(dataset_id);
          CREATE INDEX
          IF
            NOT EXISTS idx_experiments_created
            ON experiments_v1.experiments(created_at);

            -- Evaluations table
            CREATE TABLE
            IF
              NOT EXISTS experiments_v1.evaluations (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                , experiment_id UUID NOT NULL REFERENCES experiments_v1.experiments(id)
                ON DELETE CASCADE
                , metrics JSONB NOT NULL
                , ai_verdict VARCHAR(50) NOT NULL CHECK (ai_verdict IN ('pass', 'fail', 'manual_review'))
                , ai_confidence DECIMAL(3, 2)
                , human_override VARCHAR(50)
                , override_reason TEXT
                , evaluated_by VARCHAR(50) NOT NULL CHECK (evaluated_by IN ('ai', 'human'))
                , evaluator_id UUID REFERENCES auth.users(id)
                , evaluated_at TIMESTAMPTZ DEFAULT NOW()
                , created_at TIMESTAMPTZ DEFAULT NOW()
              );

              CREATE INDEX
              IF
                NOT EXISTS idx_evaluations_experiment
                ON experiments_v1.evaluations(experiment_id);
                CREATE INDEX
                IF
                  NOT EXISTS idx_evaluations_verdict
                  ON experiments_v1.evaluations(ai_verdict);
                  CREATE INDEX
                  IF
                    NOT EXISTS idx_evaluations_evaluated_at
                    ON experiments_v1.evaluations(evaluated_at);

                    -- Promotions table
                    CREATE TABLE
                    IF
                      NOT EXISTS experiments_v1.promotions (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                        , from_version_id UUID NOT NULL
                        , to_version_id UUID NOT NULL
                        , status VARCHAR(50) NOT NULL CHECK (
                          status IN ('pending', 'approved', 'rejected', 'completed')
                        )
                        , reason TEXT NOT NULL
                        , approver_id UUID REFERENCES auth.users(id)
                        , promoted_at TIMESTAMPTZ
                        , created_at TIMESTAMPTZ DEFAULT NOW()
                        , updated_at TIMESTAMPTZ DEFAULT NOW()
                      );

                      CREATE INDEX
                      IF
                        NOT EXISTS idx_promotions_status
                        ON experiments_v1.promotions(status);
                        CREATE INDEX
                        IF
                          NOT EXISTS idx_promotions_from_version
                          ON experiments_v1.promotions(from_version_id);
                          CREATE INDEX
                          IF
                            NOT EXISTS idx_promotions_to_version
                            ON experiments_v1.promotions(to_version_id);

                            -- Blacklist table
                            CREATE TABLE
                            IF
                              NOT EXISTS experiments_v1.blacklist (
                                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                , config_hash VARCHAR(64) UNIQUE NOT NULL
                                , reason TEXT NOT NULL
                                , evidence JSONB NOT NULL
                                , quarantined_at TIMESTAMPTZ DEFAULT NOW()
                                , quarantined_by UUID REFERENCES auth.users(id)
                                , review_status VARCHAR(50) NOT NULL CHECK (review_status IN ('pending', 'reviewed', 'appealed'))
                                , reviewed_by UUID REFERENCES auth.users(id)
                                , reviewed_at TIMESTAMPTZ
                              );

                              CREATE INDEX
                              IF
                                NOT EXISTS idx_blacklist_config_hash
                                ON experiments_v1.blacklist(config_hash);
                                CREATE INDEX
                                IF
                                  NOT EXISTS idx_blacklist_review_status
                                  ON experiments_v1.blacklist(review_status);

                                  -- Comparison reports table
                                  CREATE TABLE
                                  IF
                                    NOT EXISTS experiments_v1.comparison_reports (
                                      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                      , v1_experiment_id UUID NOT NULL REFERENCES experiments_v1.experiments(id)
                                      , v2_version_id UUID NOT NULL
                                      , dataset_id UUID NOT NULL
                                      , metrics JSONB NOT NULL
                                      , recommendation VARCHAR(255) NOT NULL
                                      , confidence DECIMAL(3, 2) NOT NULL
                                      , created_at TIMESTAMPTZ DEFAULT NOW()
                                    );

                                    CREATE INDEX
                                    IF
                                      NOT EXISTS idx_comparison_v1_exp
                                      ON experiments_v1.comparison_reports(v1_experiment_id);
                                      CREATE INDEX
                                      IF
                                        NOT EXISTS idx_comparison_v2_ver
                                        ON experiments_v1.comparison_reports(v2_version_id);

                                        -- Grant permissions
                                        GRANT
                                          USAGE
                                        ON SCHEMA experiments_v1
                                        TO
                                          PUBLIC;
                                        GRANT
                                          SELECT
                                          , INSERT
                                        , UPDATE
                                        ON ALL TABLES IN SCHEMA experiments_v1
                                        TO
                                          PUBLIC;