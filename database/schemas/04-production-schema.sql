-- ============================================
-- Schema: production
-- V2 Production zone data
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS production;

  -- Analysis sessions table
  CREATE TABLE
  IF
    NOT EXISTS production.analysis_sessions (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , user_id UUID NOT NULL REFERENCES auth.users(id)
      , project_id UUID NOT NULL REFERENCES projects.projects(id)
      , version VARCHAR(10) NOT NULL CHECK (
        version IN ('v1', 'v2', 'v3')
      )
      , status VARCHAR(20) NOT NULL CHECK (
        status IN (
          'created'
          , 'queued'
          , 'running'
          , 'completed'
          , 'failed'
          , 'cancelled'
        )
      )
      , started_at TIMESTAMPTZ DEFAULT NOW()
      , finished_at TIMESTAMPTZ
      , error_message TEXT
      , metadata JSONB DEFAULT '{}': :jsonb
      , created_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_sessions_user
      ON production.analysis_sessions(user_id);
      CREATE INDEX
      IF
        NOT EXISTS idx_sessions_project
        ON production.analysis_sessions(project_id);
        CREATE INDEX
        IF
          NOT EXISTS idx_sessions_status
          ON production.analysis_sessions(status);
          CREATE INDEX
          IF
            NOT EXISTS idx_sessions_started
            ON production.analysis_sessions(started_at);

            -- Analysis tasks table
            CREATE TABLE
            IF
              NOT EXISTS production.analysis_tasks (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                , session_id UUID NOT NULL REFERENCES production.analysis_sessions(id)
                ON DELETE CASCADE
                , type VARCHAR(50) NOT NULL CHECK (type IN ('static', 'dynamic', 'graph', 'ai'))
                , status VARCHAR(20) NOT NULL CHECK (
                  status IN (
                    'pending'
                    , 'running'
                    , 'completed'
                    , 'failed'
                    , 'skipped'
                  )
                )
                , result JSONB
                , error_message TEXT
                , started_at TIMESTAMPTZ
                , finished_at TIMESTAMPTZ
                , duration_ms INTEGER
                , created_at TIMESTAMPTZ DEFAULT NOW()
              );

              CREATE INDEX
              IF
                NOT EXISTS idx_tasks_session
                ON production.analysis_tasks(session_id);
                CREATE INDEX
                IF
                  NOT EXISTS idx_tasks_type
                  ON production.analysis_tasks(type);
                  CREATE INDEX
                  IF
                    NOT EXISTS idx_tasks_status
                    ON production.analysis_tasks(status);

                    -- Artifacts table
                    CREATE TABLE
                    IF
                      NOT EXISTS production.artifacts (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                        , session_id UUID NOT NULL REFERENCES production.analysis_sessions(id)
                        ON DELETE CASCADE
                        , type VARCHAR(50) NOT NULL CHECK (type IN ('report', 'patch', 'log', 'metrics'))
                        , s3_uri TEXT NOT NULL
                        , sha256 VARCHAR(64) NOT NULL
                        , size_bytes BIGINT
                        , metadata JSONB DEFAULT '{}': :jsonb
                        , created_at TIMESTAMPTZ DEFAULT NOW()
                        , expires_at TIMESTAMPTZ
                      );

                      CREATE INDEX
                      IF
                        NOT EXISTS idx_artifacts_session
                        ON production.artifacts(session_id);
                        CREATE INDEX
                        IF
                          NOT EXISTS idx_artifacts_type
                          ON production.artifacts(type);
                          CREATE INDEX
                          IF
                            NOT EXISTS idx_artifacts_expires
                            ON production.artifacts(expires_at);

                            -- Code review results table
                            CREATE TABLE
                            IF
                              NOT EXISTS production.code_review_results (
                                id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                , session_id UUID NOT NULL REFERENCES production.analysis_sessions(id)
                                ON DELETE CASCADE
                                , code_language VARCHAR(50) NOT NULL
                                , code_length INTEGER NOT NULL
                                , overall_score DECIMAL(3, 2)
                                , vulnerabilities_count INTEGER DEFAULT 0
                                , issues_count INTEGER DEFAULT 0
                                , model_used VARCHAR(100)
                                , analysis_time_ms INTEGER
                                , created_at TIMESTAMPTZ DEFAULT NOW()
                              );

                              CREATE INDEX
                              IF
                                NOT EXISTS idx_code_review_session
                                ON production.code_review_results(session_id);
                                CREATE INDEX
                                IF
                                  NOT EXISTS idx_code_review_language
                                  ON production.code_review_results(code_language);

                                  -- Grant permissions
                                  GRANT
                                    USAGE
                                  ON SCHEMA production
                                  TO
                                    PUBLIC;
                                  GRANT
                                    SELECT
                                    , INSERT
                                  , UPDATE
                                  ON ALL TABLES IN SCHEMA production
                                  TO
                                    PUBLIC;