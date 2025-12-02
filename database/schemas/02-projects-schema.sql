-- ============================================
-- Schema: projects
-- Project and version management
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS projects;

  -- Projects table
  CREATE TABLE
  IF
    NOT EXISTS projects.projects (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , name VARCHAR(255) NOT NULL
      , description TEXT
      , owner_id UUID NOT NULL REFERENCES auth.users(id)
      ON DELETE CASCADE
      , settings JSONB DEFAULT '{}': :jsonb
      , archived BOOLEAN DEFAULT FALSE
      , created_at TIMESTAMPTZ DEFAULT NOW()
      , updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_projects_owner
      ON projects.projects(owner_id);
      CREATE INDEX
      IF
        NOT EXISTS idx_projects_archived
        ON projects.projects(archived);

        -- Versions table
        CREATE TABLE
        IF
          NOT EXISTS projects.versions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
            , project_id UUID NOT NULL REFERENCES projects.projects(id)
            ON DELETE CASCADE
            , tag VARCHAR(10) NOT NULL CHECK (tag IN ('v1', 'v2', 'v3'))
            , model_config JSONB NOT NULL
            , changelog TEXT
            , promoted_at TIMESTAMPTZ
            , promoted_by UUID REFERENCES auth.users(id)
            , created_at TIMESTAMPTZ DEFAULT NOW()
            , updated_at TIMESTAMPTZ DEFAULT NOW()
            , UNIQUE(project_id, tag)
          );

          CREATE INDEX
          IF
            NOT EXISTS idx_versions_project
            ON projects.versions(project_id);
            CREATE INDEX
            IF
              NOT EXISTS idx_versions_tag
              ON projects.versions(tag);
              CREATE INDEX
              IF
                NOT EXISTS idx_versions_promoted
                ON projects.versions(promoted_at);

                -- Baselines table
                CREATE TABLE
                IF
                  NOT EXISTS projects.baselines (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                    , project_id UUID NOT NULL REFERENCES projects.projects(id)
                    ON DELETE CASCADE
                    , metric_key VARCHAR(100) NOT NULL
                    , threshold DECIMAL(10, 4) NOT NULL
                    , operator VARCHAR(10) NOT NULL CHECK (operator IN ('>', '<', '>=', '<=', '='))
                    , snapshot_id UUID
                    , active BOOLEAN DEFAULT TRUE
                    , created_at TIMESTAMPTZ DEFAULT NOW()
                    , updated_at TIMESTAMPTZ DEFAULT NOW()
                  );

                  CREATE INDEX
                  IF
                    NOT EXISTS idx_baselines_project
                    ON projects.baselines(project_id);
                    CREATE INDEX
                    IF
                      NOT EXISTS idx_baselines_active
                      ON projects.baselines(active);

                      -- Policies table
                      CREATE TABLE
                      IF
                        NOT EXISTS projects.policies (
                          id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                          , name VARCHAR(255) UNIQUE NOT NULL
                          , description TEXT
                          , rego_code TEXT NOT NULL
                          , active BOOLEAN DEFAULT TRUE
                          , created_at TIMESTAMPTZ DEFAULT NOW()
                          , updated_at TIMESTAMPTZ DEFAULT NOW()
                        );

                        CREATE INDEX
                        IF
                          NOT EXISTS idx_policies_active
                          ON projects.policies(active);

                          -- Project-Policy associations
                          CREATE TABLE
                          IF
                            NOT EXISTS projects.project_policies (
                              id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                              , project_id UUID NOT NULL REFERENCES projects.projects(id)
                              ON DELETE CASCADE
                              , policy_id UUID NOT NULL REFERENCES projects.policies(id)
                              ON DELETE CASCADE
                              , created_at TIMESTAMPTZ DEFAULT NOW()
                              , UNIQUE(project_id, policy_id)
                            );

                            CREATE INDEX
                            IF
                              NOT EXISTS idx_project_policies_project
                              ON projects.project_policies(project_id);
                              CREATE INDEX
                              IF
                                NOT EXISTS idx_project_policies_policy
                                ON projects.project_policies(policy_id);

                                -- Version history table
                                CREATE TABLE
                                IF
                                  NOT EXISTS projects.version_history (
                                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                    , version_id UUID NOT NULL REFERENCES projects.versions(id)
                                    ON DELETE CASCADE
                                    , changed_by UUID NOT NULL REFERENCES auth.users(id)
                                    , action VARCHAR(50) NOT NULL CHECK (action IN ('promoted', 'degraded', 'updated'))
                                    , from_tag VARCHAR(10)
                                    , to_tag VARCHAR(10)
                                    , reason TEXT
                                    , created_at TIMESTAMPTZ DEFAULT NOW()
                                  );

                                  CREATE INDEX
                                  IF
                                    NOT EXISTS idx_version_history_version
                                    ON projects.version_history(version_id);
                                    CREATE INDEX
                                    IF
                                      NOT EXISTS idx_version_history_changed_by
                                      ON projects.version_history(changed_by);
                                      CREATE INDEX
                                      IF
                                        NOT EXISTS idx_version_history_action
                                        ON projects.version_history(action);

                                        -- Grant permissions
                                        GRANT
                                          USAGE
                                        ON SCHEMA projects
                                        TO
                                          PUBLIC;
                                        GRANT
                                          SELECT
                                          , INSERT
                                        , UPDATE
                                        ON ALL TABLES IN SCHEMA projects
                                        TO
                                          PUBLIC;