-- ============================================
-- AI Code Review Platform - Database Schema
-- ============================================
-- This script creates all tables for the platform
-- Run with: psql -U postgres -d codereview -f 01-schema.sql

-- Enable required extensions
CREATE EXTENSION
IF
  NOT EXISTS "uuid-ossp";
  CREATE EXTENSION
  IF
    NOT EXISTS "pgcrypto";

    -- ============================================
    -- USERS & AUTHENTICATION
    -- ============================================

    -- Users table
    CREATE TABLE
    IF
      NOT EXISTS users (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
        , email VARCHAR(255) UNIQUE NOT NULL
        , password_hash VARCHAR(255)
        , name VARCHAR(255) NOT NULL
        , avatar_url TEXT
        , role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer', 'guest'))
        , status VARCHAR(50) DEFAULT 'active' CHECK (
          status IN ('active', 'inactive', 'suspended', 'pending')
        )
        , email_verified BOOLEAN DEFAULT FALSE
        , two_factor_enabled BOOLEAN DEFAULT FALSE
        , two_factor_secret VARCHAR(255)
        , last_login_at TIMESTAMP WITH TIME ZONE
        , login_count INTEGER DEFAULT 0
        , failed_login_attempts INTEGER DEFAULT 0
        , locked_until TIMESTAMP WITH TIME ZONE
        , settings JSONB DEFAULT '{}'
        , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
      );

      CREATE INDEX idx_users_email
      ON users(email);
      CREATE INDEX idx_users_status
      ON users(status);

      -- User sessions
      CREATE TABLE
      IF
        NOT EXISTS user_sessions (
          id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
          , user_id UUID NOT NULL REFERENCES users(id)
          ON DELETE CASCADE
          , token_hash VARCHAR(255) NOT NULL
          , refresh_token_hash VARCHAR(255)
          , device_info JSONB DEFAULT '{}'
          , ip_address INET
          , user_agent TEXT
          , is_active BOOLEAN DEFAULT TRUE
          , expires_at TIMESTAMP WITH TIME ZONE NOT NULL
          , last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
          , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        CREATE INDEX idx_sessions_user
        ON user_sessions(user_id);
        CREATE INDEX idx_sessions_token
        ON user_sessions(token_hash);
        CREATE INDEX idx_sessions_active
        ON user_sessions(is_active, expires_at);

        -- OAuth connections
        CREATE TABLE
        IF
          NOT EXISTS oauth_connections (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
            , user_id UUID NOT NULL REFERENCES users(id)
            ON DELETE CASCADE
            , provider VARCHAR(50) NOT NULL CHECK (
              provider IN ('github', 'gitlab', 'bitbucket', 'google')
            )
            , provider_user_id VARCHAR(255) NOT NULL
            , provider_username VARCHAR(255)
            , provider_email VARCHAR(255)
            , access_token_encrypted TEXT
            , refresh_token_encrypted TEXT
            , token_expires_at TIMESTAMP WITH TIME ZONE
            , scopes TEXT []
            , raw_data JSONB
            , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            , UNIQUE(provider, provider_user_id)
          );

          CREATE INDEX idx_oauth_user
          ON oauth_connections(user_id);
          CREATE INDEX idx_oauth_provider
          ON oauth_connections(provider, provider_user_id);

          -- API Keys
          CREATE TABLE
          IF
            NOT EXISTS api_keys (
              id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
              , user_id UUID NOT NULL REFERENCES users(id)
              ON DELETE CASCADE
              , name VARCHAR(255) NOT NULL
              , key_hash VARCHAR(255) NOT NULL UNIQUE
              , key_prefix VARCHAR(10) NOT NULL
              , scopes TEXT [] DEFAULT ARRAY ['read']
              , rate_limit INTEGER DEFAULT 1000
              , expires_at TIMESTAMP WITH TIME ZONE
              , last_used_at TIMESTAMP WITH TIME ZONE
              , usage_count INTEGER DEFAULT 0
              , is_active BOOLEAN DEFAULT TRUE
              , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
              , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );

            CREATE INDEX idx_api_keys_user
            ON api_keys(user_id);
            CREATE INDEX idx_api_keys_hash
            ON api_keys(key_hash);

            -- Invitation codes
            CREATE TABLE
            IF
              NOT EXISTS invitation_codes (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                , code VARCHAR(50) UNIQUE NOT NULL
                , created_by UUID REFERENCES users(id)
                , used_by UUID REFERENCES users(id)
                , max_uses INTEGER DEFAULT 1
                , use_count INTEGER DEFAULT 0
                , expires_at TIMESTAMP WITH TIME ZONE
                , is_active BOOLEAN DEFAULT TRUE
                , metadata JSONB DEFAULT '{}'
                , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
              );

              CREATE INDEX idx_invitation_code
              ON invitation_codes(code);

              -- ============================================
              -- PROJECTS
              -- ============================================

              CREATE TABLE
              IF
                NOT EXISTS projects (
                  id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                  , name VARCHAR(255) NOT NULL
                  , description TEXT
                  , language VARCHAR(50)
                  , framework VARCHAR(100)
                  , repository_url TEXT
                  , owner_id UUID NOT NULL REFERENCES users(id)
                  ON DELETE CASCADE
                  , status VARCHAR(50) DEFAULT 'active' CHECK (
                    status IN ('active', 'archived', 'deleted', 'pending')
                  )
                  , is_public BOOLEAN DEFAULT FALSE
                  , settings JSONB DEFAULT '{}'
                  , total_issues INTEGER DEFAULT 0
                  , open_issues INTEGER DEFAULT 0
                  , last_analysis_at TIMESTAMP WITH TIME ZONE
                  , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                  , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                );

                CREATE INDEX idx_projects_owner
                ON projects(owner_id);
                CREATE INDEX idx_projects_status
                ON projects(status);
                CREATE INDEX idx_projects_public
                ON projects(is_public)
                WHERE
                  is_public = TRUE;

                -- Project members
                CREATE TABLE
                IF
                  NOT EXISTS project_members (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                    , project_id UUID NOT NULL REFERENCES projects(id)
                    ON DELETE CASCADE
                    , user_id UUID NOT NULL REFERENCES users(id)
                    ON DELETE CASCADE
                    , role VARCHAR(50) DEFAULT 'member' CHECK (role IN ('owner', 'admin', 'member', 'viewer'))
                    , permissions JSONB DEFAULT '{}'
                    , invited_by UUID REFERENCES users(id)
                    , joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    , UNIQUE(project_id, user_id)
                  );

                  CREATE INDEX idx_project_members_project
                  ON project_members(project_id);
                  CREATE INDEX idx_project_members_user
                  ON project_members(user_id);

                  -- ============================================
                  -- REPOSITORIES
                  -- ============================================

                  CREATE TABLE
                  IF
                    NOT EXISTS repositories (
                      id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                      , name VARCHAR(255) NOT NULL
                      , full_name VARCHAR(500)
                      , description TEXT
                      , provider VARCHAR(50) DEFAULT 'github' CHECK (
                        provider IN ('github', 'gitlab', 'bitbucket', 'local')
                      )
                      , provider_repo_id VARCHAR(255)
                      , clone_url TEXT NOT NULL
                      , ssh_url TEXT
                      , default_branch VARCHAR(100) DEFAULT 'main'
                      , is_private BOOLEAN DEFAULT FALSE
                      , owner_id UUID NOT NULL REFERENCES users(id)
                      ON DELETE CASCADE
                      , project_id UUID REFERENCES projects(id)
                      ON DELETE SET NULL
                      , status VARCHAR(50) DEFAULT 'pending' CHECK (
                        status IN (
                          'pending'
                          , 'cloning'
                          , 'syncing'
                          , 'ready'
                          , 'error'
                          , 'deleted'
                        )
                      )
                      , local_path TEXT
                      , webhook_id VARCHAR(255)
                      , webhook_secret VARCHAR(255)
                      , last_synced_at TIMESTAMP WITH TIME ZONE
                      , last_commit_sha VARCHAR(40)
                      , stats JSONB DEFAULT '{}'
                      , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                      , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                    );

                    CREATE INDEX idx_repos_owner
                    ON repositories(owner_id);
                    CREATE INDEX idx_repos_project
                    ON repositories(project_id);
                    CREATE INDEX idx_repos_provider
                    ON repositories(provider, provider_repo_id);
                    CREATE INDEX idx_repos_status
                    ON repositories(status);

                    -- ============================================
                    -- ANALYSES & ISSUES
                    -- ============================================

                    CREATE TABLE
                    IF
                      NOT EXISTS analyses (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                        , project_id UUID NOT NULL REFERENCES projects(id)
                        ON DELETE CASCADE
                        , repository_id UUID REFERENCES repositories(id)
                        ON DELETE SET NULL
                        , triggered_by UUID REFERENCES users(id)
                        , type VARCHAR(50) DEFAULT 'full' CHECK (type IN ('full', 'incremental', 'diff', 'pr'))
                        , status VARCHAR(50) DEFAULT 'pending' CHECK (
                          status IN (
                            'pending'
                            , 'running'
                            , 'completed'
                            , 'failed'
                            , 'cancelled'
                          )
                        )
                        , commit_sha VARCHAR(40)
                        , branch VARCHAR(100)
                        , pr_number INTEGER
                        , config JSONB DEFAULT '{}'
                        , results JSONB DEFAULT '{}'
                        , metrics JSONB DEFAULT '{}'
                        , total_files INTEGER DEFAULT 0
                        , analyzed_files INTEGER DEFAULT 0
                        , total_issues INTEGER DEFAULT 0
                        , critical_issues INTEGER DEFAULT 0
                        , high_issues INTEGER DEFAULT 0
                        , medium_issues INTEGER DEFAULT 0
                        , low_issues INTEGER DEFAULT 0
                        , duration_ms INTEGER
                        , error_message TEXT
                        , started_at TIMESTAMP WITH TIME ZONE
                        , completed_at TIMESTAMP WITH TIME ZONE
                        , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                      );

                      CREATE INDEX idx_analyses_project
                      ON analyses(project_id);
                      CREATE INDEX idx_analyses_status
                      ON analyses(status);
                      CREATE INDEX idx_analyses_created
                      ON analyses(created_at DESC);

                      -- Issues found during analysis
                      CREATE TABLE
                      IF
                        NOT EXISTS issues (
                          id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                          , analysis_id UUID NOT NULL REFERENCES analyses(id)
                          ON DELETE CASCADE
                          , project_id UUID NOT NULL REFERENCES projects(id)
                          ON DELETE CASCADE
                          , file_path TEXT NOT NULL
                          , line_start INTEGER
                          , line_end INTEGER
                          , column_start INTEGER
                          , column_end INTEGER
                          , severity VARCHAR(20) DEFAULT 'medium' CHECK (
                            severity IN ('critical', 'high', 'medium', 'low', 'info')
                          )
                          , category VARCHAR(100) NOT NULL
                          , rule_id VARCHAR(100)
                          , title VARCHAR(500) NOT NULL
                          , description TEXT
                          , suggestion TEXT
                          , code_snippet TEXT
                          , fix_available BOOLEAN DEFAULT FALSE
                          , fix_code TEXT
                          , confidence DECIMAL(3, 2) DEFAULT 1.0
                          , status VARCHAR(50) DEFAULT 'open' CHECK (
                            status IN (
                              'open'
                              , 'confirmed'
                              , 'fixed'
                              , 'ignored'
                              , 'false_positive'
                            )
                          )
                          , resolved_by UUID REFERENCES users(id)
                          , resolved_at TIMESTAMP WITH TIME ZONE
                          , metadata JSONB DEFAULT '{}'
                          , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                          , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                        );

                        CREATE INDEX idx_issues_analysis
                        ON issues(analysis_id);
                        CREATE INDEX idx_issues_project
                        ON issues(project_id);
                        CREATE INDEX idx_issues_status
                        ON issues(status);
                        CREATE INDEX idx_issues_severity
                        ON issues(severity);
                        CREATE INDEX idx_issues_file
                        ON issues(file_path);

                        -- ============================================
                        -- AI CHAT & INTERACTIONS
                        -- ============================================

                        CREATE TABLE
                        IF
                          NOT EXISTS chat_sessions (
                            id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                            , user_id UUID NOT NULL REFERENCES users(id)
                            ON DELETE CASCADE
                            , project_id UUID REFERENCES projects(id)
                            ON DELETE SET NULL
                            , title VARCHAR(255)
                            , type VARCHAR(50) DEFAULT 'general' CHECK (
                              type IN (
                                'general'
                                , 'code_review'
                                , 'architecture'
                                , 'debug'
                                , 'refactor'
                              )
                            )
                            , status VARCHAR(50) DEFAULT 'active' CHECK (status IN ('active', 'archived', 'deleted'))
                            , context JSONB DEFAULT '{}'
                            , model_config JSONB DEFAULT '{}'
                            , message_count INTEGER DEFAULT 0
                            , token_count INTEGER DEFAULT 0
                            , last_message_at TIMESTAMP WITH TIME ZONE
                            , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                            , updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                          );

                          CREATE INDEX idx_chat_sessions_user
                          ON chat_sessions(user_id);
                          CREATE INDEX idx_chat_sessions_project
                          ON chat_sessions(project_id);

                          CREATE TABLE
                          IF
                            NOT EXISTS chat_messages (
                              id UUID PRIMARY KEY DEFAULT uuid_generate_v4()
                              , session_id UUID NOT NULL REFERENCES chat_sessions(id)
                              ON DELETE CASCADE
                              , role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system'))
                              , content TEXT NOT NULL
                              , attachments JSONB DEFAULT '[]'
                              , metadata JSONB DEFAULT '{}'
                              , token_count INTEGER DEFAULT 0
                              , model VARCHAR(100)
                              , latency_ms INTEGER
                              , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                            );

                            CREATE INDEX idx_chat_messages_session
                            ON chat_messages(session_id);
                            CREATE INDEX idx_chat_messages_created
                            ON chat_messages(created_at);

                            -- ============================================
                            -- AUDIT LOG
                            -- ============================================

                            CREATE TABLE
                            IF
                              NOT EXISTS audit_logs (
                                id BIGSERIAL PRIMARY KEY
                                , user_id UUID REFERENCES users(id)
                                ON DELETE SET NULL
                                , action VARCHAR(100) NOT NULL
                                , entity_type VARCHAR(100) NOT NULL
                                , entity_id VARCHAR(255)
                                , old_values JSONB
                                , new_values JSONB
                                , ip_address INET
                                , user_agent TEXT
                                , status VARCHAR(20) DEFAULT 'success' CHECK (status IN ('success', 'failure', 'denied'))
                                , error_message TEXT
                                , created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
                              );

                              CREATE INDEX idx_audit_user
                              ON audit_logs(user_id);
                              CREATE INDEX idx_audit_action
                              ON audit_logs(action);
                              CREATE INDEX idx_audit_entity
                              ON audit_logs(entity_type, entity_id);
                              CREATE INDEX idx_audit_created
                              ON audit_logs(created_at DESC);

                              -- ============================================
                              -- FUNCTIONS & TRIGGERS
                              -- ============================================

                              -- Update timestamp trigger function
                              CREATE OR REPLACE FUNCTION update_updated_at()
                              RETURNS TRIGGER AS $ $
                              BEGIN
                                NEW.updated_at = NOW();
                                RETURN
                                NEW;
                              END;
                              $ $
                              LANGUAGE plpgsql;

                              -- Apply update trigger to all tables with updated_at
                            DO
                              $ $
                              DECLARE t text;
                              BEGIN
                                FOR t IN
                                SELECT
                                  table_name
                                FROM
                                  information_schema.columns
                                WHERE
                                  column_name = 'updated_at'
                                  AND table_schema = 'public'
                                LOOP
                                  EXECUTE format(
                                    '
            DROP TRIGGER IF EXISTS update_%I_updated_at ON %I;
            CREATE TRIGGER update_%I_updated_at
            BEFORE UPDATE ON %I
            FOR EACH ROW
            EXECUTE FUNCTION update_updated_at();
        '
                                    , t
                                    , t
                                    , t
                                    , t
                                  );
                                END LOOP;
                              END;
                              $ $;

                              -- ============================================
                              -- INITIAL DATA
                              -- ============================================

                              -- Create default admin user (password: admin123 - change in production!)
                              INSERT INTO
                                users (
                                  id
                                  , email
                                  , password_hash
                                  , name
                                  , role
                                  , status
                                  , email_verified
                                )
                              VALUES
                                (
                                  '00000000-0000-0000-0000-000000000001'
                                  , 'admin@example.com'
                                  , crypt('admin123', gen_salt('bf', 12))
                                  , 'Admin User'
                                  , 'admin'
                                  , 'active'
                                  , TRUE
                                )
                              ON CONFLICT (email)
                            DO
                              NOTHING;

                              -- Create default invitation code
                              INSERT INTO
                                invitation_codes (code, max_uses, is_active)
                              VALUES
                                ('WELCOME2024', 100, TRUE)
                              ON CONFLICT (code)
                            DO
                              NOTHING;

                              -- ============================================
                              -- GRANTS (adjust as needed)
                              -- ============================================

                              -- Grant permissions to application user (create user first if needed)
                              -- CREATE USER app_user WITH PASSWORD 'secure_password';
                              -- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
                              -- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;

                              COMMENT ON DATABASE codereview IS 'AI Code Review Platform Database';