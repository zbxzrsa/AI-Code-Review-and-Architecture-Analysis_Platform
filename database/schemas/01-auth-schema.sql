-- ============================================
-- Schema: auth
-- User authentication and session management
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS auth;

  -- Users table
  CREATE TABLE
  IF
    NOT EXISTS auth.users (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , email VARCHAR(255) UNIQUE NOT NULL
      , password_hash VARCHAR(255) NOT NULL
      , name VARCHAR(255)
      , role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer', 'guest'))
      , verified BOOLEAN DEFAULT FALSE
      , totp_secret VARCHAR(32)
      , created_at TIMESTAMPTZ DEFAULT NOW()
      , updated_at TIMESTAMPTZ DEFAULT NOW()
      , last_login TIMESTAMPTZ
      , failed_login_attempts INTEGER DEFAULT 0
      , locked_until TIMESTAMPTZ
      , avatar VARCHAR(512)
      , settings JSONB DEFAULT '{}': :jsonb
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_users_email
      ON auth.users(email);
      CREATE INDEX
      IF
        NOT EXISTS idx_users_role
        ON auth.users(role);
        CREATE INDEX
        IF
          NOT EXISTS idx_users_verified
          ON auth.users(verified);

          -- Sessions table
          CREATE TABLE
          IF
            NOT EXISTS auth.sessions (
              id UUID PRIMARY KEY DEFAULT gen_random_uuid()
              , user_id UUID NOT NULL REFERENCES auth.users(id)
              ON DELETE CASCADE
              , refresh_token_hash VARCHAR(255) NOT NULL UNIQUE
              , expires_at TIMESTAMPTZ NOT NULL
              , device_info JSONB
              , ip_address VARCHAR(45)
              , created_at TIMESTAMPTZ DEFAULT NOW()
              , last_used TIMESTAMPTZ DEFAULT NOW()
            );

            CREATE INDEX
            IF
              NOT EXISTS idx_sessions_user
              ON auth.sessions(user_id);
              CREATE INDEX
              IF
                NOT EXISTS idx_sessions_expires
                ON auth.sessions(expires_at);
                CREATE INDEX
                IF
                  NOT EXISTS idx_sessions_token
                  ON auth.sessions(refresh_token_hash);

                  -- Invitations table
                  CREATE TABLE
                  IF
                    NOT EXISTS auth.invitations (
                      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                      , code_hash VARCHAR(255) UNIQUE NOT NULL
                      , created_by UUID REFERENCES auth.users(id)
                      ON DELETE SET NULL
                      , role VARCHAR(20) NOT NULL DEFAULT 'user' CHECK (role IN ('admin', 'user', 'viewer'))
                      , max_uses INTEGER DEFAULT 1
                      , uses INTEGER DEFAULT 0
                      , expires_at TIMESTAMPTZ NOT NULL
                      , created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX
                    IF
                      NOT EXISTS idx_invitations_code
                      ON auth.invitations(code_hash);
                      CREATE INDEX
                      IF
                        NOT EXISTS idx_invitations_expires
                        ON auth.invitations(expires_at);

                        -- Audit logs table
                        CREATE TABLE
                        IF
                          NOT EXISTS auth.audit_logs (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                            , user_id UUID REFERENCES auth.users(id)
                            ON DELETE SET NULL
                            , action VARCHAR(255) NOT NULL
                            , resource VARCHAR(255) NOT NULL
                            , status VARCHAR(50) NOT NULL CHECK (status IN ('success', 'failure', 'denied'))
                            , details JSONB
                            , ip_address VARCHAR(45)
                            , user_agent VARCHAR(512)
                            , created_at TIMESTAMPTZ DEFAULT NOW()
                          );

                          CREATE INDEX
                          IF
                            NOT EXISTS idx_audit_logs_user
                            ON auth.audit_logs(user_id);
                            CREATE INDEX
                            IF
                              NOT EXISTS idx_audit_logs_action
                              ON auth.audit_logs(action);
                              CREATE INDEX
                              IF
                                NOT EXISTS idx_audit_logs_created
                                ON auth.audit_logs(created_at);

                                -- Password resets table
                                CREATE TABLE
                                IF
                                  NOT EXISTS auth.password_resets (
                                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                    , user_id UUID NOT NULL REFERENCES auth.users(id)
                                    ON DELETE CASCADE
                                    , token_hash VARCHAR(255) UNIQUE NOT NULL
                                    , expires_at TIMESTAMPTZ NOT NULL
                                    , used BOOLEAN DEFAULT FALSE
                                    , created_at TIMESTAMPTZ DEFAULT NOW()
                                  );

                                  CREATE INDEX
                                  IF
                                    NOT EXISTS idx_password_resets_user
                                    ON auth.password_resets(user_id);
                                    CREATE INDEX
                                    IF
                                      NOT EXISTS idx_password_resets_token
                                      ON auth.password_resets(token_hash);
                                      CREATE INDEX
                                      IF
                                        NOT EXISTS idx_password_resets_expires
                                        ON auth.password_resets(expires_at);

                                        -- API Keys table
                                        CREATE TABLE
                                        IF
                                          NOT EXISTS auth.api_keys (
                                            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                            , user_id UUID NOT NULL REFERENCES auth.users(id)
                                            ON DELETE CASCADE
                                            , name VARCHAR(255) NOT NULL
                                            , key_prefix VARCHAR(12) NOT NULL
                                            , key_hash VARCHAR(255) NOT NULL UNIQUE
                                            , scopes JSONB DEFAULT '["read"]': :jsonb
                                            , expires_at TIMESTAMPTZ
                                            , last_used TIMESTAMPTZ
                                            , created_at TIMESTAMPTZ DEFAULT NOW()
                                          );

                                          CREATE INDEX
                                          IF
                                            NOT EXISTS idx_api_keys_user
                                            ON auth.api_keys(user_id);
                                            CREATE INDEX
                                            IF
                                              NOT EXISTS idx_api_keys_hash
                                              ON auth.api_keys(key_hash);