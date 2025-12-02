-- ============================================
-- Schema: providers
-- AI provider configuration and usage tracking
-- ============================================

CREATE SCHEMA
IF
  NOT EXISTS providers;

  -- Providers table
  CREATE TABLE
  IF
    NOT EXISTS providers.providers (
      id UUID PRIMARY KEY DEFAULT gen_random_uuid()
      , name VARCHAR(100) UNIQUE NOT NULL
      , provider_type VARCHAR(50) NOT NULL CHECK (
        provider_type IN ('openai', 'anthropic', 'huggingface', 'local')
      )
      , model_name VARCHAR(100) NOT NULL
      , api_endpoint VARCHAR(255)
      , is_active BOOLEAN DEFAULT TRUE
      , is_platform_provided BOOLEAN DEFAULT TRUE
      , cost_per_1k_tokens DECIMAL(10, 6) NOT NULL
      , max_tokens INTEGER NOT NULL
      , timeout_seconds INTEGER DEFAULT 30
      , created_at TIMESTAMPTZ DEFAULT NOW()
      , updated_at TIMESTAMPTZ DEFAULT NOW()
    );

    CREATE INDEX
    IF
      NOT EXISTS idx_providers_active
      ON providers.providers(is_active);
      CREATE INDEX
      IF
        NOT EXISTS idx_providers_type
        ON providers.providers(provider_type);

        -- User providers table (for user-provided API keys)
        CREATE TABLE
        IF
          NOT EXISTS providers.user_providers (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid()
            , user_id UUID NOT NULL REFERENCES auth.users(id)
            ON DELETE CASCADE
            , provider_name VARCHAR(100) NOT NULL
            , provider_type VARCHAR(50) NOT NULL CHECK (
              provider_type IN ('openai', 'anthropic', 'huggingface')
            )
            , model_name VARCHAR(100) NOT NULL
            , encrypted_api_key TEXT NOT NULL
            , encrypted_dek TEXT NOT NULL
            , -- Data Encryption Key (encrypted by KMS)
              key_last_4_chars VARCHAR(4) NOT NULL
            , is_active BOOLEAN DEFAULT TRUE
            , created_at TIMESTAMPTZ DEFAULT NOW()
            , updated_at TIMESTAMPTZ DEFAULT NOW()
            , UNIQUE(user_id, provider_type)
          );

          CREATE INDEX
          IF
            NOT EXISTS idx_user_providers_user
            ON providers.user_providers(user_id);
            CREATE INDEX
            IF
              NOT EXISTS idx_user_providers_active
              ON providers.user_providers(is_active);

              -- Provider health table
              CREATE TABLE
              IF
                NOT EXISTS providers.provider_health (
                  id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                  , provider_id UUID NOT NULL REFERENCES providers.providers(id)
                  ON DELETE CASCADE
                  , is_healthy BOOLEAN DEFAULT TRUE
                  , last_check_at TIMESTAMPTZ
                  , last_error TEXT
                  , consecutive_failures INTEGER DEFAULT 0
                  , response_time_ms DECIMAL(10, 2)
                  , success_rate DECIMAL(3, 2) DEFAULT 1.0
                  , updated_at TIMESTAMPTZ DEFAULT NOW()
                );

                CREATE INDEX
                IF
                  NOT EXISTS idx_provider_health_provider
                  ON providers.provider_health(provider_id);
                  CREATE INDEX
                  IF
                    NOT EXISTS idx_provider_health_healthy
                    ON providers.provider_health(is_healthy);

                    -- User quotas table
                    CREATE TABLE
                    IF
                      NOT EXISTS providers.user_quotas (
                        id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                        , user_id UUID UNIQUE NOT NULL REFERENCES auth.users(id)
                        ON DELETE CASCADE
                        , daily_limit INTEGER NOT NULL
                        , monthly_limit INTEGER NOT NULL
                        , daily_cost_limit DECIMAL(10, 2) NOT NULL
                        , monthly_cost_limit DECIMAL(10, 2) NOT NULL
                        , created_at TIMESTAMPTZ DEFAULT NOW()
                        , updated_at TIMESTAMPTZ DEFAULT NOW()
                      );

                      CREATE INDEX
                      IF
                        NOT EXISTS idx_quotas_user
                        ON providers.user_quotas(user_id);

                        -- Usage tracking table (time-series data)
                        CREATE TABLE
                        IF
                          NOT EXISTS providers.usage_tracking (
                            id BIGSERIAL PRIMARY KEY
                            , user_id UUID NOT NULL REFERENCES auth.users(id)
                            ON DELETE CASCADE
                            , provider VARCHAR(50)
                            , model VARCHAR(100)
                            , date DATE NOT NULL
                            , requests_count INTEGER DEFAULT 0
                            , tokens_input INTEGER DEFAULT 0
                            , tokens_output INTEGER DEFAULT 0
                            , cost_usd DECIMAL(10, 4) DEFAULT 0
                            , created_at TIMESTAMPTZ DEFAULT NOW()
                            , UNIQUE(user_id, provider, model, date)
                          );

                          CREATE INDEX
                          IF
                            NOT EXISTS idx_usage_user_date
                            ON providers.usage_tracking(user_id, date);
                            CREATE INDEX
                            IF
                              NOT EXISTS idx_usage_date
                              ON providers.usage_tracking(date);
                              CREATE INDEX
                              IF
                                NOT EXISTS idx_usage_provider
                                ON providers.usage_tracking(provider);

                                -- Cost alerts table
                                CREATE TABLE
                                IF
                                  NOT EXISTS providers.cost_alerts (
                                    id UUID PRIMARY KEY DEFAULT gen_random_uuid()
                                    , user_id UUID NOT NULL REFERENCES auth.users(id)
                                    ON DELETE CASCADE
                                    , alert_type VARCHAR(50) NOT NULL CHECK (
                                      alert_type IN (
                                        'daily_80'
                                        , 'daily_90'
                                        , 'daily_100'
                                        , 'monthly_80'
                                        , 'monthly_90'
                                        , 'monthly_100'
                                      )
                                    )
                                    , threshold_percentage INTEGER NOT NULL
                                    , triggered_at TIMESTAMPTZ DEFAULT NOW()
                                    , acknowledged BOOLEAN DEFAULT FALSE
                                    , acknowledged_at TIMESTAMPTZ
                                  );

                                  CREATE INDEX
                                  IF
                                    NOT EXISTS idx_alerts_user
                                    ON providers.cost_alerts(user_id);
                                    CREATE INDEX
                                    IF
                                      NOT EXISTS idx_alerts_triggered
                                      ON providers.cost_alerts(triggered_at);
                                      CREATE INDEX
                                      IF
                                        NOT EXISTS idx_alerts_acknowledged
                                        ON providers.cost_alerts(acknowledged);

                                        -- Grant permissions
                                        GRANT
                                          USAGE
                                        ON SCHEMA providers
                                        TO
                                          PUBLIC;
                                        GRANT
                                          SELECT
                                          , INSERT
                                        , UPDATE
                                        ON ALL TABLES IN SCHEMA providers
                                        TO
                                          PUBLIC;