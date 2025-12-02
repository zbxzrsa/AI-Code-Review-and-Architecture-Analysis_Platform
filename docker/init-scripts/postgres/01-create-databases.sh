#!/bin/bash
# =============================================================================
# PostgreSQL Initialization Script
# Creates multiple databases for microservices
# =============================================================================

set -e
set -u

# Function to create database and user
create_database() {
    local database=$1
    local user="${database}_user"
    local password="${database}_password"
    
    echo "Creating database: $database"
    
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
        -- Create user if not exists
        DO \$\$
        BEGIN
            IF NOT EXISTS (SELECT FROM pg_catalog.pg_user WHERE usename = '$user') THEN
                CREATE USER $user WITH PASSWORD '$password';
            END IF;
        END
        \$\$;
        
        -- Create database if not exists
        SELECT 'CREATE DATABASE $database OWNER $user'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$database')\gexec
        
        -- Grant privileges
        GRANT ALL PRIVILEGES ON DATABASE $database TO $user;
EOSQL
    
    echo "Database $database created successfully"
}

# Parse POSTGRES_MULTIPLE_DATABASES environment variable
if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
    echo "Creating multiple databases: $POSTGRES_MULTIPLE_DATABASES"
    
    for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
        create_database "$db"
    done
    
    echo "All databases created successfully"
else
    echo "No POSTGRES_MULTIPLE_DATABASES specified, skipping"
fi

# Create extensions in each database
for db in auth_db project_db analysis_db experiment_db; do
    echo "Creating extensions in $db"
    
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
        -- UUID extension
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- Crypto extension (for pgcrypto)
        CREATE EXTENSION IF NOT EXISTS "pgcrypto";
        
        -- Full text search
        CREATE EXTENSION IF NOT EXISTS "pg_trgm";
EOSQL
done

echo "PostgreSQL initialization complete"
