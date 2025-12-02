#!/bin/bash

# Database initialization script
# Initializes PostgreSQL database with all schemas

set -e

# Configuration
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${DB_NAME:-ai_code_review}
DB_USER=${DB_USER:-postgres}
DB_PASSWORD=${DB_PASSWORD:-postgres}
SCHEMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/schemas" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Starting database initialization...${NC}"

# Export password for psql
export PGPASSWORD=$DB_PASSWORD

# Function to execute SQL file
execute_sql() {
    local file=$1
    local description=$2
    
    echo -e "${YELLOW}Executing: $description${NC}"
    
    if psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -f "$file" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ $description completed${NC}"
    else
        echo -e "${RED}✗ $description failed${NC}"
        exit 1
    fi
}

# Create database if it doesn't exist
echo -e "${YELLOW}Creating database if not exists...${NC}"
psql -h $DB_HOST -p $DB_PORT -U $DB_USER -tc "SELECT 1 FROM pg_database WHERE datname = '$DB_NAME'" | grep -q 1 || \
    psql -h $DB_HOST -p $DB_PORT -U $DB_USER -c "CREATE DATABASE $DB_NAME"
echo -e "${GREEN}✓ Database ready${NC}"

# Execute schema files in order
execute_sql "$SCHEMA_DIR/01-auth-schema.sql" "Auth schema"
execute_sql "$SCHEMA_DIR/02-projects-schema.sql" "Projects schema"
execute_sql "$SCHEMA_DIR/03-experiments-schema.sql" "Experiments schema"
execute_sql "$SCHEMA_DIR/04-production-schema.sql" "Production schema"
execute_sql "$SCHEMA_DIR/05-quarantine-schema.sql" "Quarantine schema"
execute_sql "$SCHEMA_DIR/06-providers-schema.sql" "Providers schema"
execute_sql "$SCHEMA_DIR/07-audit-schema.sql" "Audit schema"

echo -e "${GREEN}✓ Database initialization completed successfully${NC}"
