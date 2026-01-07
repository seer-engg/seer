#!/usr/bin/env bash
#
# Database Migration Helper Script
#
# Usage:
#   ./scripts/migrate.sh              # Run migrations (auto-detects environment)
#   ./scripts/migrate.sh create       # Create a new migration
#   ./scripts/migrate.sh rollback     # Rollback one migration
#   ./scripts/migrate.sh history      # Show migration history
#   ./scripts/migrate.sh reset        # Reset database (DANGER!)
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect if we're running in Docker or locally
if [ -f /.dockerenv ]; then
    IN_DOCKER=true
    RUN_CMD="uv run"
else
    IN_DOCKER=false
    # Check if docker-compose is available
    if command -v docker-compose &> /dev/null; then
        DOCKER_AVAILABLE=true
    else
        DOCKER_AVAILABLE=false
    fi
fi

# Function to run alembic command
run_alembic() {
    if [ "$IN_DOCKER" = true ]; then
        # Already in Docker, run directly
        $RUN_CMD alembic "$@"
    elif [ "$DOCKER_AVAILABLE" = true ]; then
        # Try to run via docker-compose, fall back to local if service not running
        echo -e "${BLUE}Checking if langgraph-server is running...${NC}"
        if docker-compose ps langgraph-server | grep -q "Up"; then
            echo -e "${BLUE}Running in Docker container...${NC}"
            docker-compose exec langgraph-server uv run alembic "$@"
        else
            echo -e "${YELLOW}langgraph-server not running, executing locally...${NC}"
            uv run alembic "$@"
        fi
    else
        # Run locally
        echo -e "${BLUE}Running locally...${NC}"
        uv run alembic "$@"
    fi
}

# Function to ensure database is ready
ensure_db_ready() {
    echo -e "${BLUE}Checking database connection...${NC}"
    if [ "$DOCKER_AVAILABLE" = true ] && [ "$IN_DOCKER" = false ]; then
        # Start postgres if not running
        docker-compose up -d postgres
        echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"
        sleep 3
    fi
}

# Main command handling
COMMAND=${1:-upgrade}

case "$COMMAND" in
    "upgrade"|"up"|"migrate")
        echo -e "${GREEN}🔄 Running database migrations...${NC}"
        ensure_db_ready
        run_alembic upgrade head
        echo -e "${GREEN}✅ Migrations completed successfully!${NC}"
        ;;

    "create"|"new"|"revision")
        MESSAGE=${2:-"migration"}
        echo -e "${GREEN}📝 Creating new migration: ${MESSAGE}${NC}"
        ensure_db_ready
        run_alembic revision --autogenerate -m "$MESSAGE"
        echo -e "${GREEN}✅ Migration file created!${NC}"
        echo -e "${YELLOW}📍 Review the generated migration in alembic/versions/${NC}"
        ;;

    "rollback"|"downgrade"|"down")
        STEPS=${2:--1}
        echo -e "${YELLOW}⬇️  Rolling back $STEPS migration(s)...${NC}"
        ensure_db_ready
        run_alembic downgrade "$STEPS"
        echo -e "${GREEN}✅ Rollback completed!${NC}"
        ;;

    "history"|"log"|"show")
        echo -e "${BLUE}📜 Migration history:${NC}"
        run_alembic history --verbose
        ;;

    "current"|"status")
        echo -e "${BLUE}📍 Current migration status:${NC}"
        run_alembic current
        ;;

    "reset"|"drop")
        echo -e "${RED}⚠️  WARNING: This will DROP all tables and re-run migrations!${NC}"
        echo -e "${RED}This action cannot be undone!${NC}"
        read -p "Are you sure? Type 'YES' to confirm: " -r
        if [ "$REPLY" = "YES" ]; then
            echo -e "${YELLOW}Dropping all tables...${NC}"
            ensure_db_ready
            run_alembic downgrade base
            echo -e "${YELLOW}Re-running migrations...${NC}"
            run_alembic upgrade head
            echo -e "${GREEN}✅ Database reset complete!${NC}"
        else
            echo -e "${BLUE}Cancelled.${NC}"
        fi
        ;;

    "help"|"-h"|"--help")
        echo "Database Migration Helper"
        echo ""
        echo "Usage: ./scripts/migrate.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  upgrade, up, migrate          Run all pending migrations (default)"
        echo "  create, new, revision [name]  Create a new migration"
        echo "  rollback, down [steps]        Rollback migrations (default: -1)"
        echo "  history, log, show            Show migration history"
        echo "  current, status               Show current migration status"
        echo "  reset, drop                   Drop all tables and re-run migrations (DANGER!)"
        echo "  help, -h, --help              Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./scripts/migrate.sh                          # Run migrations"
        echo "  ./scripts/migrate.sh create add_user_fields   # Create new migration"
        echo "  ./scripts/migrate.sh rollback                 # Rollback last migration"
        echo "  ./scripts/migrate.sh history                  # Show migration history"
        ;;

    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo "Run './scripts/migrate.sh help' for usage information"
        exit 1
        ;;
esac
