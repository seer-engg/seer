# E2E Tests

True end-to-end tests using real infrastructure via Testcontainers.

## Overview

These tests validate complete API → Database → Worker → Database flows using:
- **Real PostgreSQL** with pgvector extension (via Testcontainers)
- **Real Redis/Valkey** (via Testcontainers)
- **In-process Taskiq execution** for deterministic task testing

## Prerequisites

- Docker running locally (Testcontainers requires Docker)
- All dev dependencies installed: `uv sync --group dev`

## Running Tests

```bash
# Run all e2e tests
uv run pytest tests/e2e -v

# Run specific test file
uv run pytest tests/e2e/workflows/test_workflow_lifecycle.py -v

# Run with verbose output and show locals on failure
uv run pytest tests/e2e -v --tb=long --showlocals

# Run only e2e marked tests (if mixed with other tests)
uv run pytest -m e2e -v
```

## Architecture

```
tests/e2e/
├── conftest.py              # Main conftest, env setup, workflow spec fixtures
├── fixtures/
│   ├── __init__.py
│   ├── containers.py        # Testcontainers (Postgres, Redis)
│   ├── database.py          # DB initialization, session management
│   ├── broker.py            # Taskiq in-process execution
│   └── api_client.py        # FastAPI app, HTTP clients
├── workflows/
│   ├── test_workflow_lifecycle.py  # Create → Publish → Execute
│   ├── test_trigger_execution.py   # Webhook/polling triggers
│   └── test_error_handling.py      # Failure scenarios
└── README.md
```

## Key Fixtures

### Container Fixtures (session-scoped)
- `postgres_container` - PostgreSQL with pgvector
- `redis_container` - Valkey (Redis-compatible)
- `database_url` - Connection string for Postgres
- `redis_url` - Connection string for Redis

### Database Fixtures
- `db_initialized` - Runs migrations once per session
- `db_session` - Function-scoped with transaction rollback

### Broker Fixtures
- `taskiq_direct_executor` - Executes tasks in-process, tracks executions

### Client Fixtures
- `e2e_app` - FastAPI app configured for testing
- `e2e_client` - Unauthenticated HTTP client
- `authenticated_e2e_client` - Client with JWT auth headers
- `e2e_test_user` - Test user in real database

## How It Works

### Container Lifecycle
1. Containers start once per test session (first test)
2. pgvector extension is enabled on Postgres
3. Migrations run once to set up schema
4. Tests use transaction rollback for isolation
5. Containers stop when session ends

### Task Execution
Instead of queueing tasks to Redis and running a separate worker:
1. We patch `task.kiq()` to call the task function directly
2. Tasks execute in the same async context as the test
3. `TaskExecutionTracker` records all executions for assertions

### Database Isolation
- Each test runs within a database transaction
- Transaction is rolled back after the test
- No data persists between tests
- Much faster than truncating/recreating tables

## Writing Tests

```python
import pytest

pytestmark = pytest.mark.e2e  # Auto-applied by conftest

async def test_workflow_lifecycle(
    authenticated_e2e_client,  # HTTP client with auth
    db_session,               # Ensures DB is ready
    taskiq_direct_executor,   # Enables task execution tracking
    simple_tool_workflow_spec,  # Sample workflow spec
):
    # Create workflow
    response = await authenticated_e2e_client.post(
        "/v1/workflows",
        json={"name": "Test", "spec": simple_tool_workflow_spec}
    )
    assert response.status_code == 201

    # Verify task was executed
    tracker = taskiq_direct_executor["tracker"]
    executions = tracker.get_executions("workflow_execution_task")
    # ...
```

## Troubleshooting

### Containers won't start
- Ensure Docker daemon is running: `docker info`
- Check for port conflicts on 5432 and 6379
- Try `docker system prune` to clear stale resources

### Migrations fail
- The fixture falls back to `generate_schemas()` if aerich fails
- Check migration files for syntax errors
- Ensure all model imports are correct

### Tests hang
- Check for async deadlocks in task execution
- The broker fixture may need timeout adjustment
- Use `pytest --timeout=60` to add test timeouts

### Import errors
- Ensure environment variables are set before imports
- The conftest sets these at module load time
- Clear Python's module cache if needed
