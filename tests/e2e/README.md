# E2E API Tests

This directory contains end-to-end (E2E) tests for the Seer API endpoints.

## Overview

E2E tests validate complete HTTP request/response cycles including:
- Authentication and authorization
- Request validation
- Business logic execution
- Database operations
- Response formatting

## Test Structure

```
tests/e2e/
├── conftest.py              # E2E-specific fixtures (authenticated client, full app)
├── workflows/               # Workflow API tests (~102 tests)
│   ├── test_workflow_crud_api.py          # CRUD operations (20 tests)
│   ├── test_workflow_validation_api.py    # Validation & compilation (16 tests)
│   ├── test_workflow_execution_api.py     # Execution & runs (20 tests)
│   ├── test_workflow_versioning_api.py    # Drafts & versions (26 tests)
│   ├── test_trigger_api.py                # Trigger management (13 tests)
│   └── test_registry_api.py               # Registries & schemas (15 tests)
└── tools/                   # Tools API tests (~16 tests)
    └── test_tools_api.py                  # Tool listing & execution
```

## Running Tests

### All E2E Tests
```bash
uv run pytest tests/e2e/ -v
```

### Specific Module
```bash
# Workflow tests
uv run pytest tests/e2e/workflows/ -v

# Tools tests
uv run pytest tests/e2e/tools/ -v
```

### Single Test File
```bash
uv run pytest tests/e2e/workflows/test_workflow_crud_api.py -v
```

### With Coverage
```bash
uv run pytest tests/e2e/ --cov --cov-report=html
open htmlcov/index.html
```

## Key Features

### Authentication Setup
- Tests run in **self-hosted mode** (no Clerk authentication)
- JWT tokens are created for test users
- `TokenDecodeWithoutValidationMiddleware` decodes tokens without verification
- Module cache is cleared to ensure correct mode

### Database Setup
- SQLite in-memory database (via `db_engine` fixture)
- Fast test execution (~100x faster than PostgreSQL)
- Perfect isolation between tests

### API Paths
- Base path: `/api`
- Workflows: `/api/v1/*`
- Tools: `/api/tools/*`

## Fixtures

### E2E-Specific Fixtures (conftest.py)

#### `full_app`
Full FastAPI application with all routers and middleware. Sets `SEER_MODE=self-hosted` before import.

#### `authenticated_e2e_client`
HTTP client with JWT token for authenticated requests.

```python
async def test_example(db_engine, authenticated_e2e_client):
    response = await authenticated_e2e_client.get("/api/v1/workflows")
    assert response.status_code == 200
```

#### `e2e_client`
HTTP client without authentication (for testing 401 responses).

```python
async def test_unauthorized(db_engine, e2e_client):
    response = await e2e_client.get("/api/v1/workflows")
    assert response.status_code == 401
```

#### `workflow_create_payload`
Valid payload for creating a workflow.

#### `workflow_run_payload`
Valid payload for executing a workflow.

## Test Patterns

### Standard E2E Test
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_create_workflow(db_engine, authenticated_e2e_client, workflow_create_payload):
    """Test successful workflow creation."""
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json=workflow_create_payload
    )

    assert response.status_code == 201
    data = response.json()
    assert "workflow_id" in data
    assert data["name"] == workflow_create_payload["name"]
```

### Testing Unauthorized Access
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_unauthorized(db_engine, e2e_client):
    """Test endpoint without authentication returns 401."""
    response = await e2e_client.get("/api/v1/workflows")
    assert response.status_code == 401
```

### Testing Not Found
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_not_found(db_engine, authenticated_e2e_client):
    """Test retrieving non-existent resource returns 404."""
    response = await authenticated_e2e_client.get("/api/v1/workflows/nonexistent_id")
    assert response.status_code == 404
```

### Testing Validation Errors
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_invalid_input(db_engine, authenticated_e2e_client):
    """Test invalid input returns 422."""
    response = await authenticated_e2e_client.post(
        "/api/v1/workflows",
        json={"invalid": "payload"}
    )
    assert response.status_code == 422
```

## Coverage Goals

- **Target**: 75% coverage for API endpoints
- **Workflows API**: ~102 tests covering all major endpoints
- **Tools API**: ~16 tests covering tool operations

## Troubleshooting

### Auth Errors (401)
- Ensure `SEER_MODE=self-hosted` is set in environment
- Check that module cache is being cleared in `full_app` fixture
- Verify JWT token is being created correctly

### Routes Not Found (404)
- Verify API path is correct: `/api/v1/*` not `/v1/*`
- Check that router is included in main app
- Ensure middleware isn't blocking the request

### Database Errors
- Ensure `db_engine` fixture is requested in test parameters
- Check that Tortoise ORM is initialized correctly
- Verify SQLite in-memory database is working

### Spec Normalization
- API may normalize workflow specs (adding default fields like `ui: {}`, `inputs: {}`)
- Don't assert exact spec equality - check key fields exist instead
- Example:
  ```python
  # ❌ Don't do this
  assert data["spec"] == payload["spec"]

  # ✅ Do this instead
  assert data["spec"]["version"] == payload["spec"]["version"]
  assert len(data["spec"]["nodes"]) == len(payload["spec"]["nodes"])
  ```

## Next Steps

### Additional Modules to Test
- Integrations API (`/api/integrations/*`)
- Usage API (`/api/usage/*`)
- User Settings API (`/api/settings/*`)
- Webhooks API (`/api/webhooks/*`)
- Forms API (`/api/forms/*`)

### Future Improvements
- Add performance benchmarks
- Test concurrent requests
- Add test data factories
- Improve error message assertions
- Add WebSocket testing (for streaming)
