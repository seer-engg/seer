# Seer Test Suite

Comprehensive test suite for the Seer workflow automation platform, covering unit tests, integration tests, and end-to-end API tests.

## Directory Structure

```
tests/
├── conftest.py              # Global fixtures (DB, app, auth)
├── fixtures/                # Test utilities and helpers
│   ├── workflow_specs.py    # Workflow specification builders
│   ├── mock_tools.py        # Mock tool implementations
│   └── factories.py         # Model factories
├── unit/                    # Fast unit tests (~70% of tests)
│   ├── conftest.py         # Unit test fixtures
│   ├── core/               # Compiler, runtime, expression evaluator
│   ├── tools/              # Tool registry, credential resolver
│   ├── services/           # Business logic
│   └── observability/      # Credits, usage limits
├── integration/            # Integration tests (~20% of tests)
│   ├── conftest.py        # Integration test fixtures
│   ├── database/          # ORM models, relationships
│   ├── tools/             # Tool execution with credentials
│   ├── triggers/          # Polling system
│   └── worker/            # Taskiq background tasks
└── e2e/                   # API endpoint tests (~10% of tests)
    ├── conftest.py        # E2E test fixtures
    ├── test_workflow_api.py
    ├── test_execution_api.py
    └── test_trigger_api.py
```

## Test Categories

### Unit Tests (`tests/unit/`)

Fast tests that verify individual components in isolation with mocked dependencies.

- **Purpose**: Test pure functions, business logic, and algorithms
- **Speed**: <1s per test
- **Database**: Not required (uses mocks)
- **Marker**: `@pytest.mark.unit` (optional, default if no marker)

**Run with:**
```bash
uv run pytest tests/unit -m unit
```

**Examples:**
- Expression evaluator tests
- Workflow compiler validation
- Credit calculator tests
- Usage limit configuration tests

### Integration Tests (`tests/integration/`)

Tests that verify component interactions with real database, Valkey, or other services.

- **Purpose**: Test database operations, tool execution, trigger polling
- **Speed**: 1-5s per test
- **Database**: SQLite in-memory with transaction rollback
- **Marker**: `@pytest.mark.integration` (required)

**Run with:**
```bash
uv run pytest tests/integration -m integration
```

**Examples:**
- Database model CRUD operations
- Tool execution with credential resolution
- Trigger subscription polling
- Worker task execution

### End-to-End Tests (`tests/e2e/`)

Tests that verify complete API flows including authentication, validation, and persistence.

- **Purpose**: Test full API endpoints with middleware stack
- **Speed**: 2-10s per test
- **Database**: SQLite in-memory with transaction rollback
- **Marker**: `@pytest.mark.e2e` (required)

**Run with:**
```bash
uv run pytest tests/e2e -m e2e
```

**Examples:**
- Workflow creation via API
- Workflow execution via API
- Trigger subscription management
- Usage and billing endpoints

## Running Tests

### All Tests
```bash
uv run pytest
```

### By Category
```bash
# Unit tests only (fast)
uv run pytest tests/unit -m unit

# Integration tests
uv run pytest tests/integration -m integration

# E2E tests
uv run pytest tests/e2e -m e2e
```

### With Coverage
```bash
# All tests with coverage
uv run pytest --cov --cov-report=html
open htmlcov/index.html

# Specific category with coverage
uv run pytest tests/unit -m unit --cov=src/seer/core --cov-report=term
```

### Parallel Execution
```bash
# Run tests in parallel for speed
uv run pytest tests/unit -m unit -n auto
```

### Verbose Output
```bash
# Show more details
uv run pytest tests/unit -v

# Show local variables on failure
uv run pytest tests/unit --showlocals
```

## Writing Tests

### Test Naming Convention

- **Files**: `test_*.py`
- **Functions**: `test_<what_is_being_tested>()`
- **Classes**: `Test<ComponentName>` (optional, for grouping)

### Using Fixtures

Global fixtures are available from `tests/conftest.py`:

```python
@pytest.mark.asyncio
async def test_create_workflow(test_user, sample_workflow_spec):
    """Test creating a workflow with valid spec."""
    workflow = await Workflow.create(
        user=test_user,
        workflow_id="wf_test",
        name="Test Workflow",
        spec=sample_workflow_spec,
    )
    assert workflow.id is not None
```

### Using Workflow Builders

```python
from tests.fixtures.workflow_specs import WorkflowSpecBuilder

def test_complex_workflow():
    spec = (
        WorkflowSpecBuilder()
        .add_trigger("t1", "test.trigger")
        .add_task_node("n1", "test.tool", {"param": "value"})
        .add_condition_node("c1", "${n1.result.success}")
        .add_edge("e1", "t1", "n1")
        .add_edge("e2", "n1", "c1")
        .build()
    )
    # Test with spec...
```

### Using Mock Tools

```python
from tests.fixtures.mock_tools import create_mock_tool

@pytest.mark.asyncio
async def test_tool_execution():
    tool = create_mock_tool(
        tool_id="test.api_call",
        result={"status": "success", "data": {"value": 42}}
    )

    result = await tool.execute({"param": "value"})
    assert result["status"] == "success"
    assert tool.execution_count == 1
```

### Using Factories

```python
from tests.fixtures.factories import UserFactory, WorkflowFactory

@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_creation():
    user = await UserFactory.create(email="test@example.com")
    workflow = await WorkflowFactory.create(user=user, name="Test Workflow")

    assert workflow.user_id == user.id
    assert workflow.name == "Test Workflow"
```

## Test Markers

Tests should be marked with appropriate markers:

```python
@pytest.mark.unit
def test_pure_function():
    """Fast unit test with no external dependencies."""
    pass

@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_operation():
    """Integration test requiring database."""
    pass

@pytest.mark.e2e
@pytest.mark.asyncio
async def test_api_endpoint(e2e_client):
    """End-to-end API test."""
    pass

@pytest.mark.slow
@pytest.mark.requires_external
def test_external_api():
    """Test requiring external service (may be skipped in CI)."""
    pass
```

## Coverage Goals

| Component | Target | Rationale |
|-----------|--------|-----------|
| Core Compiler | 90% | Critical path, pure functions |
| Expression Evaluator | 95% | Complex logic, many edge cases |
| Runtime Nodes | 85% | Core execution |
| Database Models | 70% | CRUD + relationships |
| API Routers | 75% | Happy paths + errors |
| Worker Tasks | 70% | Business logic |
| Trigger Polling | 65% | Complex concurrency |
| **Overall** | **75%** | Balanced rigor |

## CI/CD Integration

Tests run automatically on every push via GitHub Actions:

1. **Unit tests** - Run first (fast feedback)
2. **Integration tests** - Run with PostgreSQL service
3. **E2E tests** - Run with full stack
4. **Coverage check** - Fails if coverage <70%

## Troubleshooting

### Tests fail with database errors

Ensure the database transaction fixture is being used:

```python
@pytest.mark.asyncio
async def test_my_test(db_transaction):  # Add this fixture
    # Your test code...
```

### Tests hang or timeout

Check for:
- Missing `@pytest.mark.asyncio` decorator
- Unclosed async contexts
- Infinite loops in test logic

### Import errors

Ensure you're running tests from the project root:

```bash
cd /path/to/seer
uv run pytest tests/
```

### Coverage not calculated

Ensure pytest-cov is installed:

```bash
uv sync --all-extras
```

## Best Practices

1. **Keep tests isolated** - Each test should be independent
2. **Use fixtures liberally** - Don't repeat setup code
3. **Test edge cases** - Not just happy paths
4. **Use descriptive names** - Tests are documentation
5. **Keep tests fast** - Prefer unit tests over integration tests
6. **Mock external services** - Don't depend on external APIs
7. **Use factories** - For complex object creation
8. **Clean up resources** - Let fixtures handle cleanup

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest-asyncio](https://pytest-asyncio.readthedocs.io/)
- [FastAPI testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [Tortoise ORM testing](https://tortoise.github.io/testing.html)
