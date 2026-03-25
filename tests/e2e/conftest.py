# pylint: disable=import-outside-toplevel,redefined-outer-name,unused-import
# Reason: Test fixtures use lazy imports; pytest fixture pattern requires name reuse; imports used by pytest
"""
Main conftest for E2E tests with Testcontainers.

IMPORTANT: Environment variables MUST be set BEFORE any seer imports.
This ensures the config singleton is initialized with test values.

This module:
1. Sets test environment variables
2. Imports all fixture modules to make them available
3. Configures pytest markers
"""
# =============================================================================
# ENVIRONMENT SETUP - Must happen FIRST before any seer imports
# =============================================================================
import os
import sys

# CRITICAL: Prevent AWS parameter store loading
# Set invalid AWS credentials to make boto3 fail gracefully
os.environ["AWS_ACCESS_KEY_ID"] = "testing"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
os.environ["AWS_SECURITY_TOKEN"] = "testing"
os.environ["AWS_SESSION_TOKEN"] = "testing"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
# Disable the AWS endpoint to prevent any AWS calls
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:0"

# CRITICAL: Prevent AWS parameter store loading by setting a dummy DATABASE_URL early
# This prevents the config singleton from trying to fetch from AWS when imported
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/seer_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

# Disable analytics/observability
os.environ["POSTHOG_ENABLED"] = "false"
os.environ["SENTRY_ENABLED"] = "false"
os.environ["LANGFUSE_ENABLED"] = "false"

# Disable background processes
os.environ["TRIGGER_POLLER_ENABLED"] = "false"
os.environ["TOOL_INDEX_AUTO_GENERATE"] = "false"

# Set mode to self-hosted for simpler auth
os.environ["CLOUD_MODE"] = "false"
os.environ["IS_CLOUD_MODE"] = "false"
os.environ["SEER_MODE"] = "self-hosted"

# Database connection pool settings
os.environ["DB_MAX_CONNECTIONS"] = "5"
os.environ["DB_MIN_CONNECTIONS"] = "1"

# Disable AWS parameter loading by ensuring we're not in AWS mode
os.environ["AWS_PARAMETER_PATH"] = ""
os.environ["SSM_ENABLED"] = "false"

# Disable MCP - it mounts at root and catches all requests
os.environ["MCP_ENABLED"] = "false"

# Enable user emulation for E2E tests (bypasses Clerk auth)
os.environ["ENABLE_USER_EMULATION"] = "true"

# Set dummy Clerk credentials (required for middleware initialization even when emulation is enabled)
os.environ["CLERK_JWKS_URL"] = "https://test.clerk.accounts.dev/.well-known/jwks.json"
os.environ["CLERK_ISSUER"] = "https://test.clerk.accounts.dev"

# =============================================================================
# FIXTURE IMPORTS - Only import fixtures that don't trigger seer imports
# =============================================================================
import pytest

# Container fixtures (Postgres, Redis) - these don't import seer modules
from tests.e2e.fixtures.containers import (
    postgres_container,
    redis_container,
    database_url,
    redis_url,
)

# Database fixtures - these use lazy imports internally
from tests.e2e.fixtures.database import (
    db_initialized,
    db_session,
    clean_db_session,
)

# NOTE: Broker and API client fixtures are NOT imported here to avoid
# triggering seer module imports before the test environment is set up.
# They are imported lazily in the test functions that need them.
# Pytest will discover them from the fixtures directory.


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for async tests."""
    return "asyncio"


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "e2e: E2E tests with Testcontainers (Postgres + Redis)",
    )


def pytest_collection_modifyitems(config, items):
    """Auto-mark all tests in this directory with e2e marker."""
    for item in items:
        if "/e2e/" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)


# =============================================================================
# SESSION CLEANUP
# =============================================================================


def pytest_sessionfinish(session, exitstatus):
    """Cleanup resources at end of test session."""
    import threading

    # Shutdown PostHog if initialized
    try:
        import posthog
        posthog.shutdown()
    except Exception:
        pass

    # Reset observability state
    try:
        import seer.observability.sentry_client as sentry_client
        sentry_client.SENTRY_INITIALIZED = False
    except Exception:
        pass

    try:
        import seer.observability.posthog_client as posthog_client
        posthog_client.POSTHOG_INITIALIZED = False
    except Exception:
        pass


# =============================================================================
# WORKFLOW SPEC FIXTURES
# =============================================================================


@pytest.fixture
def simple_tool_workflow_spec() -> dict:
    """
    Simple workflow with an HTTP request tool for basic execution tests.

    Uses the http_request tool which is always registered and doesn't
    require OAuth credentials (makes GET requests).
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "n1",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get",
                },
            }
        ],
        "edges": [],
    }


@pytest.fixture
def conditional_workflow_spec() -> dict:
    """
    Workflow with conditional branching for control flow tests.

    Tests the if/else branching logic in workflow execution.
    Uses an if node that doesn't require external tools.
    """
    return {
        "version": "2",
        "triggers": [],
        "nodes": [
            {
                "id": "condition",
                "type": "if",
                "condition": "true",
            },
            {
                "id": "success_path",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?result=success",
                },
            },
            {
                "id": "failure_path",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get?result=failure",
                },
            },
        ],
        "edges": [
            {"source": "condition", "target": "success_path", "type": "conditional_true"},
            {"source": "condition", "target": "failure_path", "type": "conditional_false"},
        ],
    }


@pytest.fixture
def webhook_trigger_workflow_spec() -> dict:
    """
    Workflow with webhook trigger for trigger execution tests.

    Uses the generic webhook trigger that accepts any payload.
    Uses a simple GET request to avoid body serialization issues.
    """
    return {
        "version": "2",
        "triggers": [
            {
                "id": "webhook",
                "key": "webhook.generic",
                "mode": "webhook",
                "provider_config": {},
            }
        ],
        "nodes": [
            {
                "id": "process",
                "type": "tool",
                "tool": "http_request",
                "inputs": {
                    "method": "GET",
                    "url": "https://httpbin.org/get",
                },
            }
        ],
        "edges": [
            {"source": "webhook", "target": "process", "type": "trigger"},
        ],
    }


# =============================================================================
# API CLIENT FIXTURES - Defined here with lazy imports to avoid early seer imports
# =============================================================================


@pytest.fixture(scope="function")
async def e2e_app(database_url: str, redis_url: str):
    """
    Create FastAPI application configured for E2E testing.

    Sets environment variables to point to test containers before
    importing the app, ensuring the config singleton uses test values.
    """
    # CRITICAL: Set environment BEFORE importing app
    os.environ["DATABASE_URL"] = database_url
    os.environ["REDIS_URL"] = redis_url
    os.environ["SEER_MODE"] = "self-hosted"
    os.environ["CLOUD_MODE"] = "false"
    os.environ["IS_CLOUD_MODE"] = "false"
    os.environ["POSTHOG_ENABLED"] = "false"
    os.environ["SENTRY_ENABLED"] = "false"
    os.environ["LANGFUSE_ENABLED"] = "false"
    os.environ["TRIGGER_POLLER_ENABLED"] = "false"
    os.environ["TOOL_INDEX_AUTO_GENERATE"] = "false"
    os.environ["DB_MAX_CONNECTIONS"] = "5"
    os.environ["DB_MIN_CONNECTIONS"] = "1"
    os.environ["AWS_PARAMETER_PATH"] = ""
    os.environ["SSM_ENABLED"] = "false"
    # Disable MCP - it mounts at root and catches all requests
    os.environ["MCP_ENABLED"] = "false"
    # Disable auto browser opening
    os.environ["AUTO_OPEN_BROWSER"] = "false"
    # Set ENV to test to avoid production AWS path
    os.environ["ENV"] = "test"
    # Enable user emulation for E2E tests (bypasses Clerk auth)
    os.environ["ENABLE_USER_EMULATION"] = "true"
    # Set dummy Clerk credentials (required for middleware initialization)
    os.environ["CLERK_JWKS_URL"] = "https://test.clerk.accounts.dev/.well-known/jwks.json"
    os.environ["CLERK_ISSUER"] = "https://test.clerk.accounts.dev"

    # Clear ALL seer modules from cache to ensure fresh config
    modules_to_clear = [key for key in list(sys.modules.keys()) if key.startswith("seer")]
    for module in modules_to_clear:
        del sys.modules[module]

    # Now import the app with correct environment
    from seer.api.main import app

    # Disable lifespan to avoid database initialization conflicts
    # Both attributes must be set to None
    app.router.lifespan_context = None
    app.router.lifespan = None

    return app


@pytest.fixture(scope="function")
async def e2e_client(e2e_app):
    """Unauthenticated async HTTP client for E2E testing."""
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
async def e2e_test_user(db_session):
    """Create a test user in the real database with unique ID and personal organization.

    Creates both the user and their personal organization upfront to avoid
    race conditions when concurrent API requests hit the OrganizationContextMiddleware.
    """
    from datetime import datetime, timezone
    import uuid
    from seer.database.models import User
    from seer.services.organization_service import create_personal_organization

    # Use UUID to ensure unique user for each test
    unique_id = str(uuid.uuid4())[:8]
    user = await User.create(
        user_id=f"e2e_user_{unique_id}",
        email=f"e2e_test_{unique_id}@example.com",
        first_name="E2E",
        last_name="Tester",
        created_at=datetime.now(timezone.utc),
    )

    # Pre-create the personal organization to avoid race conditions
    # when concurrent requests hit the OrganizationContextMiddleware
    await create_personal_organization(user)

    return user


@pytest.fixture(scope="function")
async def authenticated_e2e_client(e2e_app, e2e_test_user):
    """
    Authenticated async HTTP client for E2E testing.

    Uses user emulation mode with X-Emulate-User-Id header to bypass Clerk auth.
    This is cleaner than JWT tokens for E2E tests since we control the test user.
    """
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Use user emulation header (enabled via ENABLE_USER_EMULATION=true)
        client.headers["X-Emulate-User-Id"] = e2e_test_user.user_id
        yield client


# =============================================================================
# TASKIQ FIXTURES - Defined here with lazy imports
# =============================================================================


@pytest.fixture(scope="function")
async def e2e_checkpointer(database_url: str, e2e_app):
    """
    Initialize LangGraph checkpointer for E2E tests.

    The checkpointer requires a PostgreSQL connection pool and setup.
    This fixture mirrors the production checkpointer_lifespan() but
    uses the Testcontainer's PostgreSQL instance.

    It also patches the global checkpointer singleton so that service-layer
    code calling get_checkpointer() receives the test checkpointer.
    """
    from psycopg.rows import dict_row
    from psycopg_pool import AsyncConnectionPool
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    from seer.api.agents import checkpointer as checkpointer_module

    pool = AsyncConnectionPool(
        conninfo=database_url,
        max_size=5,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
            "prepare_threshold": 0,
        },
    )
    await pool.open()

    saver = AsyncPostgresSaver(pool)
    await saver.setup()  # Creates checkpoint tables

    # Store in app state (mirrors production behavior)
    e2e_app.state.checkpointer = saver

    # Patch the global singleton so get_checkpointer() returns our test instance
    original_checkpointer = checkpointer_module._checkpointer
    checkpointer_module._checkpointer = saver

    yield saver

    # Restore original state
    checkpointer_module._checkpointer = original_checkpointer
    await pool.close()


@pytest.fixture(scope="function")
def taskiq_direct_executor(e2e_app):
    """
    Patch Taskiq tasks to execute directly in-process.

    Depends on e2e_app to ensure seer modules are imported with correct config.
    """
    from typing import Any, Dict, List
    from unittest.mock import AsyncMock, patch

    class TaskExecutionTracker:
        def __init__(self):
            self.executed_tasks: List[Dict[str, Any]] = []

        def record_execution(self, task_name, args, kwargs, result=None, error=None):
            self.executed_tasks.append({
                "task_name": task_name,
                "args": args,
                "kwargs": kwargs,
                "result": result,
                "error": str(error) if error else None,
            })

        def get_executions(self, task_name):
            return [t for t in self.executed_tasks if t["task_name"] == task_name]

        def clear(self):
            self.executed_tasks.clear()

    def create_direct_kiq_wrapper(original_task, tracker):
        async def direct_kiq(*args, **kwargs):
            task_name = getattr(original_task, "task_name", original_task.__name__)
            mock_result = AsyncMock()
            mock_result.task_id = f"mock_task_{len(tracker.executed_tasks)}"
            mock_result.is_ready = True

            try:
                # Taskiq uses 'original_func' to store the wrapped function
                task_fn = getattr(original_task, "original_func", None)
                if task_fn is None:
                    raise AttributeError(f"Task {task_name} has no original_func attribute")
                result = await task_fn(*args, **kwargs)
                tracker.record_execution(task_name, args, kwargs, result=result)
                mock_result.result = result
                mock_result.error = None
            except Exception as e:
                tracker.record_execution(task_name, args, kwargs, error=e)
                mock_result.result = None
                mock_result.error = e
                raise

            async def wait_result(timeout=None):
                return mock_result.result
            mock_result.wait_result = wait_result

            return mock_result
        return direct_kiq

    tracker = TaskExecutionTracker()
    patches = []

    # Import tasks AFTER e2e_app has set up the environment
    from seer.worker.tasks.workflows import workflow_execution_task
    from seer.worker.tasks.triggers import trigger_event_task

    workflow_kiq_wrapper = create_direct_kiq_wrapper(workflow_execution_task, tracker)
    workflow_patch = patch.object(workflow_execution_task, "kiq", workflow_kiq_wrapper)
    patches.append(workflow_patch)

    trigger_kiq_wrapper = create_direct_kiq_wrapper(trigger_event_task, tracker)
    trigger_patch = patch.object(trigger_event_task, "kiq", trigger_kiq_wrapper)
    patches.append(trigger_patch)

    for p in patches:
        p.start()

    yield {"tracker": tracker, "patches": patches}

    for p in patches:
        p.stop()
