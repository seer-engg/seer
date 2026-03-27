# pylint: disable=import-outside-toplevel,redefined-outer-name,reimported,unused-import
# Reason: Test fixtures commonly use lazy imports and pytest fixture pattern requires name reuse;
# unused imports are pytest fixtures discovered via import
"""
Global test fixtures for Seer test suite.

Provides core fixtures for:
- Async testing with anyio backend (FastAPI recommendation)
- Database setup with PostgreSQL via Testcontainers (production-identical)
- Truncation-based test isolation (handles IntegrityError-based dedup correctly)
- Test user creation
- FastAPI test client
- Sample workflow specifications
"""
# =============================================================================
# IMPORTANT: Set environment variables BEFORE any seer imports to ensure
# the config singleton is initialized with test values.
# =============================================================================
import os
os.environ["POSTHOG_ENABLED"] = "false"
os.environ["SENTRY_ENABLED"] = "false"
os.environ["TRIGGER_POLLER_ENABLED"] = "false"
os.environ["TOOL_INDEX_AUTO_GENERATE"] = "false"
os.environ["CLOUD_MODE"] = "false"
os.environ["IS_CLOUD_MODE"] = "false"
os.environ["SEER_MODE"] = "self-hosted"
os.environ["DB_MAX_CONNECTIONS"] = "5"
os.environ["DB_MIN_CONNECTIONS"] = "1"
os.environ.setdefault("DATABASE_URL", "postgresql://test:test@localhost:5432/seer_test")

from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

# =============================================================================
# Shared Testcontainers + Database Fixtures (PostgreSQL)
# These are imported so pytest discovers them as available fixtures.
# =============================================================================
from tests.fixtures.containers import (  # noqa: F401
    postgres_container,
    database_url,
)
from tests.fixtures.database import (  # noqa: F401
    db_initialized,
    db_engine,
)


# =============================================================================
# Logger Cache Reset
# =============================================================================


@pytest.fixture(autouse=True)
def reset_logger_cache():
    """Reset the logger cache before each test to ensure test isolation.

    The seer.logger module caches logger instances in _loggers dict.
    Without clearing this cache, tests may share logger instances with
    different handlers attached, causing caplog to not capture logs correctly.
    """
    from seer import logger as seer_logger
    seer_logger._loggers.clear()
    yield
    seer_logger._loggers.clear()


# =============================================================================
# Async Testing Configuration
# =============================================================================


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for anyio tests (FastAPI recommendation)."""
    return "asyncio"


# Note: db_engine fixture is imported from tests.fixtures.database (PostgreSQL via Testcontainers).
# It is NOT autouse - tests that need database should explicitly request it via parameters.


@pytest.fixture(autouse=True)
async def reset_checkpointer():
    """
    Reset checkpointer singleton between tests.

    The checkpointer uses a global singleton bound to one event loop.
    With asyncio_default_fixture_loop_scope="function", each test gets
    a new loop, causing hangs if the checkpointer isn't reset.

    This is autouse to ensure all tests get a fresh checkpointer state.
    Without this, tests hang when run without pytest-xdist (-n flag).
    """
    # Clear before test
    try:
        import seer.api.agents.checkpointer as checkpointer_module
        checkpointer_module._checkpointer = None
        checkpointer_module._checkpointer_cm = None
    except ImportError:
        pass

    yield

    # Clear after test (for safety)
    try:
        import seer.api.agents.checkpointer as checkpointer_module
        checkpointer_module._checkpointer = None
        checkpointer_module._checkpointer_cm = None
    except ImportError:
        pass


# =============================================================================
# User Fixtures
# =============================================================================


@pytest.fixture
async def test_user():
    """
    Create authenticated test user for tests requiring authentication.

    Returns:
        User: Test user with standard attributes
    """
    from datetime import datetime, timezone
    from seer.database.models import User

    user = await User.create(
        user_id="test_user_123",
        email="test@example.com",
        first_name="Test",
        last_name="User",
        created_at=datetime.now(timezone.utc),
    )
    return user


@pytest.fixture
async def test_user_with_subscription(test_user):
    """
    Create test user with active PRO subscription via their personal org.

    Useful for testing subscription-dependent features.
    """
    from datetime import datetime, timedelta, timezone
    from seer.database.organization_models import Organization, OrganizationType
    from seer.database.subscription_models import (
        BillingSubscription,
        SubscriptionStatus,
        SubscriptionTier,
    )

    # Get or create personal organization for user
    personal_org, _ = await Organization.get_or_create(
        owner=test_user,
        type=OrganizationType.PERSONAL,
        defaults={
            "name": f"{test_user.first_name}'s Workspace",
            "slug": f"personal-{test_user.user_id}",
            "settings": {},
        }
    )

    # Create active subscription on organization
    subscription = await BillingSubscription.create(
        organization=personal_org,
        stripe_subscription_id="sub_test_123",
        tier=SubscriptionTier.PRO,
        status=SubscriptionStatus.ACTIVE,
        current_period_start=datetime.now(timezone.utc),
        current_period_end=datetime.now(timezone.utc) + timedelta(days=30),
    )

    return test_user, subscription


# =============================================================================
# FastAPI Client Fixtures
# =============================================================================


@pytest.fixture
def mock_app() -> FastAPI:
    """
    Create a minimal FastAPI app for testing without full initialization.

    This is faster than using the full app with lifespan handlers.
    Use for unit tests that just need route testing.
    """
    from fastapi import FastAPI

    app = FastAPI(title="Test App")
    return app


@pytest.fixture
async def api_client(mock_app: FastAPI) -> AsyncGenerator[AsyncClient, None]:
    """
    Async HTTP client for API endpoint testing.

    Uses ASGI transport as recommended by FastAPI for testing.
    Automatically handles async context management.

    Usage:
        async def test_endpoint(api_client):
            response = await api_client.get("/v1/workflows")
            assert response.status_code == 200
    """
    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
async def authenticated_client(mock_app: FastAPI, test_user) -> AsyncGenerator[AsyncClient, None]:
    """
    Authenticated API client with test user in request state.

    Bypasses authentication middleware for testing protected endpoints.
    """
    from fastapi import Request

    # Mock authentication middleware to inject test_user
    async def mock_auth_middleware(request: Request, call_next):
        request.state.user = test_user
        response = await call_next(request)
        return response

    # Add mock middleware
    mock_app.middleware("http")(mock_auth_middleware)

    transport = ASGITransport(app=mock_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


# =============================================================================
# Workflow Fixtures
# =============================================================================


@pytest.fixture
def sample_workflow_spec() -> dict:
    """
    Minimal valid workflow spec for testing.

    Returns a version 2 workflow with:
    - One test trigger
    - One tool node
    - One edge connecting them
    """
    return {
        "version": "2",
        "triggers": [
            {
                "id": "t1",
                "key": "test.trigger",
                "mode": "polling",
                "event_schema": {},
            }
        ],
        "nodes": [
            {
                "id": "n1",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": "test"},
            }
        ],
        "edges": [
            {
                "source": "t1",
                "target": "n1",
                "type": "trigger",
            }
        ],
    }


@pytest.fixture
def complex_workflow_spec() -> dict:
    """
    Complex workflow spec with multiple node types for advanced testing.

    Includes:
    - Conditional branching
    - Multiple tools
    - Control flow nodes
    """
    return {
        "version": "2",
        "triggers": [
            {
                "id": "trigger_1",
                "key": "test.complex_trigger",
                "mode": "polling",
                "event_schema": {},
            }
        ],
        "nodes": [
            {
                "id": "task_1",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"value": {"success": True}},
            },
            {
                "id": "condition_1",
                "type": "if",
                "condition": "${task_1.result.success}",
            },
            {
                "id": "task_2",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"status": "success"},
            },
            {
                "id": "task_3",
                "type": "tool",
                "tool": "test.tool",
                "inputs": {"status": "failure"},
            },
        ],
        "edges": [
            {

                "source": "trigger_1",
                "target": "task_1",
                "type": "trigger",
            },
            {

                "source": "task_1",
                "target": "condition_1",
            },
            {

                "source": "condition_1",
                "target": "task_2",
                "type": "conditional_true",
            },
            {

                "source": "condition_1",
                "target": "task_3",
                "type": "conditional_false",
            },
        ],
    }


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_tool():
    """
    Mock tool for testing tool execution without external dependencies.
    """
    tool = AsyncMock()
    tool.execute.return_value = {"status": "success", "data": "mock_result"}
    tool.get_parameters_schema.return_value = {
        "type": "object",
        "properties": {
            "param1": {"type": "string"},
        },
    }
    return tool


@pytest.fixture
def mock_llm():
    """
    Mock LLM client for testing without API calls.
    """
    llm = AsyncMock()
    llm.agenerate.return_value = "Mock LLM response"
    return llm


@pytest.fixture
def mock_posthog():
    """
    Mock PostHog analytics client to prevent analytics calls during tests.
    """
    with patch("seer.analytics.analytics.capture") as mock_capture:
        yield mock_capture


# =============================================================================
# Environment Configuration
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def test_environment():
    """
    Test environment verification fixture.

    NOTE: Environment variables are set at the TOP of this conftest.py file
    (before any seer imports) to ensure the config singleton is properly
    initialized with test values. This fixture exists primarily as a marker
    to document this and could be used for additional session-level setup.
    """
    # Verify critical test environment settings are in place
    import os
    assert os.environ.get("POSTHOG_ENABLED") == "false", "POSTHOG_ENABLED not set correctly"
    assert os.environ.get("SENTRY_ENABLED") == "false", "SENTRY_ENABLED not set correctly"

    yield


def pytest_sessionfinish(session, exitstatus):
    """Cleanup resources at end of test session to ensure process exits cleanly."""
    global _test_exit_code
    _test_exit_code = exitstatus

    import threading
    import asyncio

    # Shutdown posthog if it was initialized
    try:
        import posthog
        posthog.shutdown()
    except Exception:
        pass

    # Reset sentry state
    try:
        import seer.observability.sentry_client as sentry_client
        sentry_client.SENTRY_INITIALIZED = False
    except Exception:
        pass

    # Reset posthog state
    try:
        import seer.observability.posthog_client as posthog_client
        posthog_client.POSTHOG_INITIALIZED = False
    except Exception:
        pass

    # Reset event loop reference
    try:
        import seer.core.event_loop as event_loop_module
        event_loop_module._MAIN_EVENT_LOOP = None
    except Exception:
        pass

    # Close the PostgreSQL checkpointer connection pool - this is crucial!
    # The AsyncConnectionPool creates a non-daemon worker thread that blocks process exit.
    try:
        from seer.api.agents.checkpointer import _checkpointer_cm, close_checkpointer
        import seer.api.agents.checkpointer as checkpointer_module

        if checkpointer_module._checkpointer_cm is not None:
            # Need to run async cleanup in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup as a task if loop is running
                    asyncio.ensure_future(close_checkpointer())
                else:
                    loop.run_until_complete(close_checkpointer())
            except RuntimeError:
                # No event loop - create a new one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(close_checkpointer())
                finally:
                    loop.close()
    except Exception:
        pass


_test_exit_code = 0  # Global to track actual exit code from pytest_sessionfinish


def pytest_unconfigure(config):
    """
    Final cleanup hook - runs after pytest_sessionfinish and all reporting.

    Forces process exit if there are non-daemon threads that would hang forever.
    This handles cases where connection pools create non-daemon worker threads
    that don't get properly cleaned up.
    """
    import threading
    import os
    import time

    # Small delay to allow normal cleanup
    time.sleep(0.1)

    threads = threading.enumerate()
    blocking_threads = [t for t in threads if t.name != "MainThread" and t.is_alive() and not t.daemon]

    if blocking_threads:
        # Force exit - blocking threads would hang forever
        # Use the actual test exit code to preserve pass/fail status
        os._exit(_test_exit_code)
