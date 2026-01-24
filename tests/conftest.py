# pylint: disable=import-outside-toplevel,redefined-outer-name,reimported
# Reason: Test fixtures commonly use lazy imports and pytest fixture pattern requires name reuse
"""
Global test fixtures for Seer test suite.

Provides core fixtures for:
- Async testing with anyio backend (FastAPI recommendation)
- Database setup with SQLite in-memory for fast tests
- Transaction rollback for test isolation
- Test user creation
- FastAPI test client
- Sample workflow specifications
"""
from typing import AsyncGenerator
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from tortoise import Tortoise

from seer.database.config import TORTOISE_ORM


# =============================================================================
# Async Testing Configuration
# =============================================================================


@pytest.fixture(scope="session")
def anyio_backend():
    """Use asyncio backend for anyio tests (FastAPI recommendation)."""
    return "asyncio"


# =============================================================================
# Database Fixtures
# =============================================================================


@pytest.fixture(scope="function")
async def db_engine():
    """
    Initialize test database (SQLite in-memory) for each test.

    Uses SQLite for fast tests (~100x faster than PostgreSQL).
    All models are created in memory for perfect isolation.
    """
    # Create test configuration with SQLite in-memory
    test_config = {
        "connections": {
            "default": {
                "engine": "tortoise.backends.sqlite",
                "credentials": {"file_path": ":memory:"},
            }
        },
        "apps": TORTOISE_ORM["apps"],
        "use_tz": True,
        "timezone": "UTC",
    }

    # Initialize Tortoise ORM
    await Tortoise.init(config=test_config)
    await Tortoise.generate_schemas()

    yield

    # Cleanup
    await Tortoise.close_connections()


# Note: db_engine fixture is NOT autouse - tests that need database
# should explicitly request it via parameters


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
    Create test user with active PRO subscription.

    Useful for testing subscription-dependent features.
    """
    from datetime import datetime, timedelta, timezone
    from seer.database.subscription_models import (
        BillingProfile,
        BillingProfileType,
        BillingSubscription,
        SubscriptionStatus,
        SubscriptionTier,
    )

    # Create billing profile
    profile = await BillingProfile.create(
        user=test_user,
        profile_type=BillingProfileType.STRIPE,
        stripe_customer_id="cus_test_123",
    )

    # Create active subscription
    subscription = await BillingSubscription.create(
        billing_profile=profile,
        subscription_id="sub_test_123",
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
    - One task node
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
                "type": "task",
                "kind": "set",
                "value": {"result": "test"},
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
    - Multiple tasks
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
                "type": "task",
                "kind": "set",
                "value": {"result": {"success": True}},
            },
            {
                "id": "condition_1",
                "type": "if",
                "condition": "${task_1.result.success}",
            },
            {
                "id": "task_2",
                "type": "task",
                "kind": "set",
                "value": {"status": "success"},
            },
            {
                "id": "task_3",
                "type": "task",
                "kind": "set",
                "value": {"status": "failure"},
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
    Set up test environment variables.

    Runs once per session and applies to all tests.
    """
    import os

    # Disable external services during tests
    os.environ["POSTHOG_ENABLED"] = "false"
    os.environ["TRIGGER_POLLER_ENABLED"] = "false"
    os.environ["TOOL_INDEX_AUTO_GENERATE"] = "false"

    # Use test mode for cloud features
    os.environ["CLOUD_MODE"] = "false"

    # Set minimal database pool for tests
    os.environ["DB_MAX_CONNECTIONS"] = "5"
    os.environ["DB_MIN_CONNECTIONS"] = "1"

    yield

    # Cleanup not needed for environment variables
