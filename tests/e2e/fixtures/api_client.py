# pylint: disable=import-outside-toplevel,redefined-outer-name
# Reason: Test fixtures use lazy imports and pytest fixture pattern requires name reuse
"""
API client fixtures for E2E tests.

Provides:
- FastAPI app configured for testing
- Unauthenticated HTTP client
- Authenticated HTTP client with JWT tokens
- Test user creation
"""
from typing import AsyncGenerator
from datetime import datetime, timezone

import pytest


@pytest.fixture(scope="function")
async def e2e_app(database_url: str, redis_url: str):
    """
    Create FastAPI application configured for E2E testing.

    Sets environment variables to point to test containers before
    importing the app, ensuring the config singleton uses test values.

    Args:
        database_url: PostgreSQL URL from container
        redis_url: Redis URL from container

    Returns:
        FastAPI: Application instance configured for testing
    """
    import os
    import sys

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

    # Disable AWS parameter store loading
    os.environ["AWS_PARAMETER_PATH"] = ""
    os.environ["SSM_ENABLED"] = "false"

    # Clear ALL seer modules from cache to ensure fresh config
    # This is necessary because the config singleton is created at module import time
    modules_to_clear = [key for key in sys.modules.keys() if key.startswith("seer")]
    for module in modules_to_clear:
        del sys.modules[module]

    # Now import the app with correct environment
    from seer.api.main import app

    # Disable lifespan to avoid database initialization conflicts
    # We manage DB lifecycle in db_session fixture
    app.router.lifespan_context = None

    return app


@pytest.fixture(scope="function")
async def e2e_client(e2e_app) -> AsyncGenerator:
    """
    Unauthenticated async HTTP client for E2E testing.

    Uses ASGI transport to call the FastAPI app directly without
    network overhead.

    Args:
        e2e_app: FastAPI application instance

    Yields:
        AsyncClient: HTTP client for making requests
    """
    from httpx import AsyncClient, ASGITransport

    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture(scope="function")
async def e2e_test_user(db_session) -> "User":
    """
    Create a test user in the real database.

    This user can be used for authenticated requests and as a workflow owner.

    Args:
        db_session: Database session fixture (ensures DB is initialized)

    Returns:
        User: Created test user instance
    """
    from seer.database.models import User

    user = await User.create(
        user_id="e2e_test_user_001",
        email="e2e_test@example.com",
        first_name="E2E",
        last_name="Tester",
        created_at=datetime.now(timezone.utc),
    )
    return user


@pytest.fixture(scope="function")
async def authenticated_e2e_client(e2e_app, e2e_test_user) -> AsyncGenerator:
    """
    Authenticated async HTTP client for E2E testing.

    Creates a JWT token containing the test user's information that will
    be decoded by the TokenDecodeWithoutValidationMiddleware in self-hosted mode.

    Self-hosted mode doesn't validate JWT signatures, so we can use any
    secret to sign the token.

    Args:
        e2e_app: FastAPI application instance
        e2e_test_user: Test user for authentication

    Yields:
        AsyncClient: Authenticated HTTP client
    """
    import jwt
    from httpx import AsyncClient, ASGITransport

    # Create JWT token with user info
    token_payload = {
        "sub": e2e_test_user.user_id,
        "email": e2e_test_user.email,
        "first_name": e2e_test_user.first_name,
        "last_name": e2e_test_user.last_name,
    }

    # Sign with any secret (self-hosted mode doesn't validate)
    token = jwt.encode(token_payload, "e2e_test_secret", algorithm="HS256")

    transport = ASGITransport(app=e2e_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Add Authorization header
        client.headers["Authorization"] = f"Bearer {token}"
        yield client


@pytest.fixture(scope="function")
async def e2e_test_user_with_org(e2e_test_user):
    """
    Create a test user with an associated organization.

    Many features require organization context, so this fixture
    sets up both user and personal organization.

    Args:
        e2e_test_user: Test user instance

    Returns:
        Tuple[User, Organization]: User and their personal organization
    """
    from seer.database.organization_models import Organization, OrganizationType

    org, _ = await Organization.get_or_create(
        owner=e2e_test_user,
        type=OrganizationType.PERSONAL,
        defaults={
            "name": f"{e2e_test_user.first_name}'s Workspace",
            "slug": f"personal-{e2e_test_user.user_id}",
            "settings": {},
        }
    )

    return e2e_test_user, org


__all__ = [
    "e2e_app",
    "e2e_client",
    "e2e_test_user",
    "authenticated_e2e_client",
    "e2e_test_user_with_org",
]
