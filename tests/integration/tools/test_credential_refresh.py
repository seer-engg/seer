"""
Integration tests for OAuth credential refresh during tool execution.

Tests:
- Token refresh when expired
- Refresh with valid refresh_token
- Handling of missing refresh tokens
- Provider-specific refresh flows (Google, GitHub, Supabase)
- Connection status updates after refresh

These tests verify that OAuth-based tools work correctly when tokens expire.
"""
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from seer.database.models_oauth import OAuthConnection


def utcnow():
    """Get current UTC time."""
    return datetime.now(timezone.utc)


# =============================================================================
# OAuth Connection State Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oauth_connection_creation(db_engine, test_user):
    """
    Test creating an OAuth connection with tokens.

    Verifies:
    - Connection is created with access token
    - Refresh token is stored
    - Expiry is tracked
    """
    expires_at = utcnow() + timedelta(hours=1)

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="test@gmail.com",
        access_token_enc="encrypted_access_token",
        refresh_token_enc="encrypted_refresh_token",
        expires_at=expires_at,
        scopes="email profile",
        status="active",
    )

    assert connection.provider == "google"
    assert connection.access_token_enc == "encrypted_access_token"
    assert connection.refresh_token_enc == "encrypted_refresh_token"
    assert connection.expires_at == expires_at
    assert connection.status == "active"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oauth_connection_expired_detection(db_engine, test_user):
    """
    Test detecting expired OAuth tokens.

    Verifies:
    - Expired tokens are correctly identified
    - Non-expired tokens are not flagged
    """
    # Expired connection
    expired_conn = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="expired@gmail.com",
        access_token_enc="expired_token",
        refresh_token_enc="refresh_token",
        expires_at=utcnow() - timedelta(hours=1),  # Expired
        status="active",
    )

    # Valid connection
    valid_conn = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="valid@gmail.com",
        access_token_enc="valid_token",
        refresh_token_enc="refresh_token",
        expires_at=utcnow() + timedelta(hours=1),  # Not expired
        status="active",
    )

    # Check expiration
    is_expired = expired_conn.expires_at < utcnow()
    is_valid = valid_conn.expires_at > utcnow()

    assert is_expired
    assert is_valid


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oauth_connection_without_expiry(db_engine, test_user):
    """
    Test OAuth connection without expiry (long-lived tokens).

    Some OAuth providers (like API keys) don't have expiry.
    """
    connection = await OAuthConnection.create(
        user=test_user,
        provider="github",
        provider_account_id="github_user",
        access_token_enc="github_token",
        refresh_token_enc=None,  # GitHub PATs don't use refresh
        expires_at=None,  # No expiry
        status="active",
    )

    assert connection.expires_at is None
    assert connection.refresh_token_enc is None


# =============================================================================
# Token Refresh Flow Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_refresh_oauth_token_updates_connection(db_engine, test_user):
    """
    Test that refresh_oauth_token updates the connection with new token.

    Mocks the HTTP call to token endpoint and verifies database update.
    """
    from seer.tools.oauth_manager import refresh_oauth_token

    # Create expired connection
    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="refresh_test@gmail.com",
        access_token_enc="old_access_token",
        refresh_token_enc="valid_refresh_token",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    # Mock successful token refresh
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_access_token",
        "expires_in": 3600,
        "scope": "email profile",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.google_client_id = "test_client_id"
        mock_config.google_client_secret = "test_client_secret"

        # Setup async context manager
        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        # Perform refresh
        refreshed = await refresh_oauth_token(connection)

        # Verify token was updated
        assert refreshed.access_token_enc == "new_access_token"
        assert refreshed.expires_at > utcnow()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_refresh_without_refresh_token_fails(db_engine, test_user):
    """
    Test that refresh fails when no refresh_token is available.
    """
    from fastapi import HTTPException
    from seer.tools.oauth_manager import refresh_oauth_token

    # Connection without refresh token
    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="no_refresh@gmail.com",
        access_token_enc="old_token",
        refresh_token_enc=None,  # No refresh token
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    with pytest.raises(HTTPException) as exc_info:
        await refresh_oauth_token(connection)

    assert exc_info.value.status_code == 401
    assert "No refresh token" in exc_info.value.detail


@pytest.mark.integration
@pytest.mark.asyncio
async def test_refresh_unsupported_provider_fails(db_engine, test_user):
    """
    Test that refresh fails for unsupported providers.
    """
    from fastapi import HTTPException
    from seer.tools.oauth_manager import refresh_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="unsupported_provider",
        provider_account_id="user@unsupported.com",
        access_token_enc="token",
        refresh_token_enc="refresh_token",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    with pytest.raises(HTTPException) as exc_info:
        await refresh_oauth_token(connection)

    assert exc_info.value.status_code == 400
    assert "not supported" in exc_info.value.detail


# =============================================================================
# get_oauth_token Integration Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_returns_valid_token(db_engine, test_user):
    """
    Test get_oauth_token returns valid token when not expired.
    """
    from seer.tools.oauth_manager import get_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="valid@gmail.com",
        access_token_enc="valid_access_token",
        refresh_token_enc="refresh_token",
        expires_at=utcnow() + timedelta(hours=1),  # Not expired
        status="active",
    )

    conn, token = await get_oauth_token(test_user, connection_id=str(connection.id))

    assert conn.id == connection.id
    assert token == "valid_access_token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_by_provider(db_engine, test_user):
    """
    Test get_oauth_token lookup by provider name.
    """
    from seer.tools.oauth_manager import get_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="github",
        provider_account_id="github_user",
        access_token_enc="github_token",
        expires_at=None,  # No expiry
        status="active",
    )

    conn, token = await get_oauth_token(test_user, provider="github")

    assert conn.id == connection.id
    assert token == "github_token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_auto_refreshes_expired(db_engine, test_user):
    """
    Test that get_oauth_token automatically refreshes expired tokens.
    """
    from seer.tools.oauth_manager import get_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="auto_refresh@gmail.com",
        access_token_enc="expired_token",
        refresh_token_enc="valid_refresh",
        expires_at=utcnow() - timedelta(hours=1),  # Expired
        status="active",
    )

    # Mock refresh
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "refreshed_token",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.google_client_id = "client_id"
        mock_config.google_client_secret = "client_secret"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        conn, token = await get_oauth_token(test_user, connection_id=str(connection.id))

        assert token == "refreshed_token"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_fails_for_inactive_connection(db_engine, test_user):
    """
    Test that get_oauth_token fails for inactive connections.
    """
    from fastapi import HTTPException
    from seer.tools.oauth_manager import get_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="inactive@gmail.com",
        access_token_enc="token",
        status="revoked",  # Not active
    )

    with pytest.raises(HTTPException) as exc_info:
        await get_oauth_token(test_user, connection_id=str(connection.id))

    assert exc_info.value.status_code == 404


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_no_connection_found(db_engine, test_user):
    """
    Test get_oauth_token raises when no connection found.
    """
    from fastapi import HTTPException
    from seer.tools.oauth_manager import get_oauth_token

    with pytest.raises(HTTPException) as exc_info:
        await get_oauth_token(test_user, provider="nonexistent")

    assert exc_info.value.status_code == 404
    assert "No active OAuth connection" in exc_info.value.detail


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_oauth_token_requires_params(db_engine, test_user):
    """
    Test that either connection_id or provider must be provided.
    """
    from fastapi import HTTPException
    from seer.tools.oauth_manager import get_oauth_token

    with pytest.raises(HTTPException) as exc_info:
        await get_oauth_token(test_user)  # No connection_id or provider

    assert exc_info.value.status_code == 400
    assert "connection_id or provider" in exc_info.value.detail


# =============================================================================
# Provider-Specific Refresh Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_google_oauth_refresh_endpoint(db_engine, test_user):
    """
    Test Google OAuth uses correct refresh endpoint.
    """
    from seer.tools.oauth_manager import refresh_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="google_user",
        access_token_enc="old_token",
        refresh_token_enc="google_refresh",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_token",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.google_client_id = "google_client"
        mock_config.google_client_secret = "google_secret"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await refresh_oauth_token(connection)

        # Verify correct endpoint was called
        call_args = mock_instance.post.call_args
        assert "oauth2.googleapis.com" in call_args[0][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_github_oauth_refresh_endpoint(db_engine, test_user):
    """
    Test GitHub OAuth uses correct refresh endpoint.
    """
    from seer.tools.oauth_manager import refresh_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="github",
        provider_account_id="github_user",
        access_token_enc="old_token",
        refresh_token_enc="github_refresh",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_token",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.github_client_id = "github_client"
        mock_config.github_client_secret = "github_secret"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await refresh_oauth_token(connection)

        # Verify correct endpoint was called
        call_args = mock_instance.post.call_args
        assert "github.com/login/oauth" in call_args[0][0]


@pytest.mark.integration
@pytest.mark.asyncio
async def test_supabase_oauth_refresh_endpoint(db_engine, test_user):
    """
    Test Supabase OAuth uses correct refresh endpoint.
    """
    from seer.tools.oauth_manager import refresh_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="supabase",
        provider_account_id="supabase_org",
        access_token_enc="old_token",
        refresh_token_enc="supabase_refresh",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "new_token",
        "expires_in": 3600,
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.supabase_client_id = "supabase_client"
        mock_config.supabase_client_secret = "supabase_secret"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await refresh_oauth_token(connection)

        # Verify correct endpoint was called
        call_args = mock_instance.post.call_args
        assert "api.supabase.com" in call_args[0][0]


# =============================================================================
# Connection Update Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_refresh_updates_connection_in_database(db_engine, test_user):
    """
    Test that refresh persists updated token to database.
    """
    from seer.tools.oauth_manager import refresh_oauth_token

    connection = await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="persist_test@gmail.com",
        access_token_enc="old_token",
        refresh_token_enc="refresh_token",
        expires_at=utcnow() - timedelta(hours=1),
        status="active",
    )
    connection_id = connection.id

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "access_token": "persisted_token",
        "expires_in": 7200,
        "scope": "new scopes",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("seer.tools.oauth_manager.httpx.AsyncClient") as mock_client, \
         patch("seer.tools.oauth_manager.config") as mock_config:

        mock_config.google_client_id = "client"
        mock_config.google_client_secret = "secret"

        mock_instance = AsyncMock()
        mock_instance.post = AsyncMock(return_value=mock_response)
        mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_instance)
        mock_client.return_value.__aexit__ = AsyncMock()

        await refresh_oauth_token(connection)

    # Fetch fresh from database
    fetched = await OAuthConnection.get(id=connection_id)

    assert fetched.access_token_enc == "persisted_token"
    assert fetched.scopes == "new scopes"
    assert fetched.expires_at > utcnow()


# =============================================================================
# User Isolation Tests
# =============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_oauth_connections_isolated_by_user(db_engine, test_user):
    """
    Test that OAuth connections are isolated by user.
    """
    from datetime import timezone
    from seer.database.models import User
    from seer.tools.oauth_manager import get_oauth_token
    from fastapi import HTTPException

    # Create second user
    user2 = await User.create(
        user_id="other_user_789",
        email="other@example.com",
        first_name="Other",
        last_name="User",
        created_at=utcnow(),
    )

    # Create connection for test_user
    await OAuthConnection.create(
        user=test_user,
        provider="google",
        provider_account_id="user1@gmail.com",
        access_token_enc="user1_token",
        expires_at=utcnow() + timedelta(hours=1),
        status="active",
    )

    # user2 should not be able to access test_user's connection
    with pytest.raises(HTTPException):
        await get_oauth_token(user2, provider="google")
