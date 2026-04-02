"""
Unit tests for OAuth 401 retry-with-refresh logic.

Tests the platform-level retry mechanism that force-refreshes tokens
when a tool execution receives a 401 from the provider API, even if
the stored expires_at has not yet passed.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


def _make_connection(provider: str = "oura", expired: bool = False) -> MagicMock:
    conn = MagicMock()
    conn.id = 30
    conn.provider = provider
    conn.refresh_token_enc = "rt_test"
    conn.access_token_enc = "old_token"
    conn.scopes = "heartrate"
    conn.status = "active"
    if expired:
        conn.expires_at = datetime.now(timezone.utc) - timedelta(hours=1)
    else:
        conn.expires_at = datetime.now(timezone.utc) + timedelta(days=25)
    conn.save = AsyncMock()
    return conn


# =============================================================================
# get_oauth_token force_refresh tests
# =============================================================================


@pytest.mark.unit
class TestGetOAuthTokenForceRefresh:
    """Tests that force_refresh bypasses the expires_at check."""

    @pytest.mark.asyncio
    async def test_force_refresh_triggers_refresh_even_when_not_expired(self):
        """Token has 25 days left, but force_refresh=True should still refresh."""
        from seer.tools.oauth_manager import get_oauth_token

        mock_user = MagicMock()
        conn = _make_connection(provider="oura", expired=False)

        refreshed_conn = MagicMock()
        refreshed_conn.access_token_enc = "new_token"
        refreshed_conn.refresh_token_enc = "rt_test"

        with (
            patch("seer.tools.oauth_manager.OAuthConnection") as MockOAuth,
            patch("seer.tools.oauth_manager.refresh_oauth_token", new_callable=AsyncMock, return_value=refreshed_conn) as mock_refresh,
        ):
            MockOAuth.filter.return_value.first = AsyncMock(return_value=conn)
            result_conn, token = await get_oauth_token(mock_user, connection_id="30", force_refresh=True)

        mock_refresh.assert_awaited_once_with(conn)
        assert token == "new_token"

    @pytest.mark.asyncio
    async def test_no_force_refresh_skips_refresh_when_not_expired(self):
        """Without force_refresh, a non-expired token should not be refreshed."""
        from seer.tools.oauth_manager import get_oauth_token

        mock_user = MagicMock()
        conn = _make_connection(provider="oura", expired=False)

        with (
            patch("seer.tools.oauth_manager.OAuthConnection") as MockOAuth,
            patch("seer.tools.oauth_manager.refresh_oauth_token", new_callable=AsyncMock) as mock_refresh,
        ):
            MockOAuth.filter.return_value.first = AsyncMock(return_value=conn)
            result_conn, token = await get_oauth_token(mock_user, connection_id="30")

        mock_refresh.assert_not_awaited()
        assert token == "old_token"

    @pytest.mark.asyncio
    async def test_force_refresh_no_refresh_token_raises_401(self):
        """force_refresh with no refresh_token should raise 401, not silently pass."""
        from seer.tools.oauth_manager import get_oauth_token

        mock_user = MagicMock()
        conn = _make_connection(provider="oura", expired=False)
        conn.refresh_token_enc = None  # No refresh token

        with patch("seer.tools.oauth_manager.OAuthConnection") as MockOAuth:
            MockOAuth.filter.return_value.first = AsyncMock(return_value=conn)

            with pytest.raises(HTTPException) as exc_info:
                await get_oauth_token(mock_user, connection_id="30", force_refresh=True)

        assert exc_info.value.status_code == 401
        assert "no refresh token" in exc_info.value.detail.lower()


# =============================================================================
# CredentialResolver force_refresh pass-through tests
# =============================================================================


@pytest.mark.unit
class TestCredentialResolverForceRefresh:
    """Tests that resolve(force_refresh=True) passes through to get_oauth_token."""

    @pytest.mark.asyncio
    async def test_resolve_passes_force_refresh_to_get_oauth_token(self, mock_user):
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool = MagicMock()
        mock_tool.name = "oura_get_heart_rate"
        mock_tool.required_scopes = ["heartrate"]
        mock_tool.requires_oauth = True
        mock_tool.required_secrets = []
        mock_tool.optional_secrets = []
        mock_tool.default_resource = None
        mock_tool.provider = "oura"
        mock_tool.integration_type = "oura"

        conn = _make_connection(provider="oura")

        with (
            patch("seer.tools.credential_resolver.get_oauth_token", new_callable=AsyncMock, return_value=(conn, "refreshed_token")) as mock_get_token,
            patch("seer.tools.credential_resolver.validate_scopes", return_value=(True, None)),
        ):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool, connection_id="30")
            result = await resolver.resolve({}, force_refresh=True)

        mock_get_token.assert_awaited_once()
        call_kwargs = mock_get_token.call_args.kwargs
        assert call_kwargs["force_refresh"] is True
        assert result.access_token == "refreshed_token"

    @pytest.mark.asyncio
    async def test_resolve_defaults_force_refresh_false(self, mock_user):
        from seer.tools.credential_resolver import CredentialResolver

        mock_tool = MagicMock()
        mock_tool.name = "oura_get_heart_rate"
        mock_tool.required_scopes = ["heartrate"]
        mock_tool.requires_oauth = True
        mock_tool.required_secrets = []
        mock_tool.optional_secrets = []
        mock_tool.default_resource = None
        mock_tool.provider = "oura"
        mock_tool.integration_type = "oura"

        conn = _make_connection(provider="oura")

        with (
            patch("seer.tools.credential_resolver.get_oauth_token", new_callable=AsyncMock, return_value=(conn, "old_token")) as mock_get_token,
            patch("seer.tools.credential_resolver.validate_scopes", return_value=(True, None)),
        ):
            resolver = CredentialResolver(user=mock_user, tool=mock_tool, connection_id="30")
            await resolver.resolve({})

        call_kwargs = mock_get_token.call_args.kwargs
        assert call_kwargs["force_refresh"] is False


# =============================================================================
# executor.py retry-on-401 tests
# =============================================================================


@pytest.mark.unit
class TestExecutorRetryOn401:
    """Tests that execute_tool retries once with force_refresh on 401."""

    @pytest.mark.asyncio
    async def test_retry_succeeds_after_401(self):
        """Tool returns 401, executor refreshes and retries, second call succeeds."""
        from seer.tools.executor import execute_tool

        mock_user = MagicMock()
        mock_user.user_id = "user_44"

        conn = _make_connection(provider="oura")
        resolved_stale = MagicMock()
        resolved_stale.connection = conn
        resolved_stale.access_token = "stale_token"

        resolved_fresh = MagicMock()
        resolved_fresh.connection = conn
        resolved_fresh.access_token = "fresh_token"

        mock_tool = MagicMock()
        mock_tool.name = "oura_get_heart_rate"
        mock_tool.get_parameters_schema.return_value = {}
        mock_tool.execute = AsyncMock(
            side_effect=[
                HTTPException(status_code=401, detail="Oura OAuth token expired"),
                {"heart_rate": [72, 75, 68]},
            ]
        )

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve = AsyncMock(
            side_effect=[resolved_stale, resolved_fresh]
        )

        with (
            patch("seer.tools.executor.get_tool", return_value=mock_tool),
            patch("seer.tools.executor.coerce_arguments", side_effect=lambda a, s: a),
            patch("seer.tools.executor.CredentialResolver", return_value=mock_resolver_instance),
        ):
            result = await execute_tool("oura_get_heart_rate", mock_user, connection_id="30", arguments={})

        assert result == {"heart_rate": [72, 75, 68]}
        assert mock_tool.execute.await_count == 2
        # Second resolve call should have force_refresh=True
        second_resolve_call = mock_resolver_instance.resolve.call_args_list[1]
        assert second_resolve_call.kwargs["force_refresh"] is True

    @pytest.mark.asyncio
    async def test_401_without_refresh_token_raises_immediately(self):
        """Tool returns 401 but connection has no refresh token — no retry, raise."""
        from seer.tools.executor import execute_tool

        mock_user = MagicMock()
        mock_user.user_id = "user_44"

        conn = _make_connection(provider="oura")
        conn.refresh_token_enc = None  # No refresh token

        resolved = MagicMock()
        resolved.connection = conn
        resolved.access_token = "stale_token"

        mock_tool = MagicMock()
        mock_tool.name = "oura_get_heart_rate"
        mock_tool.get_parameters_schema.return_value = {}
        mock_tool.execute = AsyncMock(
            side_effect=HTTPException(status_code=401, detail="Token expired")
        )

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve = AsyncMock(return_value=resolved)

        with (
            patch("seer.tools.executor.get_tool", return_value=mock_tool),
            patch("seer.tools.executor.coerce_arguments", side_effect=lambda a, s: a),
            patch("seer.tools.executor.CredentialResolver", return_value=mock_resolver_instance),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_tool("oura_get_heart_rate", mock_user, connection_id="30", arguments={})

        assert exc_info.value.status_code == 401
        assert mock_tool.execute.await_count == 1  # No retry

    @pytest.mark.asyncio
    async def test_non_401_error_not_retried(self):
        """Non-401 HTTP errors should not trigger retry."""
        from seer.tools.executor import execute_tool

        mock_user = MagicMock()
        mock_user.user_id = "user_44"

        conn = _make_connection(provider="oura")
        resolved = MagicMock()
        resolved.connection = conn
        resolved.access_token = "token"

        mock_tool = MagicMock()
        mock_tool.name = "oura_get_heart_rate"
        mock_tool.get_parameters_schema.return_value = {}
        mock_tool.execute = AsyncMock(
            side_effect=HTTPException(status_code=429, detail="Rate limited")
        )

        mock_resolver_instance = MagicMock()
        mock_resolver_instance.resolve = AsyncMock(return_value=resolved)

        with (
            patch("seer.tools.executor.get_tool", return_value=mock_tool),
            patch("seer.tools.executor.coerce_arguments", side_effect=lambda a, s: a),
            patch("seer.tools.executor.CredentialResolver", return_value=mock_resolver_instance),
        ):
            with pytest.raises(HTTPException) as exc_info:
                await execute_tool("oura_get_heart_rate", mock_user, connection_id="30", arguments={})

        assert exc_info.value.status_code == 429
        assert mock_tool.execute.await_count == 1
