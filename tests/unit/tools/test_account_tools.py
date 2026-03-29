"""Tests for OAuth account discovery tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_user():
    """Create a mock user object."""
    user = MagicMock()
    user.id = 1
    user.user_id = "user_123"
    user.email = "test@example.com"
    return user


@pytest.fixture
def mock_oauth_connection():
    """Create a mock OAuth connection factory."""
    def _create(
        conn_id: int = 1,
        provider: str = "google",
        provider_account_id: str = "alice@gmail.com",
        scopes: str = "https://www.googleapis.com/auth/gmail.send",
        status: str = "active",
        display_name: str = None,
    ):
        conn = MagicMock()
        conn.id = conn_id
        conn.provider = provider
        conn.provider_account_id = provider_account_id
        conn.scopes = scopes
        conn.status = status
        # Mock the provider_metadata for display name
        conn.provider_metadata = {"email": display_name or provider_account_id}
        return conn
    return _create


@pytest.fixture
def mock_tool():
    """Create a mock tool with OAuth requirements."""
    def _create(
        name: str = "gmail_send_email",
        provider: str = "google",
        required_scopes: list = None,
    ):
        tool = MagicMock()
        tool.name = name
        tool.provider = provider
        tool.required_scopes = required_scopes or ["https://www.googleapis.com/auth/gmail.send"]
        return tool
    return _create


@pytest.fixture
def mock_trigger_definition():
    """Create a mock trigger definition."""
    def _create(
        key: str = "poll.gmail.email_received",
        provider: str = "gmail",
        requires_connection: bool = True,
        required_scopes: list = None,
    ):
        definition = MagicMock()
        definition.key = key
        definition.provider = provider
        definition.meta = MagicMock()
        definition.meta.requires_connection = requires_connection
        definition.meta.required_scopes = required_scopes or ["https://www.googleapis.com/auth/gmail.readonly"]
        return definition
    return _create


class TestGetToolAccounts:
    """Tests for get_tool_accounts_impl."""

    @pytest.mark.asyncio
    async def test_single_account_no_selection_required(
        self, mock_user, mock_oauth_connection, mock_tool
    ):
        """Test get_tool_accounts returns single account without selection required."""
        from seer.tools.unified_tools import get_tool_accounts_impl

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.tools.base.get_tool") as mock_get_tool, \
             patch("seer.services.integrations.auth.helpers.list_connections", new_callable=AsyncMock) as mock_list, \
             patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has_scopes:

            mock_get_user.return_value = mock_user
            mock_get_tool.return_value = mock_tool()
            mock_list.return_value = [mock_oauth_connection()]
            mock_display.return_value = "alice@gmail.com"
            mock_has_scopes.return_value = True

            result = await get_tool_accounts_impl("gmail_send_email", reasoning="checking accounts")

            data = json.loads(result)
            assert data["tool_name"] == "gmail_send_email"
            assert data["provider"] == "google"
            assert len(data["accounts"]) == 1
            assert data["requires_selection"] is False
            assert data["accounts"][0]["display_name"] == "alice@gmail.com"

    @pytest.mark.asyncio
    async def test_multiple_accounts_requires_selection(
        self, mock_user, mock_oauth_connection, mock_tool
    ):
        """Test get_tool_accounts returns multiple accounts with selection required."""
        from seer.tools.unified_tools import get_tool_accounts_impl

        conn1 = mock_oauth_connection(conn_id=1, provider_account_id="alice@gmail.com")
        conn2 = mock_oauth_connection(conn_id=4, provider_account_id="bob@work.com")

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.tools.base.get_tool") as mock_get_tool, \
             patch("seer.services.integrations.auth.helpers.list_connections", new_callable=AsyncMock) as mock_list, \
             patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has_scopes:

            mock_get_user.return_value = mock_user
            mock_get_tool.return_value = mock_tool()
            mock_list.return_value = [conn1, conn2]
            mock_display.side_effect = ["alice@gmail.com", "bob@work.com"]
            mock_has_scopes.return_value = True

            result = await get_tool_accounts_impl("gmail_send_email")

            data = json.loads(result)
            assert data["requires_selection"] is True
            assert len(data["accounts"]) == 2
            assert data["accounts"][0]["id"] == 1
            assert data["accounts"][1]["id"] == 4

    @pytest.mark.asyncio
    async def test_tool_not_found(self, mock_user):
        """Test get_tool_accounts handles tool not found."""
        from seer.tools.unified_tools import get_tool_accounts_impl

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.tools.base.get_tool") as mock_get_tool:

            mock_get_user.return_value = mock_user
            mock_get_tool.return_value = None

            result = await get_tool_accounts_impl("nonexistent_tool")

            data = json.loads(result)
            assert "error" in data
            assert "not found" in data["error"]

    @pytest.mark.asyncio
    async def test_tool_without_oauth(self, mock_user, mock_tool):
        """Test get_tool_accounts handles tools without OAuth."""
        from seer.tools.unified_tools import get_tool_accounts_impl

        tool = mock_tool()
        tool.required_scopes = []

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.tools.base.get_tool") as mock_get_tool:

            mock_get_user.return_value = mock_user
            mock_get_tool.return_value = tool

            result = await get_tool_accounts_impl("some_local_tool")

            data = json.loads(result)
            assert data["accounts"] == []
            assert data["requires_selection"] is False
            assert "message" in data
            assert "not require OAuth" in data["message"]

    @pytest.mark.asyncio
    async def test_no_user_context(self):
        """Test get_tool_accounts handles missing user context."""
        from seer.tools.unified_tools import get_tool_accounts_impl

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = None

            result = await get_tool_accounts_impl("gmail_send_email")

            data = json.loads(result)
            assert "error" in data
            assert "User context" in data["error"]


class TestGetTriggerAccounts:
    """Tests for get_trigger_accounts_impl."""

    @pytest.mark.asyncio
    async def test_single_account_no_selection_required(
        self, mock_user, mock_oauth_connection, mock_trigger_definition
    ):
        """Test get_trigger_accounts returns single account without selection required."""
        from seer.tools.unified_tools import get_trigger_accounts_impl

        conn = mock_oauth_connection(provider="google", scopes="https://www.googleapis.com/auth/gmail.readonly")

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry, \
             patch("seer.services.integrations.auth.oauth.get_oauth_provider") as mock_get_provider, \
             patch("seer.database.models_oauth.OAuthConnection") as mock_conn_model, \
             patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has_scopes:

            mock_get_user.return_value = mock_user
            mock_registry.maybe_get.return_value = mock_trigger_definition()
            mock_get_provider.return_value = "google"
            mock_conn_model.filter.return_value.all = AsyncMock(return_value=[conn])
            mock_display.return_value = "alice@gmail.com"
            mock_has_scopes.return_value = True

            result = await get_trigger_accounts_impl("poll.gmail.email_received")

            data = json.loads(result)
            assert data["trigger_key"] == "poll.gmail.email_received"
            assert data["provider"] == "gmail"
            assert len(data["accounts"]) == 1
            assert data["requires_selection"] is False

    @pytest.mark.asyncio
    async def test_multiple_accounts_requires_selection(
        self, mock_user, mock_oauth_connection, mock_trigger_definition
    ):
        """Test get_trigger_accounts returns multiple accounts with selection required."""
        from seer.tools.unified_tools import get_trigger_accounts_impl

        conn1 = mock_oauth_connection(conn_id=1, provider="google")
        conn2 = mock_oauth_connection(conn_id=2, provider="google", provider_account_id="bob@gmail.com")

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry, \
             patch("seer.services.integrations.auth.oauth.get_oauth_provider") as mock_get_provider, \
             patch("seer.database.models_oauth.OAuthConnection") as mock_conn_model, \
             patch("seer.services.integrations.auth.helpers.get_connection_display_name") as mock_display, \
             patch("seer.services.integrations.auth.helpers.has_required_scopes") as mock_has_scopes:

            mock_get_user.return_value = mock_user
            mock_registry.maybe_get.return_value = mock_trigger_definition()
            mock_get_provider.return_value = "google"
            mock_conn_model.filter.return_value.all = AsyncMock(return_value=[conn1, conn2])
            mock_display.side_effect = ["alice@gmail.com", "bob@gmail.com"]
            mock_has_scopes.return_value = True

            result = await get_trigger_accounts_impl("poll.gmail.email_received")

            data = json.loads(result)
            assert data["requires_selection"] is True
            assert len(data["accounts"]) == 2

    @pytest.mark.asyncio
    async def test_trigger_not_found(self, mock_user):
        """Test get_trigger_accounts handles trigger not found."""
        from seer.tools.unified_tools import get_trigger_accounts_impl

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:

            mock_get_user.return_value = mock_user
            mock_registry.maybe_get.return_value = None

            result = await get_trigger_accounts_impl("nonexistent.trigger")

            data = json.loads(result)
            assert "error" in data
            assert "not found" in data["error"]

    @pytest.mark.asyncio
    async def test_trigger_without_oauth(self, mock_user, mock_trigger_definition):
        """Test get_trigger_accounts handles triggers without OAuth (webhook.generic)."""
        from seer.tools.unified_tools import get_trigger_accounts_impl

        trigger = mock_trigger_definition(key="webhook.generic", requires_connection=False)

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user, \
             patch("seer.core.registry.trigger_registry.trigger_registry") as mock_registry:

            mock_get_user.return_value = mock_user
            mock_registry.maybe_get.return_value = trigger

            result = await get_trigger_accounts_impl("webhook.generic")

            data = json.loads(result)
            assert data["accounts"] == []
            assert data["requires_selection"] is False
            assert "message" in data
            assert "not require OAuth" in data["message"]

    @pytest.mark.asyncio
    async def test_no_user_context(self):
        """Test get_trigger_accounts handles missing user context."""
        from seer.tools.unified_tools import get_trigger_accounts_impl

        with patch("seer.tools.unified_tools._get_unified_user", new_callable=AsyncMock) as mock_get_user:
            mock_get_user.return_value = None

            result = await get_trigger_accounts_impl("poll.gmail.email_received")

            data = json.loads(result)
            assert "error" in data
            assert "User context" in data["error"]
