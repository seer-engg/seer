"""
Unit tests for single tool status endpoint.

Tests the GET /api/integrations/tools/{tool_name}/status endpoint that supports
connection-specific status queries for multi-account OAuth scenarios.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


# =============================================================================
# get_single_tool_status Service Function Tests
# =============================================================================


@pytest.mark.unit
class TestGetSingleToolStatusService:
    """Tests for get_single_tool_status service function."""

    @pytest.fixture
    def mock_user(self):
        """Create a mock User object."""
        user = MagicMock()
        user.user_id = "test_user_123"
        return user

    @pytest.fixture
    def mock_tool(self):
        """Create a mock tool with required properties."""
        tool = MagicMock()
        tool.name = "gmail_send_email"
        tool.provider = "google"
        tool.integration_type = "gmail"
        tool.required_scopes = ["https://www.googleapis.com/auth/gmail.send"]
        return tool

    @pytest.fixture
    def mock_oauth_connection(self):
        """Create a mock OAuthConnection."""
        conn = MagicMock()
        conn.id = 123
        conn.provider = "google"
        conn.provider_account_id = "alice@gmail.com"
        conn.scopes = "https://www.googleapis.com/auth/gmail.send https://www.googleapis.com/auth/gmail.readonly"
        conn.refresh_token_enc = "encrypted_token"
        conn.status = "active"
        return conn

    @pytest.mark.asyncio
    async def test_get_single_tool_status_tool_not_found(self, mock_user):
        """Test returns None when tool is not found."""
        from seer.services.integrations.tool_status_service import get_single_tool_status

        # get_tool is imported inside the function from seer.tools.base
        with patch("seer.tools.base.get_tool", return_value=None):
            result = await get_single_tool_status(mock_user, "nonexistent_tool")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_single_tool_status_no_connection_id(self, mock_user, mock_tool, mock_oauth_connection):
        """Test returns global status when no connection_id is provided."""
        from seer.services.integrations.tool_status_service import get_single_tool_status

        with patch("seer.tools.base.get_tool", return_value=mock_tool), \
             patch("seer.services.integrations.tool_status_service.get_oauth_provider", return_value="google"), \
             patch("seer.services.integrations.tool_status_service.list_connections_with_shared", new_callable=AsyncMock, return_value=[mock_oauth_connection]), \
             patch("seer.services.integrations.tool_status_service.build_provider_secrets_map", new_callable=AsyncMock, return_value={}):

            result = await get_single_tool_status(mock_user, "gmail_send_email")

        assert result is not None
        assert result["tool_name"] == "gmail_send_email"
        assert result["provider"] == "google"
        assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_get_single_tool_status_with_valid_connection_id(self, mock_user, mock_tool, mock_oauth_connection):
        """Test returns connection-specific status when connection_id is provided."""
        from seer.services.integrations.tool_status_service import get_single_tool_status
        from seer.database import OAuthConnection

        # Mock the filter().first() chain
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=mock_oauth_connection)

        with patch("seer.tools.base.get_tool", return_value=mock_tool), \
             patch("seer.services.integrations.tool_status_service.get_oauth_provider", return_value="google"), \
             patch.object(OAuthConnection, "filter", return_value=mock_filter), \
             patch("seer.services.integrations.tool_status_service.build_provider_secrets_map", new_callable=AsyncMock, return_value={}):

            result = await get_single_tool_status(mock_user, "gmail_send_email", connection_id=123)

        assert result is not None
        assert result["tool_name"] == "gmail_send_email"
        assert result["connected"] is True
        assert result["connection_id"] == "google:123"
        assert result["provider_account_id"] == "alice@gmail.com"

    @pytest.mark.asyncio
    async def test_get_single_tool_status_with_invalid_connection_id(self, mock_user, mock_tool):
        """Test returns not connected when connection_id is not found."""
        from seer.services.integrations.tool_status_service import get_single_tool_status
        from seer.database import OAuthConnection

        # Mock the filter().first() chain returning None
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=None)

        with patch("seer.tools.base.get_tool", return_value=mock_tool), \
             patch("seer.services.integrations.tool_status_service.get_oauth_provider", return_value="google"), \
             patch.object(OAuthConnection, "filter", return_value=mock_filter), \
             patch("seer.services.integrations.tool_status_service.build_provider_secrets_map", new_callable=AsyncMock, return_value={}):

            result = await get_single_tool_status(mock_user, "gmail_send_email", connection_id=999)

        assert result is not None
        assert result["tool_name"] == "gmail_send_email"
        assert result["connected"] is False
        assert result["connection_id"] is None

    @pytest.mark.asyncio
    async def test_get_single_tool_status_connection_wrong_provider(self, mock_user, mock_tool):
        """Test returns not connected when connection exists but for wrong provider."""
        from seer.services.integrations.tool_status_service import get_single_tool_status
        from seer.database import OAuthConnection

        # Create a connection for a different provider
        wrong_provider_conn = MagicMock()
        wrong_provider_conn.id = 123
        wrong_provider_conn.provider = "github"  # Wrong provider for gmail tool
        wrong_provider_conn.scopes = "repo user"

        # Mock the filter().first() chain
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=wrong_provider_conn)

        with patch("seer.tools.base.get_tool", return_value=mock_tool), \
             patch("seer.services.integrations.tool_status_service.get_oauth_provider", return_value="google"), \
             patch.object(OAuthConnection, "filter", return_value=mock_filter), \
             patch("seer.services.integrations.tool_status_service.build_provider_secrets_map", new_callable=AsyncMock, return_value={}):

            result = await get_single_tool_status(mock_user, "gmail_send_email", connection_id=123)

        assert result is not None
        assert result["connected"] is False

    @pytest.mark.asyncio
    async def test_get_single_tool_status_missing_scopes(self, mock_user, mock_tool):
        """Test correctly identifies missing scopes for a connection."""
        from seer.services.integrations.tool_status_service import get_single_tool_status
        from seer.database import OAuthConnection

        # Connection with read-only scope (missing send scope)
        read_only_conn = MagicMock()
        read_only_conn.id = 123
        read_only_conn.provider = "google"
        read_only_conn.provider_account_id = "alice@gmail.com"
        read_only_conn.scopes = "https://www.googleapis.com/auth/gmail.readonly"  # Missing send scope
        read_only_conn.refresh_token_enc = "encrypted_token"

        # Mock the filter().first() chain
        mock_filter = MagicMock()
        mock_filter.first = AsyncMock(return_value=read_only_conn)

        with patch("seer.tools.base.get_tool", return_value=mock_tool), \
             patch("seer.services.integrations.tool_status_service.get_oauth_provider", return_value="google"), \
             patch.object(OAuthConnection, "filter", return_value=mock_filter), \
             patch("seer.services.integrations.tool_status_service.build_provider_secrets_map", new_callable=AsyncMock, return_value={}):

            result = await get_single_tool_status(mock_user, "gmail_send_email", connection_id=123)

        assert result is not None
        assert result["connected"] is False
        assert "https://www.googleapis.com/auth/gmail.send" in result["missing_scopes"]


# =============================================================================
# get_tool_status Endpoint Tests
# =============================================================================


@pytest.mark.unit
class TestGetToolStatusEndpoint:
    """Tests for GET /api/integrations/tools/{tool_name}/status endpoint."""

    @pytest.fixture
    def mock_request(self):
        """Create a mock request with user state and organization."""
        request = MagicMock()
        request.state.db_user = MagicMock()
        request.state.db_user.user_id = "test_user_123"
        # Mock organization for shared connection lookup
        request.state.organization = MagicMock()
        request.state.organization.id = 1
        return request

    @pytest.mark.asyncio
    async def test_get_tool_status_endpoint_no_connection_id(self, mock_request):
        """Test endpoint returns global status when no connection_id provided."""
        from seer.api.integrations.router import get_tool_status

        expected_status = {
            "tool_name": "gmail_send_email",
            "integration_type": "gmail",
            "provider": "google",
            "supports_oauth": True,
            "supports_manual_secrets": False,
            "connected": True,
            "missing_scopes": [],
            "connection_id": "google:123",
            "provider_account_id": "alice@gmail.com",
        }

        with patch("seer.api.integrations.router.get_single_tool_status", new_callable=AsyncMock, return_value=expected_status):
            result = await get_tool_status(mock_request, "gmail_send_email", connection_id=None)

        assert result["tool_name"] == "gmail_send_email"
        assert result["connected"] is True

    @pytest.mark.asyncio
    async def test_get_tool_status_endpoint_with_connection_id(self, mock_request):
        """Test endpoint returns connection-specific status."""
        from seer.api.integrations.router import get_tool_status

        expected_status = {
            "tool_name": "gmail_send_email",
            "integration_type": "gmail",
            "provider": "google",
            "supports_oauth": True,
            "supports_manual_secrets": False,
            "connected": True,
            "missing_scopes": [],
            "connection_id": "google:456",
            "provider_account_id": "bob@gmail.com",
        }

        with patch("seer.api.integrations.router.get_single_tool_status", new_callable=AsyncMock, return_value=expected_status) as mock_service:
            result = await get_tool_status(mock_request, "gmail_send_email", connection_id=456)

        # Now called with organization_id as 4th argument
        mock_service.assert_called_once_with(mock_request.state.db_user, "gmail_send_email", 456, 1)
        assert result["connection_id"] == "google:456"
        assert result["provider_account_id"] == "bob@gmail.com"

    @pytest.mark.asyncio
    async def test_get_tool_status_endpoint_tool_not_found(self, mock_request):
        """Test endpoint raises exception when tool is not found."""
        from seer.api.integrations.router import get_tool_status

        with patch("seer.api.integrations.router.get_single_tool_status", new_callable=AsyncMock, return_value=None):
            # The endpoint uses raise_problem which raises ProblemDetailException
            with pytest.raises(Exception) as exc_info:
                await get_tool_status(mock_request, "nonexistent_tool", connection_id=None)

            # Check the exception message contains expected text
            assert "not found" in str(exc_info.value).lower() or "Tool" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_tool_status_endpoint_invalid_connection_returns_not_connected(self, mock_request):
        """Test endpoint returns not connected for invalid connection_id."""
        from seer.api.integrations.router import get_tool_status

        expected_status = {
            "tool_name": "gmail_send_email",
            "integration_type": "gmail",
            "provider": "google",
            "supports_oauth": True,
            "supports_manual_secrets": False,
            "connected": False,  # Not connected because connection not found
            "missing_scopes": ["https://www.googleapis.com/auth/gmail.send"],
            "connection_id": None,
            "provider_account_id": None,
        }

        with patch("seer.api.integrations.router.get_single_tool_status", new_callable=AsyncMock, return_value=expected_status):
            result = await get_tool_status(mock_request, "gmail_send_email", connection_id=999)

        assert result["connected"] is False
        assert result["connection_id"] is None
        assert len(result["missing_scopes"]) > 0
