"""
Unit tests for trigger multi-account OAuth support.

Tests the get_trigger_accounts endpoint, MultipleAccountsError handling,
and updated _auto_select_provider_connection behavior.
"""
# pylint: disable=redefined-outer-name
# Reason: pytest fixture pattern requires name reuse
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# MultipleAccountsError Tests
# =============================================================================


@pytest.mark.unit
class TestMultipleAccountsError:
    """Tests for MultipleAccountsError exception."""

    def test_error_message_format(self):
        """Test error message includes provider and account names."""
        from seer.api.workflows.services.triggers import MultipleAccountsError

        error = MultipleAccountsError("google", ["alice@gmail.com", "bob@work.com"])

        assert "google" in str(error)
        assert "alice@gmail.com" in str(error)
        assert "bob@work.com" in str(error)
        assert "provider_connection_id" in str(error)

    def test_error_attributes(self):
        """Test error has provider and account_names attributes."""
        from seer.api.workflows.services.triggers import MultipleAccountsError

        error = MultipleAccountsError("google", ["a@g.com", "b@g.com"])

        assert error.provider == "google"
        assert error.account_names == ["a@g.com", "b@g.com"]


# =============================================================================
# Auto Select Provider Connection Tests (Updated Behavior)
# =============================================================================


@pytest.mark.unit
class TestAutoSelectProviderConnectionUpdated:
    """Tests for updated _auto_select_provider_connection function."""

    @pytest.fixture
    def mock_trigger_definition(self):
        """Create a mock trigger definition."""
        definition = MagicMock()
        definition.key = "poll.gmail.email_received"
        definition.provider = "gmail"
        return definition

    @pytest.mark.asyncio
    async def test_auto_select_single_connection(self, mock_user, mock_trigger_definition):
        """Test auto-select returns connection ID when only one account exists."""
        from seer.api.workflows.services.triggers import _auto_select_provider_connection

        mock_connection = MagicMock()
        mock_connection.id = 123

        with patch("seer.api.workflows.services.triggers.get_oauth_provider", return_value="google"):
            with patch("seer.api.workflows.services.triggers.OAuthConnection") as MockOAuthConnection:
                mock_query = MagicMock()
                mock_query.order_by = MagicMock(
                    return_value=MagicMock(all=AsyncMock(return_value=[mock_connection]))
                )
                MockOAuthConnection.filter = MagicMock(return_value=mock_query)

                result = await _auto_select_provider_connection(mock_user, mock_trigger_definition)

        assert result == 123

    @pytest.mark.asyncio
    async def test_auto_select_returns_none_when_no_connection(self, mock_user, mock_trigger_definition):
        """Test auto-select returns None when no active connection found."""
        from seer.api.workflows.services.triggers import _auto_select_provider_connection

        with patch("seer.api.workflows.services.triggers.get_oauth_provider", return_value="google"):
            with patch("seer.api.workflows.services.triggers.OAuthConnection") as MockOAuthConnection:
                mock_query = MagicMock()
                mock_query.order_by = MagicMock(
                    return_value=MagicMock(all=AsyncMock(return_value=[]))
                )
                MockOAuthConnection.filter = MagicMock(return_value=mock_query)

                result = await _auto_select_provider_connection(mock_user, mock_trigger_definition)

        assert result is None

    @pytest.mark.asyncio
    async def test_auto_select_raises_error_with_multiple_connections(
        self, mock_user, mock_trigger_definition
    ):
        """Test auto-select raises MultipleAccountsError when multiple accounts exist."""
        from seer.api.workflows.services.triggers import (
            _auto_select_provider_connection,
            MultipleAccountsError,
        )

        mock_conn1 = MagicMock()
        mock_conn1.id = 1
        mock_conn1.provider = "google"
        mock_conn1.provider_account_id = "123456789"
        mock_conn1.provider_metadata = {"email": "alice@gmail.com"}

        mock_conn2 = MagicMock()
        mock_conn2.id = 2
        mock_conn2.provider = "google"
        mock_conn2.provider_account_id = "987654321"
        mock_conn2.provider_metadata = {"email": "bob@work.com"}

        with patch("seer.api.workflows.services.triggers.get_oauth_provider", return_value="google"):
            with patch("seer.api.workflows.services.triggers.OAuthConnection") as MockOAuthConnection:
                mock_query = MagicMock()
                mock_query.order_by = MagicMock(
                    return_value=MagicMock(all=AsyncMock(return_value=[mock_conn1, mock_conn2]))
                )
                MockOAuthConnection.filter = MagicMock(return_value=mock_query)

                with pytest.raises(MultipleAccountsError) as exc_info:
                    await _auto_select_provider_connection(mock_user, mock_trigger_definition)

        assert exc_info.value.provider == "google"
        assert "alice@gmail.com" in exc_info.value.account_names
        assert "bob@work.com" in exc_info.value.account_names


# =============================================================================
# Get Trigger Accounts Tests
# =============================================================================


@pytest.mark.unit
class TestGetTriggerAccounts:
    """Tests for get_trigger_accounts service function."""

    @pytest.fixture
    def mock_trigger_definition(self):
        """Create a mock trigger definition."""
        definition = MagicMock()
        definition.key = "poll.gmail.email_received"
        definition.title = "Gmail New Email"
        definition.provider = "gmail"
        definition.meta = MagicMock()
        definition.meta.requires_connection = True
        definition.meta.required_scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        return definition

    @pytest.mark.asyncio
    async def test_get_trigger_accounts_not_found(self, mock_user):
        """Test get_trigger_accounts raises 404 for unknown trigger."""
        from seer.api.workflows.services.catalog import get_trigger_accounts
        from fastapi import HTTPException

        with patch(
            "seer.api.workflows.services.catalog.trigger_registry.get", return_value=None
        ):
            with pytest.raises(HTTPException) as exc_info:
                await get_trigger_accounts(mock_user, "unknown.trigger")

        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_get_trigger_accounts_no_connection_required(self, mock_user):
        """Test get_trigger_accounts returns empty for triggers not requiring OAuth."""
        from seer.api.workflows.services.catalog import get_trigger_accounts

        mock_definition = MagicMock()
        mock_definition.key = "webhook.generic"
        mock_definition.provider = "webhook"
        mock_definition.meta = MagicMock()
        mock_definition.meta.requires_connection = False

        with patch(
            "seer.api.workflows.services.catalog.trigger_registry.get",
            return_value=mock_definition,
        ):
            result = await get_trigger_accounts(mock_user, "webhook.generic")

        assert result.trigger_key == "webhook.generic"
        assert result.accounts == []
        assert result.requires_selection is False

    @pytest.mark.asyncio
    async def test_get_trigger_accounts_single_account(self, mock_user, mock_trigger_definition):
        """Test get_trigger_accounts returns single account without selection required."""
        from seer.api.workflows.services.catalog import get_trigger_accounts

        mock_conn = MagicMock()
        mock_conn.id = 1
        mock_conn.provider = "google"
        mock_conn.provider_account_id = "123456789"
        mock_conn.provider_metadata = {"email": "alice@gmail.com"}
        mock_conn.scopes = "https://www.googleapis.com/auth/gmail.readonly"

        with patch(
            "seer.api.workflows.services.catalog.trigger_registry.get",
            return_value=mock_trigger_definition,
        ):
            with patch(
                "seer.api.workflows.services.catalog.get_oauth_provider", return_value="google"
            ):
                with patch(
                    "seer.api.workflows.services.catalog.OAuthConnection"
                ) as MockOAuthConnection:
                    MockOAuthConnection.filter = MagicMock(
                        return_value=MagicMock(all=AsyncMock(return_value=[mock_conn]))
                    )

                    result = await get_trigger_accounts(mock_user, "poll.gmail.email_received")

        assert result.trigger_key == "poll.gmail.email_received"
        assert result.provider == "gmail"
        assert len(result.accounts) == 1
        assert result.accounts[0].id == 1
        assert result.accounts[0].display_name == "alice@gmail.com"
        assert result.requires_selection is False

    @pytest.mark.asyncio
    async def test_get_trigger_accounts_multiple_accounts(self, mock_user, mock_trigger_definition):
        """Test get_trigger_accounts returns multiple accounts with selection required."""
        from seer.api.workflows.services.catalog import get_trigger_accounts

        mock_conn1 = MagicMock()
        mock_conn1.id = 1
        mock_conn1.provider = "google"
        mock_conn1.provider_account_id = "123456789"
        mock_conn1.provider_metadata = {"email": "alice@gmail.com"}
        mock_conn1.scopes = "https://www.googleapis.com/auth/gmail.readonly"

        mock_conn2 = MagicMock()
        mock_conn2.id = 2
        mock_conn2.provider = "google"
        mock_conn2.provider_account_id = "987654321"
        mock_conn2.provider_metadata = {"email": "bob@work.com"}
        mock_conn2.scopes = "https://www.googleapis.com/auth/gmail.readonly"

        with patch(
            "seer.api.workflows.services.catalog.trigger_registry.get",
            return_value=mock_trigger_definition,
        ):
            with patch(
                "seer.api.workflows.services.catalog.get_oauth_provider", return_value="google"
            ):
                with patch(
                    "seer.api.workflows.services.catalog.OAuthConnection"
                ) as MockOAuthConnection:
                    MockOAuthConnection.filter = MagicMock(
                        return_value=MagicMock(all=AsyncMock(return_value=[mock_conn1, mock_conn2]))
                    )

                    result = await get_trigger_accounts(mock_user, "poll.gmail.email_received")

        assert result.trigger_key == "poll.gmail.email_received"
        assert len(result.accounts) == 2
        assert result.requires_selection is True
        display_names = [a.display_name for a in result.accounts]
        assert "alice@gmail.com" in display_names
        assert "bob@work.com" in display_names

    @pytest.mark.asyncio
    async def test_get_trigger_accounts_scope_validation(self, mock_user, mock_trigger_definition):
        """Test get_trigger_accounts validates scopes correctly."""
        from seer.api.workflows.services.catalog import get_trigger_accounts

        # Account with missing scopes
        mock_conn = MagicMock()
        mock_conn.id = 1
        mock_conn.provider = "google"
        mock_conn.provider_account_id = "123456789"
        mock_conn.provider_metadata = {"email": "alice@gmail.com"}
        mock_conn.scopes = "https://www.googleapis.com/auth/gmail.compose"  # Different scope

        with patch(
            "seer.api.workflows.services.catalog.trigger_registry.get",
            return_value=mock_trigger_definition,
        ):
            with patch(
                "seer.api.workflows.services.catalog.get_oauth_provider", return_value="google"
            ):
                with patch(
                    "seer.api.workflows.services.catalog.OAuthConnection"
                ) as MockOAuthConnection:
                    MockOAuthConnection.filter = MagicMock(
                        return_value=MagicMock(all=AsyncMock(return_value=[mock_conn]))
                    )

                    result = await get_trigger_accounts(mock_user, "poll.gmail.email_received")

        assert len(result.accounts) == 1
        assert result.accounts[0].has_required_scopes is False
        assert "https://www.googleapis.com/auth/gmail.readonly" in result.accounts[0].missing_scopes


# =============================================================================
# Subscription Response Display Name Tests
# =============================================================================


@pytest.mark.unit
class TestSubscriptionDisplayName:
    """Tests for connection_display_name in subscription responses."""

    @pytest.mark.asyncio
    async def test_serialize_subscription_includes_display_name(self):
        """Test _serialize_subscription includes connection_display_name."""
        from seer.api.workflows.services.triggers import _serialize_subscription

        mock_subscription = MagicMock()
        mock_subscription.id = 1
        mock_subscription.workflow_id = 1
        mock_subscription.trigger_key = "poll.gmail.email_received"
        mock_subscription.provider_connection_id = 123
        mock_subscription.enabled = True
        mock_subscription.filters = {}
        mock_subscription.provider_config = {}
        mock_subscription.secret_token = None
        mock_subscription.form_suffix = None
        mock_subscription.form_fields = None
        mock_subscription.form_config = None
        mock_subscription.created_at = MagicMock()
        mock_subscription.updated_at = MagicMock()

        mock_conn = MagicMock()
        mock_conn.provider = "google"
        mock_conn.provider_account_id = "123456789"
        mock_conn.provider_metadata = {"email": "alice@gmail.com"}

        with patch(
            "seer.api.workflows.services.triggers.make_workflow_public_id",
            return_value="wf_abc123",
        ):
            with patch(
                "seer.api.workflows.services.triggers.OAuthConnection"
            ) as MockOAuthConnection:
                MockOAuthConnection.get_or_none = AsyncMock(return_value=mock_conn)

                result = await _serialize_subscription(mock_subscription)

        assert result.connection_display_name == "alice@gmail.com"
        assert result.provider_connection_id == 123

    @pytest.mark.asyncio
    async def test_serialize_subscription_no_connection(self):
        """Test _serialize_subscription handles None provider_connection_id."""
        from seer.api.workflows.services.triggers import _serialize_subscription

        mock_subscription = MagicMock()
        mock_subscription.id = 1
        mock_subscription.workflow_id = 1
        mock_subscription.trigger_key = "webhook.generic"
        mock_subscription.provider_connection_id = None  # No connection
        mock_subscription.enabled = True
        mock_subscription.filters = {}
        mock_subscription.provider_config = {}
        mock_subscription.secret_token = "secret123"
        mock_subscription.form_suffix = None
        mock_subscription.form_fields = None
        mock_subscription.form_config = None
        mock_subscription.created_at = MagicMock()
        mock_subscription.updated_at = MagicMock()

        with patch(
            "seer.api.workflows.services.triggers.make_workflow_public_id",
            return_value="wf_abc123",
        ):
            result = await _serialize_subscription(mock_subscription)

        assert result.connection_display_name is None
        assert result.provider_connection_id is None
