"""
Unit tests for TriggerEventBrowser service.

Tests the browsing of real trigger events from connected accounts.
"""
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_user():
    """Create a mock user."""
    user = MagicMock()
    user.id = 1
    user.user_id = "user_123"
    return user


@pytest.fixture
def mock_gmail_message_data():
    """Create mock Gmail message data."""
    return {
        "id": "msg_12345",
        "threadId": "thread_12345",
        "internalDate": "1735689600000",  # 2025-01-01 00:00:00 UTC
        "snippet": "This is a test email snippet...",
        "labelIds": ["INBOX", "UNREAD"],
        "historyId": "12345",
        "payload": {
            "headers": [
                {"name": "From", "value": "Test Sender <sender@example.com>"},
                {"name": "To", "value": "recipient@example.com"},
                {"name": "Subject", "value": "Test Subject"},
                {"name": "Date", "value": "Wed, 01 Jan 2025 00:00:00 +0000"},
            ]
        }
    }


@pytest.fixture
def mock_gmail_list_response(mock_gmail_message_data):
    """Create mock Gmail list response."""
    return {
        "messages": [
            {"id": "msg_12345"},
            {"id": "msg_67890"}
        ],
        "nextPageToken": "next_page_token_123"
    }


# =============================================================================
# TriggerEventBrowser Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEventBrowser:
    """Tests for TriggerEventBrowser class."""

    def test_get_supported_trigger_keys(self):
        """Test getting supported trigger keys."""
        from seer.services.integrations.trigger_event_browser import TriggerEventBrowser

        keys = TriggerEventBrowser.get_supported_trigger_keys()

        assert "poll.gmail.email_received" in keys
        assert isinstance(keys, list)

    def test_get_provider_for_trigger(self):
        """Test getting provider for trigger key."""
        from seer.services.integrations.trigger_event_browser import TriggerEventBrowser

        provider = TriggerEventBrowser.get_provider_for_trigger("poll.gmail.email_received")
        assert provider == "google"

        provider = TriggerEventBrowser.get_provider_for_trigger("unknown.trigger")
        assert provider is None


@pytest.mark.unit
class TestTriggerEventBrowserListEvents:
    """Tests for TriggerEventBrowser.list_events method."""

    @pytest.mark.asyncio
    async def test_list_gmail_events_success(self, mock_user, mock_gmail_message_data, mock_gmail_list_response):
        """Test successful listing of Gmail events."""
        from seer.services.integrations.trigger_event_browser import (
            TriggerEventBrowser,
            TriggerEventListOptions,
        )

        browser = TriggerEventBrowser(mock_user)

        options = TriggerEventListOptions(
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
            page_size=25,
        )

        # Mock OAuth token
        mock_connection = MagicMock()
        mock_connection.id = 123

        # Mock HTTP responses
        mock_list_response = MagicMock()
        mock_list_response.status_code = 200
        mock_list_response.json.return_value = mock_gmail_list_response

        mock_msg_response = MagicMock()
        mock_msg_response.status_code = 200
        mock_msg_response.json.return_value = mock_gmail_message_data

        with patch("seer.services.integrations.trigger_event_polling.get_oauth_token", new_callable=AsyncMock) as mock_get_token, \
             patch("seer.services.integrations.trigger_event_polling.httpx.AsyncClient") as mock_client_class:

            mock_get_token.return_value = (mock_connection, "access_token_123")

            # Set up async context manager for httpx client
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_list_response, mock_msg_response, mock_msg_response])
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await browser.list_events(
                provider_connection_id=123,
                options=options,
            )

            assert "items" in result
            assert "next_page_token" in result
            assert result["trigger_key"] == "poll.gmail.email_received"
            assert result["supports_search"] is True

    @pytest.mark.asyncio
    async def test_list_gmail_events_empty_result(self, mock_user):
        """Test listing Gmail events when no messages found."""
        from seer.services.integrations.trigger_event_browser import (
            TriggerEventBrowser,
            TriggerEventListOptions,
        )

        browser = TriggerEventBrowser(mock_user)

        options = TriggerEventListOptions(
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
        )

        mock_connection = MagicMock()
        mock_list_response = MagicMock()
        mock_list_response.status_code = 200
        mock_list_response.json.return_value = {"messages": []}

        with patch("seer.services.integrations.trigger_event_polling.get_oauth_token", new_callable=AsyncMock) as mock_get_token, \
             patch("seer.services.integrations.trigger_event_polling.httpx.AsyncClient") as mock_client_class:

            mock_get_token.return_value = (mock_connection, "access_token_123")

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_list_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await browser.list_events(
                provider_connection_id=123,
                options=options,
            )

            assert result["items"] == []
            assert result["next_page_token"] is None

    @pytest.mark.asyncio
    async def test_list_events_unsupported_trigger_key(self, mock_user):
        """Test that unsupported trigger key raises ValueError."""
        from seer.services.integrations.trigger_event_browser import (
            TriggerEventBrowser,
            TriggerEventListOptions,
        )

        browser = TriggerEventBrowser(mock_user)

        options = TriggerEventListOptions(
            trigger_key="unsupported.trigger",
            trigger_id="trigger_test",
        )

        with pytest.raises(ValueError) as exc_info:
            await browser.list_events(
                provider_connection_id=123,
                options=options,
            )

        assert "does not support event browsing" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_list_gmail_events_auth_error(self, mock_user):
        """Test handling of Gmail authentication error."""
        from seer.services.integrations.trigger_event_browser import (
            TriggerEventBrowser,
            TriggerEventListOptions,
        )
        from fastapi import HTTPException

        browser = TriggerEventBrowser(mock_user)

        options = TriggerEventListOptions(
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
        )

        mock_connection = MagicMock()
        mock_error_response = MagicMock()
        mock_error_response.status_code = 401
        mock_error_response.text = "Invalid credentials"

        with patch("seer.services.integrations.trigger_event_polling.get_oauth_token", new_callable=AsyncMock) as mock_get_token, \
             patch("seer.services.integrations.trigger_event_polling.httpx.AsyncClient") as mock_client_class:

            mock_get_token.return_value = (mock_connection, "expired_token")

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_error_response)
            mock_client_class.return_value.__aenter__.return_value = mock_client

            with pytest.raises(HTTPException) as exc_info:
                await browser.list_events(
                    provider_connection_id=123,
                    options=options,
                )

            assert exc_info.value.status_code == 401
            assert "authentication" in exc_info.value.detail.lower()

    @pytest.mark.asyncio
    async def test_list_gmail_events_with_filters(self, mock_user, mock_gmail_message_data, mock_gmail_list_response):
        """Test listing Gmail events with custom filters."""
        from seer.services.integrations.trigger_event_browser import (
            TriggerEventBrowser,
            TriggerEventListOptions,
        )

        browser = TriggerEventBrowser(mock_user)

        options = TriggerEventListOptions(
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
            filter_params={
                "label_ids": ["INBOX", "STARRED"],
                "query": "is:unread"
            }
        )

        mock_connection = MagicMock()
        mock_list_response = MagicMock()
        mock_list_response.status_code = 200
        mock_list_response.json.return_value = mock_gmail_list_response

        mock_msg_response = MagicMock()
        mock_msg_response.status_code = 200
        mock_msg_response.json.return_value = mock_gmail_message_data

        with patch("seer.services.integrations.trigger_event_polling.get_oauth_token", new_callable=AsyncMock) as mock_get_token, \
             patch("seer.services.integrations.trigger_event_polling.httpx.AsyncClient") as mock_client_class:

            mock_get_token.return_value = (mock_connection, "access_token_123")

            mock_client = AsyncMock()
            mock_client.get = AsyncMock(side_effect=[mock_list_response, mock_msg_response, mock_msg_response])
            mock_client_class.return_value.__aenter__.return_value = mock_client

            result = await browser.list_events(
                provider_connection_id=123,
                options=options,
            )

            # Verify the request was made (filters are applied internally)
            assert mock_client.get.called
            assert "items" in result


# =============================================================================
# TriggerEventItem Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEventItemBuilding:
    """Tests for building TriggerEventItem from Gmail data."""

    def test_build_gmail_event_item(self, mock_user, mock_gmail_message_data):
        """Test building event item from Gmail message data."""
        from seer.services.integrations.trigger_event_polling import _build_gmail_event_item
        from seer.core.triggers.polling.adapters.gmail_email_received import GmailEmailReceivedAdapter

        item = _build_gmail_event_item(
            msg_data=mock_gmail_message_data,
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
            provider_connection_id=123,
            gmail_adapter=GmailEmailReceivedAdapter(),
        )

        # Check item structure
        assert item["id"] == "msg_12345"
        assert "Test Subject" in item["display_title"]
        assert "Test Sender" in item["display_title"]
        assert item["display_subtitle"] is not None
        assert item["preview"] == "This is a test email snippet..."

        # Check envelope structure
        envelope = item["envelope"]
        assert envelope["trigger_key"] == "poll.gmail.email_received"
        assert envelope["trigger_id"] == "trigger_test"
        assert envelope["provider"] == "gmail"
        assert envelope["account_id"] == 123
        assert "data" in envelope

        # Check payload normalization
        data = envelope["data"]
        assert data["message_id"] == "msg_12345"
        assert data["thread_id"] == "thread_12345"
        assert data["subject"] == "Test Subject"
        assert data["from"]["email"] == "sender@example.com"
        assert data["from"]["name"] == "Test Sender"

    def test_build_gmail_event_item_no_subject(self, mock_user):
        """Test building event item when email has no subject."""
        from seer.services.integrations.trigger_event_polling import _build_gmail_event_item
        from seer.core.triggers.polling.adapters.gmail_email_received import GmailEmailReceivedAdapter

        msg_data = {
            "id": "msg_no_subject",
            "threadId": "thread_123",
            "internalDate": "1735689600000",
            "snippet": "Email with no subject",
            "payload": {
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                ]
            }
        }

        item = _build_gmail_event_item(
            msg_data=msg_data,
            trigger_key="poll.gmail.email_received",
            trigger_id="trigger_test",
            provider_connection_id=123,
            gmail_adapter=GmailEmailReceivedAdapter(),
        )

        assert "(No subject)" in item["display_title"]


# =============================================================================
# TriggerEventListOptions Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerEventListOptions:
    """Tests for TriggerEventListOptions dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        from seer.services.integrations.trigger_event_browser import TriggerEventListOptions

        options = TriggerEventListOptions(trigger_key="poll.gmail.email_received")

        assert options.trigger_key == "poll.gmail.email_received"
        assert options.trigger_id == "trigger_test"
        assert options.page_size == 25
        assert options.page_token is None
        assert options.filter_params is None

    def test_custom_values(self):
        """Test custom values are preserved."""
        from seer.services.integrations.trigger_event_browser import TriggerEventListOptions

        options = TriggerEventListOptions(
            trigger_key="poll.gmail.email_received",
            trigger_id="my_trigger",
            page_size=10,
            page_token="abc123",
            filter_params={"label_ids": ["INBOX"]}
        )

        assert options.trigger_id == "my_trigger"
        assert options.page_size == 10
        assert options.page_token == "abc123"
        assert options.filter_params == {"label_ids": ["INBOX"]}


# =============================================================================
# TRIGGER_PROVIDER_MAP Tests
# =============================================================================


@pytest.mark.unit
class TestTriggerProviderMap:
    """Tests for TRIGGER_PROVIDER_MAP constant."""

    def test_gmail_trigger_mapped_to_google(self):
        """Test Gmail trigger is mapped to Google provider."""
        from seer.services.integrations.trigger_event_browser import TRIGGER_PROVIDER_MAP

        assert TRIGGER_PROVIDER_MAP["poll.gmail.email_received"] == "google"

    def test_map_contains_all_supported_triggers(self):
        """Test map contains expected triggers."""
        from seer.services.integrations.trigger_event_browser import TRIGGER_PROVIDER_MAP

        # At minimum, Gmail should be supported
        assert "poll.gmail.email_received" in TRIGGER_PROVIDER_MAP
