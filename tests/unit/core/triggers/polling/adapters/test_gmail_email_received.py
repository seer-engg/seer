"""Unit tests for GmailEmailReceivedAdapter, including headers handling."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.core.triggers.polling.adapters.gmail_email_received import (
    GmailEmailReceivedAdapter,
    _parse_address,
    _parse_address_list,
)
from seer.core.triggers.polling.adapters.base import (
    PollContext,
    PollAdapterError,
)
from seer.database import TriggerSubscription, User, OAuthConnection


@pytest.fixture
def mock_poll_context():
    """Create a mock PollContext for testing."""
    subscription = Mock(spec=TriggerSubscription)
    subscription.provider_config = {
        "label_ids": ["INBOX"],
        "max_results": 10,
    }

    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)

    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token="test_token",
    )


@pytest.fixture
def adapter():
    """Create a GmailEmailReceivedAdapter instance."""
    return GmailEmailReceivedAdapter()


@pytest.mark.unit
class TestMetadataHeadersParameter:
    """Test that metadataHeaders is passed correctly as a list for repeated query params."""

    @pytest.mark.asyncio
    async def test_poll_passes_metadata_headers_as_list(self, adapter, mock_poll_context):
        """
        Test that metadataHeaders is passed as a list, not a comma-separated string.

        Gmail API requires repeated query parameters for metadataHeaders:
        ?metadataHeaders=From&metadataHeaders=To&metadataHeaders=Subject&metadataHeaders=Date

        httpx requires a list to produce this format.
        """
        cursor = {"watermark_ms": 1000000000000, "overlap_ms": 300000}

        # Mock Gmail list response with one message
        list_response = Mock(spec=httpx.Response)
        list_response.status_code = 200
        list_response.json.return_value = {"messages": [{"id": "msg123"}]}

        # Mock Gmail get message response with headers
        msg_response = Mock(spec=httpx.Response)
        msg_response.status_code = 200
        msg_response.json.return_value = {
            "id": "msg123",
            "threadId": "thread123",
            "internalDate": "1706889327000",
            "snippet": "Test email",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "From", "value": "sender@example.com"},
                    {"name": "To", "value": "recipient@example.com"},
                    {"name": "Subject", "value": "Test Subject"},
                    {"name": "Date", "value": "Wed, 4 Feb 2026 16:35:27 +0000"},
                ],
            },
            "labelIds": ["INBOX"],
        }

        captured_params = {}

        async def mock_get(url, headers=None, params=None):
            if "/messages/" in url and "msg123" in url:
                # Capture the params for the message get request
                captured_params["msg_get"] = params
                return msg_response
            return list_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        # Verify metadataHeaders was passed as a list
        assert "msg_get" in captured_params, "Message get request was not captured"
        params = captured_params["msg_get"]

        assert "metadataHeaders" in params, "metadataHeaders parameter missing"
        assert isinstance(params["metadataHeaders"], list), (
            f"metadataHeaders should be a list, got {type(params['metadataHeaders'])}"
        )
        assert params["metadataHeaders"] == ["From", "To", "Subject", "Date"], (
            f"metadataHeaders should be ['From', 'To', 'Subject', 'Date'], got {params['metadataHeaders']}"
        )

    @pytest.mark.asyncio
    async def test_poll_correctly_parses_headers_from_response(self, adapter, mock_poll_context):
        """Test that headers from the Gmail API response are correctly parsed."""
        cursor = {"watermark_ms": 1000000000000, "overlap_ms": 300000}

        list_response = Mock(spec=httpx.Response)
        list_response.status_code = 200
        list_response.json.return_value = {"messages": [{"id": "msg123"}]}

        msg_response = Mock(spec=httpx.Response)
        msg_response.status_code = 200
        msg_response.json.return_value = {
            "id": "msg123",
            "threadId": "thread123",
            "internalDate": "1706889327000",
            "snippet": "Test email snippet",
            "payload": {
                "mimeType": "multipart/alternative",
                "headers": [
                    {"name": "From", "value": "John Doe <john@example.com>"},
                    {"name": "To", "value": "Jane Doe <jane@example.com>"},
                    {"name": "Subject", "value": "Important Email Subject"},
                    {"name": "Date", "value": "Wed, 4 Feb 2026 16:35:27 +0000"},
                ],
            },
            "labelIds": ["INBOX", "IMPORTANT"],
            "historyId": "12345",
        }

        async def mock_get(url, headers=None, params=None):
            if "/messages/" in url and "msg123" in url:
                return msg_response
            return list_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 1
        event = result.events[0]
        payload = event.payload

        # Verify headers are correctly parsed in the normalized payload
        assert payload["subject"] == "Important Email Subject"
        assert payload["date_header"] == "Wed, 4 Feb 2026 16:35:27 +0000"

        # Verify from parsing
        assert payload["from"]["name"] == "John Doe"
        assert payload["from"]["email"] == "john@example.com"

        # Verify to parsing (returns list)
        assert len(payload["to"]) == 1
        assert payload["to"][0]["name"] == "Jane Doe"
        assert payload["to"][0]["email"] == "jane@example.com"

        # Verify raw data contains headers
        assert "headers" in event.raw["payload"]
        assert len(event.raw["payload"]["headers"]) == 4


@pytest.mark.unit
class TestNormalizeMessage:
    """Test the _normalize_message method."""

    def test_normalize_message_with_headers(self, adapter):
        """Test normalization with complete headers."""
        msg_data = {
            "id": "msg123",
            "threadId": "thread123",
            "internalDate": "1706889327000",
            "snippet": "Test snippet",
            "payload": {
                "headers": [
                    {"name": "From", "value": "Alice <alice@example.com>"},
                    {"name": "To", "value": "Bob <bob@example.com>, Carol <carol@example.com>"},
                    {"name": "Subject", "value": "Hello"},
                    {"name": "Date", "value": "Wed, 4 Feb 2026 10:00:00 +0000"},
                ],
            },
            "labelIds": ["INBOX"],
            "historyId": "999",
        }

        result = adapter._normalize_message(msg_data)

        assert result["message_id"] == "msg123"
        assert result["thread_id"] == "thread123"
        assert result["subject"] == "Hello"
        assert result["date_header"] == "Wed, 4 Feb 2026 10:00:00 +0000"
        assert result["from"]["name"] == "Alice"
        assert result["from"]["email"] == "alice@example.com"
        assert len(result["to"]) == 2
        assert result["to"][0]["email"] == "bob@example.com"
        assert result["to"][1]["email"] == "carol@example.com"

    def test_normalize_message_without_headers(self, adapter):
        """Test normalization when headers are missing (pre-fix behavior)."""
        msg_data = {
            "id": "msg123",
            "threadId": "thread123",
            "internalDate": "1706889327000",
            "snippet": "Test snippet",
            "payload": {
                "mimeType": "multipart/alternative",
                # No headers - this was the bug scenario
            },
            "labelIds": ["INBOX"],
        }

        result = adapter._normalize_message(msg_data)

        assert result["message_id"] == "msg123"
        assert result["subject"] is None
        assert result["date_header"] is None
        assert result["from"]["name"] is None
        assert result["from"]["email"] is None
        assert result["to"] == []


@pytest.mark.unit
class TestAddressParsingHelpers:
    """Test the address parsing helper functions."""

    def test_parse_address_with_name(self):
        """Test parsing address with display name."""
        result = _parse_address("John Doe <john@example.com>")
        assert result["name"] == "John Doe"
        assert result["email"] == "john@example.com"

    def test_parse_address_without_name(self):
        """Test parsing address without display name."""
        result = _parse_address("john@example.com")
        assert result["name"] is None
        assert result["email"] == "john@example.com"

    def test_parse_address_empty(self):
        """Test parsing empty address."""
        result = _parse_address("")
        assert result["name"] is None
        assert result["email"] is None

    def test_parse_address_none(self):
        """Test parsing None address."""
        result = _parse_address(None)
        assert result["name"] is None
        assert result["email"] is None

    def test_parse_address_list_multiple(self):
        """Test parsing multiple addresses."""
        result = _parse_address_list("Alice <alice@example.com>, Bob <bob@example.com>")
        assert len(result) == 2
        assert result[0]["name"] == "Alice"
        assert result[0]["email"] == "alice@example.com"
        assert result[1]["name"] == "Bob"
        assert result[1]["email"] == "bob@example.com"

    def test_parse_address_list_empty(self):
        """Test parsing empty address list."""
        result = _parse_address_list("")
        assert result == []

    def test_parse_address_list_none(self):
        """Test parsing None address list."""
        result = _parse_address_list(None)
        assert result == []


@pytest.mark.unit
class TestBootstrapCursor:
    """Test the bootstrap_cursor method."""

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_returns_initial_watermark(self, adapter, mock_poll_context):
        """Test bootstrap_cursor returns current time as watermark."""
        cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert "watermark_ms" in cursor
        assert "overlap_ms" in cursor
        assert cursor["overlap_ms"] == 300000  # 5 minutes default

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_uses_config_overlap(self, adapter, mock_poll_context):
        """Test bootstrap_cursor respects overlap_ms from config."""
        mock_poll_context.subscription.provider_config["overlap_ms"] = 600000  # 10 minutes

        cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["overlap_ms"] == 600000


@pytest.mark.unit
class TestRaiseForStatus:
    """Test the _raise_for_status method directly."""

    @pytest.mark.asyncio
    async def test_raise_for_status_401(self, adapter):
        """Test _raise_for_status raises PollAdapterError on 401 response."""
        response = Mock()
        response.status_code = 401
        response.text = "Unauthorized"

        with pytest.raises(PollAdapterError, match="Gmail authentication error"):
            await adapter._raise_for_status(response)

    @pytest.mark.asyncio
    async def test_raise_for_status_403(self, adapter):
        """Test _raise_for_status raises PollAdapterError on 403 response."""
        response = Mock()
        response.status_code = 403
        response.text = "Forbidden"

        with pytest.raises(PollAdapterError, match="Gmail authentication error"):
            await adapter._raise_for_status(response)

    @pytest.mark.asyncio
    async def test_raise_for_status_429(self, adapter):
        """Test _raise_for_status raises PollAdapterError with backoff on 429 response."""
        response = Mock()
        response.status_code = 429
        response.text = "Rate limited"

        with pytest.raises(PollAdapterError, match="Gmail rate limited"):
            await adapter._raise_for_status(response)

    @pytest.mark.asyncio
    async def test_raise_for_status_500(self, adapter):
        """Test _raise_for_status raises PollAdapterError on 500 response."""
        response = Mock()
        response.status_code = 500
        response.text = "Internal error"

        with pytest.raises(PollAdapterError, match="Gmail API error"):
            await adapter._raise_for_status(response)

    @pytest.mark.asyncio
    async def test_raise_for_status_200_no_raise(self, adapter):
        """Test _raise_for_status does not raise on 200 response."""
        response = Mock()
        response.status_code = 200

        # Should not raise
        await adapter._raise_for_status(response)


@pytest.mark.unit
class TestPollWithEmptyMessages:
    """Test poll method with empty message list."""

    @pytest.mark.asyncio
    async def test_poll_returns_empty_when_no_messages(self, adapter, mock_poll_context):
        """Test poll returns empty result when no messages."""
        cursor = {"watermark_ms": 1000000000000, "overlap_ms": 300000}

        empty_response = Mock()
        empty_response.status_code = 200
        empty_response.json.return_value = {"messages": []}

        async def mock_get(*_args, **_kwargs):
            return empty_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.gmail_email_received.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events == []
        assert result.cursor["watermark_ms"] == 1000000000000
