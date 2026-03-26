"""Unit tests for GoogleCalendarEventChangedAdapter."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.core.triggers.polling.adapters.google_calendar_common import (
    parse_rfc3339 as _parse_rfc3339,
)
from seer.core.triggers.polling.adapters.google_calendar_event_changed import (
    GoogleCalendarEventChangedAdapter,
    _is_newly_created,
    _normalize_event_with_type as _normalize_event,
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
        "calendar_id": "primary",
        "max_results": 25,
    }

    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)

    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token="test_access_token",
    )


@pytest.fixture
def adapter():
    """Create a GoogleCalendarEventChangedAdapter instance."""
    return GoogleCalendarEventChangedAdapter()


@pytest.mark.unit
class TestParseRfc3339:
    """Tests for the _parse_rfc3339 helper function."""

    def test_parse_rfc3339_with_z(self):
        """Test parsing timestamp with Z suffix."""
        result = _parse_rfc3339("2026-01-15T10:00:00Z")
        assert result is not None
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 10

    def test_parse_rfc3339_with_offset(self):
        """Test parsing timestamp with timezone offset."""
        result = _parse_rfc3339("2026-01-15T10:00:00-05:00")
        assert result is not None
        assert result.hour == 10

    def test_parse_rfc3339_empty(self):
        """Test parsing empty string returns None."""
        assert _parse_rfc3339("") is None
        assert _parse_rfc3339(None) is None


@pytest.mark.unit
class TestIsNewlyCreated:
    """Tests for the _is_newly_created helper function."""

    def test_newly_created_same_timestamps(self):
        """Test event with same created/updated timestamps is detected as created."""
        event = {
            "created": "2026-01-15T10:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        assert _is_newly_created(event) is True

    def test_not_newly_created_different_timestamps(self):
        """Test event with different timestamps is detected as updated."""
        event = {
            "created": "2026-01-15T10:00:00Z",
            "updated": "2026-01-15T12:00:00Z",
        }
        assert _is_newly_created(event) is False

    def test_newly_created_within_one_second(self):
        """Test event created within 1 second of update is detected as created."""
        event = {
            "created": "2026-01-15T10:00:00.000Z",
            "updated": "2026-01-15T10:00:00.500Z",
        }
        assert _is_newly_created(event) is True

    def test_missing_timestamps(self):
        """Test missing timestamps returns False."""
        assert _is_newly_created({}) is False
        assert _is_newly_created({"created": "2026-01-15T10:00:00Z"}) is False


@pytest.mark.unit
class TestNormalizeEvent:
    """Tests for the _normalize_event helper function."""

    def test_normalize_event_with_attendees(self):
        """Test normalizing a full event with attendees."""
        event = {
            "id": "event123",
            "summary": "Team Meeting",
            "description": "Weekly sync",
            "location": "Room A",
            "status": "confirmed",
            "htmlLink": "https://calendar.google.com/event?eid=xxx",
            "created": "2026-01-15T10:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
            "start": {"dateTime": "2026-01-15T14:00:00-05:00", "timeZone": "America/New_York"},
            "end": {"dateTime": "2026-01-15T15:00:00-05:00", "timeZone": "America/New_York"},
            "organizer": {"email": "org@example.com", "displayName": "Organizer", "self": True},
            "attendees": [
                {"email": "a@example.com", "displayName": "Alice", "responseStatus": "accepted"},
                {"email": "b@example.com", "displayName": "Bob", "responseStatus": "needsAction", "optional": True},
            ],
        }

        result = _normalize_event(event, "primary")

        assert result["event_id"] == "event123"
        assert result["event_type"] == "created"  # same created/updated timestamps
        assert result["calendar_id"] == "primary"
        assert result["summary"] == "Team Meeting"
        assert result["status"] == "confirmed"
        assert result["is_all_day"] is False
        assert len(result["attendees"]) == 2
        assert result["attendees"][0]["email"] == "a@example.com"
        assert result["attendees"][1]["optional"] is True

    def test_normalize_all_day_event(self):
        """Test normalizing an all-day event."""
        event = {
            "id": "allday123",
            "summary": "Holiday",
            "created": "2026-01-10T10:00:00Z",
            "updated": "2026-01-10T12:00:00Z",  # Different - so it's an update
            "start": {"date": "2026-01-01"},
            "end": {"date": "2026-01-02"},
        }

        result = _normalize_event(event, "work_calendar")

        assert result["event_id"] == "allday123"
        assert result["event_type"] == "updated"  # timestamps differ
        assert result["is_all_day"] is True
        assert result["start"]["datetime"] == "2026-01-01"


@pytest.mark.unit
class TestBootstrapCursor:
    """Tests for the bootstrap_cursor method."""

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_gets_sync_token(self, adapter, mock_poll_context):
        """Test bootstrap_cursor performs full sync and returns syncToken."""
        sync_response = Mock(spec=httpx.Response)
        sync_response.status_code = 200
        sync_response.json.return_value = {
            "items": [{"id": "event1"}],
            "nextSyncToken": "initial_sync_token_abc",
        }

        async def mock_get(*args, **kwargs):
            return sync_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert "sync_token" in cursor
        assert cursor["sync_token"] == "initial_sync_token_abc"
        assert cursor["calendar_id"] == "primary"

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_paginates(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles pagination during full sync."""
        page1_response = Mock(spec=httpx.Response)
        page1_response.status_code = 200
        page1_response.json.return_value = {
            "items": [{"id": "event1"}],
            "nextPageToken": "page2_token",
        }

        page2_response = Mock(spec=httpx.Response)
        page2_response.status_code = 200
        page2_response.json.return_value = {
            "items": [{"id": "event2"}],
            "nextSyncToken": "final_sync_token",
        }

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return page1_response
            return page2_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["sync_token"] == "final_sync_token"
        assert call_count == 2


@pytest.mark.unit
class TestPoll:
    """Tests for the poll method."""

    @pytest.mark.asyncio
    async def test_poll_returns_events(self, adapter, mock_poll_context):
        """Test polling returns normalized events."""
        cursor = {"sync_token": "existing_token", "calendar_id": "primary"}

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "new_event",
                    "summary": "New Meeting",
                    "status": "confirmed",
                    "created": "2026-01-15T10:00:00Z",
                    "updated": "2026-01-15T10:00:00Z",
                    "start": {"dateTime": "2026-01-20T14:00:00Z"},
                    "end": {"dateTime": "2026-01-20T15:00:00Z"},
                }
            ],
            "nextSyncToken": "new_sync_token",
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 1
        assert result.events[0].payload["event_id"] == "new_event"
        assert result.events[0].payload["event_type"] == "created"
        assert result.cursor["sync_token"] == "new_sync_token"
        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_poll_handles_410_gone(self, adapter, mock_poll_context):
        """Test poll handles 410 Gone by performing full resync."""
        cursor = {"sync_token": "expired_token", "calendar_id": "primary"}

        gone_response = Mock(spec=httpx.Response)
        gone_response.status_code = 410
        gone_response.text = "Sync token expired"

        resync_response = Mock(spec=httpx.Response)
        resync_response.status_code = 200
        resync_response.json.return_value = {
            "items": [],
            "nextSyncToken": "fresh_sync_token",
        }

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return gone_response
            return resync_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        # Should have performed full resync
        assert result.cursor["sync_token"] == "fresh_sync_token"
        assert result.events == []

    @pytest.mark.asyncio
    async def test_poll_skips_cancelled_events(self, adapter, mock_poll_context):
        """Test that cancelled events (deletions) are skipped."""
        cursor = {"sync_token": "token", "calendar_id": "primary"}

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {"id": "active_event", "status": "confirmed", "created": "2026-01-15T10:00:00Z", "updated": "2026-01-15T10:00:00Z"},
                {"id": "cancelled_event", "status": "cancelled"},
            ],
            "nextSyncToken": "next_token",
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 1
        assert result.events[0].payload["event_id"] == "active_event"

    @pytest.mark.asyncio
    async def test_poll_event_type_detection_created(self, adapter, mock_poll_context):
        """Test event_type is 'created' when timestamps are the same."""
        cursor = {"sync_token": "token", "calendar_id": "primary"}

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "new_event",
                    "status": "confirmed",
                    "created": "2026-01-15T10:00:00Z",
                    "updated": "2026-01-15T10:00:00Z",
                }
            ],
            "nextSyncToken": "next_token",
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events[0].payload["event_type"] == "created"

    @pytest.mark.asyncio
    async def test_poll_event_type_detection_updated(self, adapter, mock_poll_context):
        """Test event_type is 'updated' when timestamps differ."""
        cursor = {"sync_token": "token", "calendar_id": "primary"}

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "modified_event",
                    "status": "confirmed",
                    "created": "2026-01-10T10:00:00Z",
                    "updated": "2026-01-15T14:00:00Z",
                }
            ],
            "nextSyncToken": "next_token",
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events[0].payload["event_type"] == "updated"


@pytest.mark.unit
class TestRaiseForStatus:
    """Tests for the _raise_for_status method."""

    @pytest.mark.asyncio
    async def test_401_raises_permanent_error(self, adapter):
        """Test 401 response raises permanent PollAdapterError."""
        response = Mock(spec=httpx.Response)
        response.status_code = 401
        response.text = "Unauthorized"

        with pytest.raises(PollAdapterError, match="authentication error") as exc_info:
            await adapter._raise_for_status(response, "test operation")

        assert exc_info.value.permanent is True

    @pytest.mark.asyncio
    async def test_403_raises_permanent_error(self, adapter):
        """Test 403 response raises permanent PollAdapterError."""
        response = Mock(spec=httpx.Response)
        response.status_code = 403
        response.text = "Forbidden"

        with pytest.raises(PollAdapterError, match="authentication error") as exc_info:
            await adapter._raise_for_status(response, "test operation")

        assert exc_info.value.permanent is True

    @pytest.mark.asyncio
    async def test_429_raises_with_backoff(self, adapter):
        """Test 429 response raises PollAdapterError with backoff hint."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.text = "Rate limited"

        with pytest.raises(PollAdapterError, match="rate limited") as exc_info:
            await adapter._raise_for_status(response, "test operation")

        assert exc_info.value.backoff_seconds == 60

    @pytest.mark.asyncio
    async def test_500_raises_error(self, adapter):
        """Test 500 response raises PollAdapterError."""
        response = Mock(spec=httpx.Response)
        response.status_code = 500
        response.text = "Internal error"

        with pytest.raises(PollAdapterError, match="API error"):
            await adapter._raise_for_status(response, "test operation")

    @pytest.mark.asyncio
    async def test_200_no_raise(self, adapter):
        """Test 200 response does not raise."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200

        await adapter._raise_for_status(response, "test operation")  # Should not raise


@pytest.mark.unit
class TestResolveCalendarId:
    """Tests for the _resolve_calendar_id method."""

    def test_resolve_calendar_id_from_config(self, adapter, mock_poll_context):
        """Test calendar_id is resolved from config."""
        mock_poll_context.subscription.provider_config = {"calendar_id": "custom@group.calendar.google.com"}

        result = adapter._resolve_calendar_id(mock_poll_context)
        assert result == "custom@group.calendar.google.com"

    def test_resolve_calendar_id_defaults_to_primary(self, adapter, mock_poll_context):
        """Test calendar_id defaults to 'primary' when not in config."""
        mock_poll_context.subscription.provider_config = {}

        result = adapter._resolve_calendar_id(mock_poll_context)
        assert result == "primary"

    def test_resolve_calendar_id_handles_empty_string(self, adapter, mock_poll_context):
        """Test empty string calendar_id defaults to 'primary'."""
        mock_poll_context.subscription.provider_config = {"calendar_id": "  "}

        result = adapter._resolve_calendar_id(mock_poll_context)
        assert result == "primary"


@pytest.mark.unit
class TestResolveMaxResults:
    """Tests for the _resolve_max_results method."""

    def test_resolve_max_results_from_config(self, adapter, mock_poll_context):
        """Test max_results is resolved from config."""
        mock_poll_context.subscription.provider_config = {"max_results": 30}

        result = adapter._resolve_max_results(mock_poll_context)
        assert result == 30

    def test_resolve_max_results_bounded_upper(self, adapter, mock_poll_context):
        """Test max_results is bounded at upper limit."""
        mock_poll_context.subscription.provider_config = {"max_results": 100}

        result = adapter._resolve_max_results(mock_poll_context)
        assert result == 50  # MAX_EVENTS_PER_POLL

    def test_resolve_max_results_bounded_lower(self, adapter, mock_poll_context):
        """Test max_results is bounded at lower limit."""
        mock_poll_context.subscription.provider_config = {"max_results": 0}

        result = adapter._resolve_max_results(mock_poll_context)
        assert result == 1

    def test_resolve_max_results_handles_invalid(self, adapter, mock_poll_context):
        """Test invalid max_results uses default."""
        mock_poll_context.subscription.provider_config = {"max_results": "invalid"}

        result = adapter._resolve_max_results(mock_poll_context)
        assert result == 50  # Default MAX_EVENTS_PER_POLL
