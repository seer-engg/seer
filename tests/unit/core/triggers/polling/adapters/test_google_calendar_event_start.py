"""Unit tests for GoogleCalendarEventStartAdapter."""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.core.triggers.polling.adapters.google_calendar_event_start import (
    GoogleCalendarEventStartAdapter,
    _parse_rfc3339,
    _normalize_event,
    _get_event_start_utc,
    _make_instance_key,
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
        "offset_minutes": -15,
        "include_all_day_events": False,
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
    """Create a GoogleCalendarEventStartAdapter instance."""
    return GoogleCalendarEventStartAdapter()


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
class TestGetEventStartUtc:
    """Tests for the _get_event_start_utc helper function."""

    def test_datetime_event(self):
        """Test getting start time from a datetime event."""
        event = {"start": {"dateTime": "2026-01-15T10:00:00Z"}}
        result = _get_event_start_utc(event)
        assert result is not None
        assert result.hour == 10

    def test_all_day_event(self):
        """Test getting start time from an all-day event."""
        event = {"start": {"date": "2026-01-15"}}
        result = _get_event_start_utc(event)
        assert result is not None
        assert result.year == 2026
        assert result.month == 1
        assert result.day == 15
        assert result.hour == 0  # Midnight UTC

    def test_missing_start(self):
        """Test handling missing start time."""
        assert _get_event_start_utc({}) is None
        assert _get_event_start_utc({"start": {}}) is None


@pytest.mark.unit
class TestMakeInstanceKey:
    """Tests for the _make_instance_key helper function."""

    def test_makes_unique_key(self):
        """Test that instance key combines event ID and start time."""
        start = datetime(2026, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        key = _make_instance_key("event123", start)
        assert "event123" in key
        assert "2026-01-15" in key


@pytest.mark.unit
class TestNormalizeEvent:
    """Tests for the _normalize_event helper function."""

    def test_normalize_event_basic(self):
        """Test normalizing a basic event."""
        event = {
            "id": "event123",
            "summary": "Team Meeting",
            "description": "Weekly sync",
            "location": "Room A",
            "status": "confirmed",
            "htmlLink": "https://calendar.google.com/event?eid=xxx",
            "start": {"dateTime": "2026-01-15T14:00:00Z"},
            "end": {"dateTime": "2026-01-15T15:00:00Z"},
        }

        result = _normalize_event(event, "primary")

        assert result["event_id"] == "event123"
        assert result["calendar_id"] == "primary"
        assert result["summary"] == "Team Meeting"
        assert result["is_all_day"] is False

    def test_normalize_all_day_event(self):
        """Test normalizing an all-day event."""
        event = {
            "id": "allday123",
            "summary": "Holiday",
            "start": {"date": "2026-01-01"},
            "end": {"date": "2026-01-02"},
        }

        result = _normalize_event(event, "work_calendar")

        assert result["event_id"] == "allday123"
        assert result["is_all_day"] is True
        assert result["start"]["datetime"] == "2026-01-01"


@pytest.mark.unit
class TestBootstrapCursor:
    """Tests for the bootstrap_cursor method."""

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_initializes_empty(self, adapter, mock_poll_context):
        """Test bootstrap_cursor creates empty cursor with bootstrapped_at."""
        cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert "triggered_instances" in cursor
        assert cursor["triggered_instances"] == {}
        assert "bootstrapped_at" in cursor
        assert "last_cleanup_utc" in cursor


@pytest.mark.unit
class TestPoll:
    """Tests for the poll method."""

    @pytest.mark.asyncio
    async def test_poll_triggers_event_at_offset(self, adapter, mock_poll_context):
        """Test poll emits event when trigger time has passed."""
        now = datetime.now(timezone.utc)
        # Event starts 10 minutes from now, offset is -15 minutes
        # So trigger time was 5 minutes ago
        event_start = now + timedelta(minutes=10)

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        mock_poll_context.subscription.provider_config = {
            "calendar_id": "primary",
            "offset_minutes": -15,
            "include_all_day_events": False,
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "upcoming_event",
                    "summary": "Meeting Soon",
                    "status": "confirmed",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
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
        assert result.events[0].payload["event_id"] == "upcoming_event"
        assert result.events[0].payload["trigger_type"] == "before_start"
        assert result.events[0].payload["offset_minutes"] == -15

    @pytest.mark.asyncio
    async def test_poll_skips_already_triggered(self, adapter, mock_poll_context):
        """Test poll doesn't re-trigger events."""
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(minutes=10)
        instance_key = f"already_triggered_{event_start.isoformat()}"

        cursor = {
            "triggered_instances": {instance_key: now.isoformat()},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "already_triggered",
                    "summary": "Already triggered",
                    "status": "confirmed",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_poll_skips_cancelled_events(self, adapter, mock_poll_context):
        """Test that cancelled events are skipped."""
        now = datetime.now(timezone.utc)
        event_start = now + timedelta(minutes=10)

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "cancelled_event",
                    "summary": "Cancelled",
                    "status": "cancelled",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_poll_skips_all_day_by_default(self, adapter, mock_poll_context):
        """Test that all-day events are skipped by default."""
        now = datetime.now(timezone.utc)

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        mock_poll_context.subscription.provider_config = {
            "calendar_id": "primary",
            "offset_minutes": -15,
            "include_all_day_events": False,
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "all_day_event",
                    "summary": "All Day",
                    "status": "confirmed",
                    "start": {"date": "2026-01-15"},
                    "end": {"date": "2026-01-16"},
                }
            ],
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_poll_skips_events_before_bootstrap(self, adapter, mock_poll_context):
        """Test that events with trigger time before bootstrap are skipped."""
        now = datetime.now(timezone.utc)
        # Event would have triggered in the past (before bootstrap)
        event_start = now - timedelta(minutes=5)

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": now.isoformat(),  # Bootstrap at now
        }

        mock_poll_context.subscription.provider_config = {
            "calendar_id": "primary",
            "offset_minutes": -15,  # Trigger 15 min before = trigger_time was 20 min ago
            "include_all_day_events": False,
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "old_event",
                    "summary": "Old Event",
                    "status": "confirmed",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
        }

        async def mock_get(*args, **kwargs):
            return poll_response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch("httpx.AsyncClient", return_value=mock_client):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_poll_after_start_trigger_type(self, adapter, mock_poll_context):
        """Test trigger_type is 'after_start' for positive offset."""
        now = datetime.now(timezone.utc)
        # Event started 10 minutes ago, offset is +5 minutes
        # So trigger time was 5 minutes ago
        event_start = now - timedelta(minutes=10)

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        mock_poll_context.subscription.provider_config = {
            "calendar_id": "primary",
            "offset_minutes": 5,
            "include_all_day_events": False,
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "past_event",
                    "summary": "Past Event",
                    "status": "confirmed",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
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
        assert result.events[0].payload["trigger_type"] == "after_start"

    @pytest.mark.asyncio
    async def test_poll_at_start_trigger_type(self, adapter, mock_poll_context):
        """Test trigger_type is 'at_start' for zero offset."""
        now = datetime.now(timezone.utc)
        event_start = now - timedelta(minutes=5)  # Started 5 minutes ago

        cursor = {
            "triggered_instances": {},
            "last_cleanup_utc": now.isoformat(),
            "bootstrapped_at": (now - timedelta(hours=1)).isoformat(),
        }

        mock_poll_context.subscription.provider_config = {
            "calendar_id": "primary",
            "offset_minutes": 0,
            "include_all_day_events": False,
        }

        poll_response = Mock(spec=httpx.Response)
        poll_response.status_code = 200
        poll_response.json.return_value = {
            "items": [
                {
                    "id": "event_now",
                    "summary": "Event Now",
                    "status": "confirmed",
                    "start": {"dateTime": event_start.isoformat()},
                    "end": {"dateTime": (event_start + timedelta(hours=1)).isoformat()},
                }
            ],
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
        assert result.events[0].payload["trigger_type"] == "at_start"


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
    async def test_429_raises_with_backoff(self, adapter):
        """Test 429 response raises PollAdapterError with backoff hint."""
        response = Mock(spec=httpx.Response)
        response.status_code = 429
        response.text = "Rate limited"

        with pytest.raises(PollAdapterError, match="rate limited") as exc_info:
            await adapter._raise_for_status(response, "test operation")

        assert exc_info.value.backoff_seconds == 60

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


@pytest.mark.unit
class TestResolveOffsetMinutes:
    """Tests for the _resolve_offset_minutes method."""

    def test_resolve_offset_from_config(self, adapter, mock_poll_context):
        """Test offset_minutes is resolved from config."""
        mock_poll_context.subscription.provider_config = {"offset_minutes": -30}

        result = adapter._resolve_offset_minutes(mock_poll_context)
        assert result == -30

    def test_resolve_offset_bounded_upper(self, adapter, mock_poll_context):
        """Test offset_minutes is bounded at upper limit (24h)."""
        mock_poll_context.subscription.provider_config = {"offset_minutes": 2000}

        result = adapter._resolve_offset_minutes(mock_poll_context)
        assert result == 1440

    def test_resolve_offset_bounded_lower(self, adapter, mock_poll_context):
        """Test offset_minutes is bounded at lower limit (-24h)."""
        mock_poll_context.subscription.provider_config = {"offset_minutes": -2000}

        result = adapter._resolve_offset_minutes(mock_poll_context)
        assert result == -1440

    def test_resolve_offset_defaults_to_minus_15(self, adapter, mock_poll_context):
        """Test offset_minutes defaults to -15 when not in config."""
        mock_poll_context.subscription.provider_config = {}

        result = adapter._resolve_offset_minutes(mock_poll_context)
        assert result == -15


@pytest.mark.unit
class TestResolveIncludeAllDay:
    """Tests for the _resolve_include_all_day method."""

    def test_resolve_include_all_day_true(self, adapter, mock_poll_context):
        """Test include_all_day_events is resolved from config."""
        mock_poll_context.subscription.provider_config = {"include_all_day_events": True}

        result = adapter._resolve_include_all_day(mock_poll_context)
        assert result is True

    def test_resolve_include_all_day_defaults_false(self, adapter, mock_poll_context):
        """Test include_all_day_events defaults to False."""
        mock_poll_context.subscription.provider_config = {}

        result = adapter._resolve_include_all_day(mock_poll_context)
        assert result is False
