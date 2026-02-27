"""Unit tests for Google Calendar tools."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.tools.google.gcalendar import (
    GoogleCalendarListEventsTool,
    GoogleCalendarGetEventTool,
    GoogleCalendarCreateEventTool,
    GoogleCalendarUpdateEventTool,
    GoogleCalendarDeleteEventTool,
)


@pytest.fixture
def mock_access_token():
    return "test_access_token"


@pytest.fixture
def list_events_tool():
    return GoogleCalendarListEventsTool()


@pytest.fixture
def get_event_tool():
    return GoogleCalendarGetEventTool()


@pytest.fixture
def create_event_tool():
    return GoogleCalendarCreateEventTool()


@pytest.fixture
def update_event_tool():
    return GoogleCalendarUpdateEventTool()


@pytest.fixture
def delete_event_tool():
    return GoogleCalendarDeleteEventTool()


@pytest.mark.unit
class TestGoogleCalendarListEventsTool:
    """Tests for GoogleCalendarListEventsTool."""

    @pytest.mark.asyncio
    async def test_list_events_returns_events(self, list_events_tool, mock_access_token):
        """Test listing events returns events from the API."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "items": [
                {
                    "id": "event1",
                    "summary": "Meeting",
                    "start": {"dateTime": "2026-01-15T10:00:00Z"},
                    "end": {"dateTime": "2026-01-15T11:00:00Z"},
                }
            ],
            "nextSyncToken": "abc123",
        }

        with patch.object(
            list_events_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await list_events_tool.execute(
                mock_access_token,
                {"calendar_id": "primary", "max_results": 10},
            )

        assert "items" in result
        assert len(result["items"]) == 1
        assert result["items"][0]["id"] == "event1"
        assert result["items"][0]["summary"] == "Meeting"

    @pytest.mark.asyncio
    async def test_list_events_with_time_range(self, list_events_tool, mock_access_token):
        """Test listing events with time range parameters."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": [], "nextSyncToken": "xyz"}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            list_events_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await list_events_tool.execute(
                mock_access_token,
                {
                    "calendar_id": "primary",
                    "time_min": "2026-01-01T00:00:00Z",
                    "time_max": "2026-12-31T23:59:59Z",
                },
            )

        assert captured_params.get("timeMin") == "2026-01-01T00:00:00Z"
        assert captured_params.get("timeMax") == "2026-12-31T23:59:59Z"

    @pytest.mark.asyncio
    async def test_list_events_with_search_query(self, list_events_tool, mock_access_token):
        """Test listing events with search query."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            list_events_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await list_events_tool.execute(
                mock_access_token,
                {"calendar_id": "primary", "q": "team meeting"},
            )

        assert captured_params.get("q") == "team meeting"

    @pytest.mark.asyncio
    async def test_list_events_max_results_bounded(self, list_events_tool, mock_access_token):
        """Test max_results is bounded between 1 and 250."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"items": []}

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}))
            return mock_response

        with patch.object(
            list_events_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            # Test upper bound
            await list_events_tool.execute(
                mock_access_token,
                {"calendar_id": "primary", "max_results": 500},
            )
            assert captured_params.get("maxResults") == 250

    def test_tool_metadata(self, list_events_tool):
        """Test tool has correct metadata."""
        assert list_events_tool.name == "google_calendar_list_events"
        assert list_events_tool.integration_type == "google_calendar"
        assert "calendar.readonly" in list_events_tool.required_scopes[0]


@pytest.mark.unit
class TestGoogleCalendarGetEventTool:
    """Tests for GoogleCalendarGetEventTool."""

    @pytest.mark.asyncio
    async def test_get_event_by_id(self, get_event_tool, mock_access_token):
        """Test getting a single event by ID."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "id": "event123",
            "summary": "Important Meeting",
            "description": "Discuss project timeline",
            "start": {"dateTime": "2026-01-15T14:00:00Z"},
            "end": {"dateTime": "2026-01-15T15:00:00Z"},
        }

        with patch.object(
            get_event_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await get_event_tool.execute(
                mock_access_token,
                {"calendar_id": "primary", "event_id": "event123"},
            )

        assert result["id"] == "event123"
        assert result["summary"] == "Important Meeting"

    @pytest.mark.asyncio
    async def test_get_event_requires_event_id(self, get_event_tool, mock_access_token):
        """Test that event_id is required."""
        with pytest.raises(ValueError, match="event_id is required"):
            await get_event_tool.execute(
                mock_access_token,
                {"calendar_id": "primary"},
            )


@pytest.mark.unit
class TestGoogleCalendarCreateEventTool:
    """Tests for GoogleCalendarCreateEventTool."""

    @pytest.mark.asyncio
    async def test_create_event_minimal(self, create_event_tool, mock_access_token):
        """Test creating an event with minimal required fields."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "id": "new_event_id",
            "summary": "New Meeting",
            "htmlLink": "https://calendar.google.com/event?eid=xxx",
        }

        with patch.object(
            create_event_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await create_event_tool.execute(
                mock_access_token,
                {"summary": "New Meeting"},
            )

        assert result["id"] == "new_event_id"
        assert result["summary"] == "New Meeting"

    @pytest.mark.asyncio
    async def test_create_event_with_full_details(self, create_event_tool, mock_access_token):
        """Test creating an event with all details."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"id": "full_event"}

        captured_body = {}

        async def capture_request(*args, **kwargs):
            captured_body.update(kwargs.get("json_body", {}))
            return mock_response

        with patch.object(
            create_event_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await create_event_tool.execute(
                mock_access_token,
                {
                    "summary": "Team Sync",
                    "description": "Weekly team sync meeting",
                    "location": "Conference Room B",
                    "start_datetime": "2026-01-20T09:00:00-05:00",
                    "end_datetime": "2026-01-20T10:00:00-05:00",
                    "timezone": "America/New_York",
                    "attendees": [{"email": "team@example.com"}],
                },
            )

        assert captured_body["summary"] == "Team Sync"
        assert captured_body["description"] == "Weekly team sync meeting"
        assert captured_body["location"] == "Conference Room B"
        assert captured_body["start"]["dateTime"] == "2026-01-20T09:00:00-05:00"
        assert captured_body["attendees"] == [{"email": "team@example.com"}]

    @pytest.mark.asyncio
    async def test_create_event_all_day(self, create_event_tool, mock_access_token):
        """Test creating an all-day event."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"id": "allday_event"}

        captured_body = {}

        async def capture_request(*args, **kwargs):
            captured_body.update(kwargs.get("json_body", {}))
            return mock_response

        with patch.object(
            create_event_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await create_event_tool.execute(
                mock_access_token,
                {
                    "summary": "Company Holiday",
                    "start_date": "2026-01-01",
                    "end_date": "2026-01-02",
                },
            )

        assert captured_body["start"]["date"] == "2026-01-01"
        assert captured_body["end"]["date"] == "2026-01-02"
        assert "dateTime" not in captured_body["start"]

    @pytest.mark.asyncio
    async def test_create_event_requires_summary(self, create_event_tool, mock_access_token):
        """Test that summary is required."""
        with pytest.raises(ValueError, match="summary is required"):
            await create_event_tool.execute(
                mock_access_token,
                {"description": "No title provided"},
            )

    def test_tool_requires_events_scope(self, create_event_tool):
        """Test that create tool requires calendar.events scope."""
        assert "calendar.events" in create_event_tool.required_scopes[0]


@pytest.mark.unit
class TestGoogleCalendarUpdateEventTool:
    """Tests for GoogleCalendarUpdateEventTool."""

    @pytest.mark.asyncio
    async def test_update_event(self, update_event_tool, mock_access_token):
        """Test updating an existing event."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {
            "id": "event123",
            "summary": "Updated Meeting Title",
        }

        with patch.object(
            update_event_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await update_event_tool.execute(
                mock_access_token,
                {"event_id": "event123", "summary": "Updated Meeting Title"},
            )

        assert result["summary"] == "Updated Meeting Title"

    @pytest.mark.asyncio
    async def test_update_event_partial(self, update_event_tool, mock_access_token):
        """Test partial update only sends provided fields."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.is_error = False
        mock_response.json.return_value = {"id": "event123"}

        captured_body = {}

        async def capture_request(*args, **kwargs):
            captured_body.update(kwargs.get("json_body", {}))
            return mock_response

        with patch.object(
            update_event_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await update_event_tool.execute(
                mock_access_token,
                {"event_id": "event123", "location": "New Location"},
            )

        assert captured_body == {"location": "New Location"}

    @pytest.mark.asyncio
    async def test_update_event_requires_event_id(self, update_event_tool, mock_access_token):
        """Test that event_id is required."""
        with pytest.raises(ValueError, match="event_id is required"):
            await update_event_tool.execute(
                mock_access_token,
                {"summary": "New Title"},
            )


@pytest.mark.unit
class TestGoogleCalendarDeleteEventTool:
    """Tests for GoogleCalendarDeleteEventTool."""

    @pytest.mark.asyncio
    async def test_delete_event(self, delete_event_tool, mock_access_token):
        """Test deleting an event."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.is_error = False

        with patch.object(
            delete_event_tool, "_execute_request_with_retry", return_value=mock_response
        ):
            result = await delete_event_tool.execute(
                mock_access_token,
                {"event_id": "event_to_delete"},
            )

        assert result["success"] is True
        assert result["event_id"] == "event_to_delete"

    @pytest.mark.asyncio
    async def test_delete_event_requires_event_id(self, delete_event_tool, mock_access_token):
        """Test that event_id is required."""
        with pytest.raises(ValueError, match="event_id is required"):
            await delete_event_tool.execute(
                mock_access_token,
                {"calendar_id": "primary"},
            )

    @pytest.mark.asyncio
    async def test_delete_event_with_notifications(self, delete_event_tool, mock_access_token):
        """Test delete event sends notification parameter."""
        mock_response = Mock(spec=httpx.Response)
        mock_response.status_code = 204
        mock_response.is_error = False

        captured_params = {}

        async def capture_request(*args, **kwargs):
            captured_params.update(kwargs.get("params", {}) or {})
            return mock_response

        with patch.object(
            delete_event_tool, "_execute_request_with_retry", side_effect=capture_request
        ):
            await delete_event_tool.execute(
                mock_access_token,
                {"event_id": "event123", "send_updates": "all"},
            )

        assert captured_params.get("sendUpdates") == "all"
