"""Unit tests for AirtableNewRecordInViewAdapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.core.triggers.polling.adapters.airtable_new_record_in_view import (
    AirtableNewRecordInViewAdapter,
    _parse_created_time,
    MAX_SEEN_IDS_LIMIT,
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
        "base_id": "appTestBase123",
        "table_id": "tblTestTable456",
        "view_id": "viwTestView789",
        "max_records": 50,
    }

    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)

    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token="test_airtable_token",
    )


@pytest.fixture
def adapter():
    """Create an AirtableNewRecordInViewAdapter instance."""
    return AirtableNewRecordInViewAdapter()


@pytest.fixture
def sample_airtable_records():
    """Sample Airtable API response records."""
    return [
        {
            "id": "recABC123",
            "createdTime": "2026-02-27T10:00:00.000Z",
            "fields": {
                "Name": "Task 1",
                "Status": "To Do",
                "Priority": "High",
            },
        },
        {
            "id": "recDEF456",
            "createdTime": "2026-02-27T11:00:00.000Z",
            "fields": {
                "Name": "Task 2",
                "Status": "In Progress",
                "Priority": "Medium",
            },
        },
        {
            "id": "recGHI789",
            "createdTime": "2026-02-27T12:00:00.000Z",
            "fields": {
                "Name": "Task 3",
                "Status": "Done",
                "Priority": "Low",
            },
        },
    ]


@pytest.mark.unit
class TestParseCreatedTime:
    """Test the _parse_created_time helper function."""

    def test_parse_valid_timestamp(self):
        """Test parsing valid Airtable timestamp."""
        result = _parse_created_time("2026-02-27T10:00:00.000Z")
        assert result is not None
        assert result.year == 2026
        assert result.month == 2
        assert result.day == 27

    def test_parse_none(self):
        """Test parsing None value."""
        assert _parse_created_time(None) is None

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        assert _parse_created_time("") is None

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns None."""
        assert _parse_created_time("not-a-date") is None


@pytest.mark.unit
class TestBootstrapCursor:
    """Test the bootstrap_cursor method."""

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_captures_existing_ids(self, adapter, mock_poll_context, sample_airtable_records):
        """Test bootstrap_cursor captures all existing record IDs."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": sample_airtable_records}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert "seen_record_ids" in cursor
        assert "last_poll_time" in cursor
        assert set(cursor["seen_record_ids"]) == {"recABC123", "recDEF456", "recGHI789"}

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_empty_view(self, adapter, mock_poll_context):
        """Test bootstrap_cursor with empty view."""
        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": []}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["seen_record_ids"] == []


@pytest.mark.unit
class TestPoll:
    """Test the poll method."""

    @pytest.mark.asyncio
    async def test_poll_detects_new_records(self, adapter, mock_poll_context, sample_airtable_records):
        """Test poll detects newly added records."""
        # Cursor has seen first two records
        cursor = {
            "seen_record_ids": ["recABC123", "recDEF456"],
            "last_poll_time": "2026-02-27T09:00:00Z",
        }

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": sample_airtable_records}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        # Should detect recGHI789 as new
        assert len(result.events) == 1
        assert result.events[0].provider_event_id == "recGHI789"
        assert result.events[0].payload["record_id"] == "recGHI789"
        assert result.events[0].payload["fields"]["Name"] == "Task 3"

    @pytest.mark.asyncio
    async def test_poll_no_new_records(self, adapter, mock_poll_context, sample_airtable_records):
        """Test poll returns no events when all records already seen."""
        # Cursor has seen all records
        cursor = {
            "seen_record_ids": ["recABC123", "recDEF456", "recGHI789"],
            "last_poll_time": "2026-02-27T09:00:00Z",
        }

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": sample_airtable_records}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0

    @pytest.mark.asyncio
    async def test_poll_updates_cursor_with_all_ids(self, adapter, mock_poll_context, sample_airtable_records):
        """Test poll updates cursor with union of seen and current IDs."""
        # Old cursor has recOLD which is no longer in view
        cursor = {
            "seen_record_ids": ["recABC123", "recOLD999"],
            "last_poll_time": "2026-02-27T09:00:00Z",
        }

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": sample_airtable_records}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        # Cursor should include old + current IDs
        expected_ids = {"recABC123", "recOLD999", "recDEF456", "recGHI789"}
        assert set(result.cursor["seen_record_ids"]) == expected_ids

    @pytest.mark.asyncio
    async def test_poll_passes_correct_params(self, adapter, mock_poll_context):
        """Test poll passes correct parameters to Airtable API."""
        cursor = {"seen_record_ids": [], "last_poll_time": "2026-02-27T09:00:00Z"}

        captured_params = {}

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": []}

        async def mock_get(url, headers=None, params=None):
            captured_params["url"] = url
            captured_params["headers"] = headers
            captured_params["params"] = params
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            await adapter.poll(mock_poll_context, cursor)

        assert "appTestBase123" in captured_params["url"]
        assert "tblTestTable456" in captured_params["url"]
        assert captured_params["params"]["view"] == "viwTestView789"
        assert captured_params["params"]["maxRecords"] == 50
        assert "Bearer test_airtable_token" in captured_params["headers"]["Authorization"]


@pytest.mark.unit
class TestNormalizeRecord:
    """Test the _normalize_record method."""

    def test_normalize_record_complete(self, adapter):
        """Test normalization with complete record."""
        record = {
            "id": "recABC123",
            "createdTime": "2026-02-27T10:00:00.000Z",
            "fields": {
                "Name": "Test Task",
                "Status": "Done",
            },
        }
        config = {
            "base_id": "appBase123",
            "table_id": "tblTable456",
            "view_id": "viwView789",
        }

        result = adapter._normalize_record(record, config)

        assert result["record_id"] == "recABC123"
        assert result["created_time"] == "2026-02-27T10:00:00.000Z"
        assert result["fields"]["Name"] == "Test Task"
        assert result["base_id"] == "appBase123"
        assert result["table_id"] == "tblTable456"
        assert result["view_id"] == "viwView789"

    def test_normalize_record_empty_fields(self, adapter):
        """Test normalization with empty fields."""
        record = {
            "id": "recABC123",
            "createdTime": "2026-02-27T10:00:00.000Z",
            "fields": {},
        }
        config = {"base_id": "appBase123", "table_id": "tblTable456", "view_id": "viwView789"}

        result = adapter._normalize_record(record, config)

        assert result["record_id"] == "recABC123"
        assert result["fields"] == {}


@pytest.mark.unit
class TestRaiseForStatus:
    """Test the _raise_for_status method."""

    @pytest.mark.asyncio
    async def test_raise_for_status_200_no_raise(self, adapter):
        """Test _raise_for_status does not raise on 200 response."""
        response = Mock()
        response.status_code = 200

        # Should not raise
        await adapter._raise_for_status(response)

    @pytest.mark.asyncio
    async def test_raise_for_status_401(self, adapter):
        """Test _raise_for_status raises permanent error on 401."""
        response = Mock()
        response.status_code = 401
        response.text = "Unauthorized"

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter._raise_for_status(response)

        assert exc_info.value.permanent is True
        assert "authentication" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raise_for_status_403(self, adapter):
        """Test _raise_for_status raises permanent error on 403."""
        response = Mock()
        response.status_code = 403
        response.text = "Forbidden"

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter._raise_for_status(response)

        assert exc_info.value.permanent is True

    @pytest.mark.asyncio
    async def test_raise_for_status_404(self, adapter):
        """Test _raise_for_status raises permanent error on 404."""
        response = Mock()
        response.status_code = 404
        response.text = "Not found"

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter._raise_for_status(response)

        assert exc_info.value.permanent is True
        assert "not found" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raise_for_status_429(self, adapter):
        """Test _raise_for_status raises with backoff on 429."""
        response = Mock()
        response.status_code = 429
        response.text = "Rate limited"

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter._raise_for_status(response)

        assert exc_info.value.backoff_seconds == 60
        assert "rate limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_raise_for_status_500(self, adapter):
        """Test _raise_for_status raises generic error on 500."""
        response = Mock()
        response.status_code = 500
        response.text = "Server error"

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter._raise_for_status(response)

        assert exc_info.value.permanent is False
        assert exc_info.value.backoff_seconds is None


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_poll_missing_config(self, adapter):
        """Test poll raises error when config is missing required fields."""
        subscription = Mock(spec=TriggerSubscription)
        subscription.provider_config = {"base_id": "appBase123"}  # Missing table_id and view_id

        ctx = PollContext(
            subscription=subscription,
            user=Mock(spec=User),
            connection=Mock(spec=OAuthConnection),
            access_token="test_token",
        )

        cursor = {"seen_record_ids": [], "last_poll_time": "2026-02-27T09:00:00Z"}

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter.poll(ctx, cursor)

        assert exc_info.value.permanent is True
        assert "missing" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_poll_missing_access_token(self, adapter, mock_poll_context):
        """Test poll raises error when access token is missing."""
        mock_poll_context.access_token = None

        cursor = {"seen_record_ids": [], "last_poll_time": "2026-02-27T09:00:00Z"}

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter.poll(mock_poll_context, cursor)

        assert exc_info.value.permanent is True
        assert "token" in str(exc_info.value).lower()


@pytest.mark.unit
class TestCursorSizeLimit:
    """Test cursor size limiting to prevent unbounded growth."""

    @pytest.mark.asyncio
    async def test_cursor_resets_when_exceeds_limit(self, adapter, mock_poll_context):
        """Test cursor resets to current IDs when seen_ids exceeds limit."""
        # Create cursor with many old IDs
        old_ids = [f"recOLD{i:05d}" for i in range(MAX_SEEN_IDS_LIMIT)]
        cursor = {
            "seen_record_ids": old_ids,
            "last_poll_time": "2026-02-27T09:00:00Z",
        }

        # Current view has only 3 records
        current_records = [
            {"id": "recNEW001", "createdTime": "2026-02-27T10:00:00.000Z", "fields": {}},
            {"id": "recNEW002", "createdTime": "2026-02-27T11:00:00.000Z", "fields": {}},
            {"id": "recNEW003", "createdTime": "2026-02-27T12:00:00.000Z", "fields": {}},
        ]

        response = Mock(spec=httpx.Response)
        response.status_code = 200
        response.json.return_value = {"records": current_records}

        async def mock_get(*_args, **_kwargs):
            return response

        mock_client = AsyncMock()
        mock_client.get = mock_get
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock()

        with patch(
            "seer.core.triggers.polling.adapters.airtable_new_record_in_view.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        # Cursor should be reset to only current IDs
        assert len(result.cursor["seen_record_ids"]) == 3
        assert set(result.cursor["seen_record_ids"]) == {"recNEW001", "recNEW002", "recNEW003"}


@pytest.mark.unit
class TestTriggerKey:
    """Test trigger key configuration."""

    def test_trigger_key(self, adapter):
        """Test adapter has correct trigger key."""
        assert adapter.trigger_key == "poll.airtable.new_record_in_view"
