"""Unit tests for GoogleSheetsRowAddedAdapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest
import httpx

from seer.core.triggers.polling.adapters.google_sheets_row_added import (
    GoogleSheetsRowAddedAdapter,
    DEFAULT_MAX_NEW_ROWS,
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
        "spreadsheet_id": "test_spreadsheet_123",
        "sheet_name": "Sheet1",
        "has_header_row": True,
    }

    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)

    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token="test_google_token",
    )


@pytest.fixture
def adapter():
    """Create a GoogleSheetsRowAddedAdapter instance."""
    return GoogleSheetsRowAddedAdapter()


@pytest.fixture
def sample_sheets_response():
    """Sample Google Sheets API response with header and data rows."""
    return {
        "range": "Sheet1!A1:D4",
        "majorDimension": "ROWS",
        "values": [
            ["Name", "Email", "Status", "Date"],
            ["Alice", "alice@example.com", "Active", "2026-03-01"],
            ["Bob", "bob@example.com", "Pending", "2026-03-02"],
            ["Charlie", "charlie@example.com", "Active", "2026-03-03"],
        ],
    }


def _mock_http_client(response_data, status_code=200):
    """Create a mock httpx.AsyncClient returning the given data."""
    response = Mock(spec=httpx.Response)
    response.status_code = status_code
    response.json.return_value = response_data
    response.text = str(response_data)

    async def mock_get(*_args, **_kwargs):
        return response

    mock_client = AsyncMock()
    mock_client.get = mock_get
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()
    return mock_client


@pytest.mark.unit
class TestBootstrapCursor:
    """Test the bootstrap_cursor method."""

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_captures_row_count(self, adapter, mock_poll_context, sample_sheets_response):
        """Test bootstrap captures total row count including header."""
        mock_client = _mock_http_client(sample_sheets_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["last_row_count"] == 4  # 1 header + 3 data rows
        assert "last_poll_time" in cursor

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_with_header_row(self, adapter, mock_poll_context, sample_sheets_response):
        """Test bootstrap captures headers when has_header_row is True."""
        mock_client = _mock_http_client(sample_sheets_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["headers"] == ["Name", "Email", "Status", "Date"]

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_without_header_row(self, adapter):
        """Test bootstrap does not capture headers when has_header_row is False."""
        subscription = Mock(spec=TriggerSubscription)
        subscription.provider_config = {
            "spreadsheet_id": "test123",
            "sheet_name": "Sheet1",
            "has_header_row": False,
        }
        ctx = PollContext(
            subscription=subscription,
            user=Mock(spec=User),
            connection=Mock(spec=OAuthConnection),
            access_token="test_token",
        )

        response_data = {"values": [["Row1Col1", "Row1Col2"], ["Row2Col1", "Row2Col2"]]}
        mock_client = _mock_http_client(response_data)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(ctx)

        assert cursor["last_row_count"] == 2
        assert cursor["headers"] is None

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_empty_sheet(self, adapter, mock_poll_context):
        """Test bootstrap with empty sheet."""
        mock_client = _mock_http_client({"values": []})

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["last_row_count"] == 0
        assert cursor["headers"] is None

    @pytest.mark.asyncio
    async def test_bootstrap_cursor_no_values_key(self, adapter, mock_poll_context):
        """Test bootstrap when API returns no values key."""
        mock_client = _mock_http_client({})

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

        assert cursor["last_row_count"] == 0


@pytest.mark.unit
class TestPoll:
    """Test the poll method."""

    @pytest.mark.asyncio
    async def test_poll_detects_new_rows(self, adapter, mock_poll_context):
        """Test poll detects newly appended rows."""
        cursor = {
            "last_row_count": 4,
            "headers": ["Name", "Email", "Status", "Date"],
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {
            "values": [
                ["Dave", "dave@example.com", "Active", "2026-03-04"],
                ["Eve", "eve@example.com", "Pending", "2026-03-05"],
            ]
        }
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 2
        assert result.events[0].payload["row_number"] == 5
        assert result.events[0].payload["fields"]["Name"] == "Dave"
        assert result.events[1].payload["row_number"] == 6
        assert result.events[1].payload["fields"]["Email"] == "eve@example.com"

    @pytest.mark.asyncio
    async def test_poll_no_new_rows(self, adapter, mock_poll_context):
        """Test poll returns no events when no new rows."""
        cursor = {
            "last_row_count": 4,
            "headers": ["Name", "Email"],
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        mock_client = _mock_http_client({})

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert len(result.events) == 0
        assert result.cursor == cursor

    @pytest.mark.asyncio
    async def test_poll_with_header_normalization(self, adapter, mock_poll_context):
        """Test poll normalizes rows using header names."""
        cursor = {
            "last_row_count": 1,
            "headers": ["Name", "Email"],
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {"values": [["Alice", "alice@example.com"]]}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events[0].payload["fields"] == {
            "Name": "Alice",
            "Email": "alice@example.com",
        }
        assert result.events[0].payload["row_values"] is None

    @pytest.mark.asyncio
    async def test_poll_without_header_row(self, adapter, mock_poll_context):
        """Test poll returns row_values when no headers."""
        cursor = {
            "last_row_count": 2,
            "headers": None,
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {"values": [["Val1", "Val2", "Val3"]]}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events[0].payload["row_values"] == ["Val1", "Val2", "Val3"]
        assert result.events[0].payload["fields"] is None

    @pytest.mark.asyncio
    async def test_poll_updates_cursor(self, adapter, mock_poll_context):
        """Test poll updates cursor with new row count."""
        cursor = {
            "last_row_count": 4,
            "headers": ["Name"],
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {"values": [["New1"], ["New2"], ["New3"]]}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.cursor["last_row_count"] == 7
        assert result.cursor["headers"] == ["Name"]

    @pytest.mark.asyncio
    async def test_poll_has_more_flag(self, adapter, mock_poll_context):
        """Test has_more is True when max rows returned."""
        cursor = {
            "last_row_count": 0,
            "headers": None,
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        # Return exactly DEFAULT_MAX_NEW_ROWS rows
        rows = [[f"val_{i}"] for i in range(DEFAULT_MAX_NEW_ROWS)]
        new_rows_response = {"values": rows}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.has_more is True
        assert len(result.events) == DEFAULT_MAX_NEW_ROWS

    @pytest.mark.asyncio
    async def test_poll_no_more_when_fewer_rows(self, adapter, mock_poll_context):
        """Test has_more is False when fewer than max rows returned."""
        cursor = {
            "last_row_count": 4,
            "headers": ["Name"],
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {"values": [["New1"]]}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.has_more is False

    @pytest.mark.asyncio
    async def test_poll_provider_event_id_format(self, adapter, mock_poll_context):
        """Test provider_event_id uses correct format for deduplication."""
        cursor = {
            "last_row_count": 3,
            "headers": None,
            "last_poll_time": "2026-03-01T00:00:00Z",
        }

        new_rows_response = {"values": [["val1"]]}
        mock_client = _mock_http_client(new_rows_response)

        with patch(
            "seer.core.triggers.polling.adapters.google_sheets_row_added.httpx.AsyncClient",
            return_value=mock_client,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

        assert result.events[0].provider_event_id == "test_spreadsheet_123:Sheet1:4"


@pytest.mark.unit
class TestNormalizeRow:
    """Test the _normalize_row method."""

    def test_normalize_with_headers(self, adapter):
        """Test normalization with column headers."""
        row = ["Alice", "alice@example.com", "Active"]
        config = {"spreadsheet_id": "test123", "sheet_name": "Sheet1"}
        headers = ["Name", "Email", "Status"]

        result = adapter._normalize_row(row, 5, headers, config)

        assert result["fields"] == {"Name": "Alice", "Email": "alice@example.com", "Status": "Active"}
        assert result["row_values"] is None
        assert result["row_number"] == 5
        assert result["spreadsheet_id"] == "test123"
        assert result["sheet_name"] == "Sheet1"

    def test_normalize_short_row(self, adapter):
        """Test normalization when row has fewer values than headers."""
        row = ["Alice"]
        config = {"spreadsheet_id": "test123", "sheet_name": "Sheet1"}
        headers = ["Name", "Email", "Status"]

        result = adapter._normalize_row(row, 5, headers, config)

        assert result["fields"] == {"Name": "Alice", "Email": None, "Status": None}

    def test_normalize_row_with_more_columns_than_headers(self, adapter):
        """Test normalization when row has more values than headers (overflow columns get generated names)."""
        row = ["Alice", "alice@example.com", "Active", "extra_value"]
        config = {"spreadsheet_id": "test123", "sheet_name": "Sheet1"}
        headers = ["Name", "Email"]

        result = adapter._normalize_row(row, 5, headers, config)

        assert result["fields"] == {
            "Name": "Alice",
            "Email": "alice@example.com",
            "column_3": "Active",
            "column_4": "extra_value",
        }

    def test_normalize_without_headers(self, adapter):
        """Test normalization without column headers."""
        row = ["Val1", "Val2", "Val3"]
        config = {"spreadsheet_id": "test123", "sheet_name": "Sheet1"}

        result = adapter._normalize_row(row, 3, None, config)

        assert result["row_values"] == ["Val1", "Val2", "Val3"]
        assert result["fields"] is None
        assert result["row_number"] == 3


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling scenarios."""

    @pytest.mark.asyncio
    async def test_poll_missing_spreadsheet_id(self, adapter):
        """Test poll raises error when spreadsheet_id is missing."""
        subscription = Mock(spec=TriggerSubscription)
        subscription.provider_config = {"sheet_name": "Sheet1"}

        ctx = PollContext(
            subscription=subscription,
            user=Mock(spec=User),
            connection=Mock(spec=OAuthConnection),
            access_token="test_token",
        )

        cursor = {"last_row_count": 0, "headers": None, "last_poll_time": "2026-03-01T00:00:00Z"}

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter.poll(ctx, cursor)

        assert exc_info.value.permanent is True
        assert "spreadsheet_id" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_poll_missing_access_token(self, adapter, mock_poll_context):
        """Test poll raises error when access token is missing."""
        mock_poll_context.access_token = None

        cursor = {"last_row_count": 0, "headers": None, "last_poll_time": "2026-03-01T00:00:00Z"}

        with pytest.raises(PollAdapterError) as exc_info:
            await adapter.poll(mock_poll_context, cursor)

        assert exc_info.value.permanent is True
        assert "token" in str(exc_info.value).lower()


@pytest.mark.unit
class TestRaiseForStatus:
    """Test the _raise_for_status method."""

    @pytest.mark.asyncio
    async def test_raise_for_status_200_no_raise(self, adapter):
        """Test _raise_for_status does not raise on 200 response."""
        response = Mock()
        response.status_code = 200
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
class TestTriggerKey:
    """Test trigger key configuration."""

    def test_trigger_key(self, adapter):
        """Test adapter has correct trigger key."""
        assert adapter.trigger_key == "poll.google_sheets.row_added"
