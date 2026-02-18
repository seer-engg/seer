"""Unit tests for SlackMessageReceivedAdapter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from seer.core.triggers.polling.adapters.slack_message_received import (
    SlackMessageReceivedAdapter,
    _parse_slack_timestamp,
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
        "channel_id": "C01234567890",
        "workspace_id": "T01234567890",
        "max_results": 50,
        "include_bot_messages": False,
        "only_app_mentions": False,
    }

    user = Mock(spec=User)
    connection = Mock(spec=OAuthConnection)

    return PollContext(
        subscription=subscription,
        user=user,
        connection=connection,
        access_token="xoxb-test-token-12345",
    )


@pytest.fixture
def adapter():
    """Create a SlackMessageReceivedAdapter instance."""
    return SlackMessageReceivedAdapter()


@pytest.mark.unit
class TestParseSlackTimestamp:
    """Test timestamp parsing utility."""

    def test_parse_valid_timestamp(self):
        """Test parsing valid Slack timestamp."""
        ts = "1735630123.456789"
        result = _parse_slack_timestamp(ts)
        assert result is not None
        assert result.year == 2024
        assert result.month == 12

    def test_parse_none_timestamp(self):
        """Test parsing None returns None."""
        assert _parse_slack_timestamp(None) is None

    def test_parse_empty_string(self):
        """Test parsing empty string returns None."""
        assert _parse_slack_timestamp("") is None

    def test_parse_invalid_timestamp(self):
        """Test parsing invalid timestamp returns None."""
        assert _parse_slack_timestamp("invalid") is None


@pytest.mark.unit
class TestBootstrapCursorWithExistingMessages:
    """Test bootstrap_cursor when channel has existing messages."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_single_message(self, adapter, mock_poll_context):
        """Test bootstrap_cursor returns most recent message ts when channel has one message."""
        mock_message = {
            "ts": "1735630123.456789",
            "text": "Test message",
            "user": "U12345678",
        }

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=[mock_message],
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            assert cursor == {"last_ts": "1735630123.456789"}

    @pytest.mark.asyncio
    async def test_bootstrap_calls_fetch_with_correct_params(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor calls _fetch_slack_messages with correct parameters."""
        mock_message = {"ts": "1735630123.456789", "text": "Test"}

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=[mock_message],
        ) as mock_fetch:
            await adapter.bootstrap_cursor(mock_poll_context)

            # Verify it was called with max_results=1 to fetch only most recent
            mock_fetch.assert_called_once_with(
                "xoxb-test-token-12345",  # access_token
                "C01234567890",  # channel_id from provider_config
                oldest_ts=None,  # oldest_ts (None for first fetch)
                max_results=1,  # max_results=1 to get only most recent
            )


@pytest.mark.unit
class TestBootstrapCursorWithEmptyChannel:
    """Test bootstrap_cursor when channel has no messages."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_empty_list(self, adapter, mock_poll_context):
        """Test bootstrap_cursor returns None when channel has no messages."""
        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=[],
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            assert cursor == {"last_ts": None}

    @pytest.mark.asyncio
    async def test_bootstrap_with_none_response(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles None response from API."""
        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=None,
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            assert cursor == {"last_ts": None}


@pytest.mark.unit
class TestBootstrapCursorErrorHandling:
    """Test bootstrap_cursor error handling and fallback behavior."""

    @pytest.mark.asyncio
    async def test_bootstrap_handles_poll_adapter_error(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor catches PollAdapterError and returns None cursor."""
        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            side_effect=PollAdapterError("Rate limited", backoff_seconds=60),
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            # Should fall back to None cursor on error
            assert cursor == {"last_ts": None}

    @pytest.mark.asyncio
    async def test_bootstrap_handles_auth_error(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles authentication errors gracefully."""
        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            side_effect=PollAdapterError(
                "Slack authentication error", permanent=True
            ),
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            # Should fall back to None cursor even on permanent errors during bootstrap
            assert cursor == {"last_ts": None}

    @pytest.mark.asyncio
    async def test_bootstrap_handles_missing_access_token(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor propagates PollAdapterError from _get_access_token."""
        mock_poll_context.access_token = None

        # _get_access_token raises error before we catch it in try/except
        # This should propagate up
        with pytest.raises(PollAdapterError, match="access token not available"):
            await adapter.bootstrap_cursor(mock_poll_context)

    @pytest.mark.asyncio
    async def test_bootstrap_handles_missing_channel_id(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor propagates error when channel_id is missing."""
        # Remove channel_id from config
        mock_poll_context.subscription.provider_config = {}

        with pytest.raises(PollAdapterError, match="channel_id is required"):
            await adapter.bootstrap_cursor(mock_poll_context)


@pytest.mark.unit
class TestPollFetchesNewMessages:
    """Test poll() fetches and processes new messages."""

    @pytest.mark.asyncio
    async def test_poll_with_new_messages(self, adapter, mock_poll_context):
        """Test poll() correctly processes new messages."""
        cursor = {"last_ts": "1735630100.000000"}
        new_messages = [
            {
                "ts": "1735630200.000000",
                "text": "New message 1",
                "user": "U12345678",
            },
            {
                "ts": "1735630300.000000",
                "text": "New message 2",
                "user": "U87654321",
            },
        ]

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=new_messages,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

            assert len(result.events) == 2
            assert result.events[0].provider_event_id == "1735630200.000000"
            assert result.events[1].provider_event_id == "1735630300.000000"
            # Cursor should advance to newest message
            assert result.cursor["last_ts"] == "1735630300.000000"

    @pytest.mark.asyncio
    async def test_poll_with_no_new_messages(self, adapter, mock_poll_context):
        """Test poll() with no new messages returns empty events list."""
        cursor = {"last_ts": "1735630100.000000"}

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await adapter.poll(mock_poll_context, cursor)

            assert len(result.events) == 0
            # Cursor should remain unchanged
            assert result.cursor["last_ts"] == "1735630100.000000"
            assert result.has_more is False


@pytest.mark.unit
class TestPollFiltersMessages:
    """Test poll() message filtering."""

    @pytest.mark.asyncio
    async def test_poll_filters_bot_messages(self, adapter, mock_poll_context):
        """Test poll() filters out bot messages when include_bot_messages is False."""
        cursor = {"last_ts": "1735630100.000000"}
        messages = [
            {
                "ts": "1735630200.000000",
                "text": "Human message",
                "user": "U12345678",
            },
            {
                "ts": "1735630300.000000",
                "text": "Bot message",
                "bot_id": "B12345678",
            },
        ]

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=messages,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

            # Only human message should be included
            assert len(result.events) == 1
            assert result.events[0].payload["text"] == "Human message"
            # Cursor should still advance to newest message
            assert result.cursor["last_ts"] == "1735630300.000000"

    @pytest.mark.asyncio
    async def test_poll_includes_bot_messages_when_configured(
        self, adapter, mock_poll_context
    ):
        """Test poll() includes bot messages when include_bot_messages is True."""
        mock_poll_context.subscription.provider_config["include_bot_messages"] = True
        cursor = {"last_ts": "1735630100.000000"}
        messages = [
            {
                "ts": "1735630200.000000",
                "text": "Human message",
                "user": "U12345678",
            },
            {
                "ts": "1735630300.000000",
                "text": "Bot message",
                "bot_id": "B12345678",
            },
        ]

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=messages,
        ):
            result = await adapter.poll(mock_poll_context, cursor)

            # Both messages should be included
            assert len(result.events) == 2

    @pytest.mark.asyncio
    async def test_poll_app_mentions_filter(self, adapter, mock_poll_context):
        """Test poll() only_app_mentions filter."""
        mock_poll_context.subscription.provider_config["only_app_mentions"] = True
        cursor = {"last_ts": "1735630100.000000"}
        messages = [
            {
                "ts": "1735630200.000000",
                "text": "Hello <@U_BOT_ID> please help",
                "user": "U12345678",
            },
            {
                "ts": "1735630300.000000",
                "text": "No mention here",
                "user": "U87654321",
            },
        ]

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=messages,
        ):
            with patch.object(
                adapter,
                "_get_bot_user_id",
                new_callable=AsyncMock,
                return_value="U_BOT_ID",
            ):
                result = await adapter.poll(mock_poll_context, cursor)

                # Only message with bot mention should be included
                assert len(result.events) == 1
                assert "please help" in result.events[0].payload["text"]


@pytest.mark.unit
class TestErrorHandling:
    """Test error handling in the adapter."""

    @pytest.mark.asyncio
    async def test_auth_error_is_permanent(self, adapter, mock_poll_context):
        """Test authentication errors are marked as permanent."""
        cursor = {"last_ts": None}

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            side_effect=PollAdapterError(
                "Slack authentication error", permanent=True
            ),
        ):
            with pytest.raises(PollAdapterError) as exc_info:
                await adapter.poll(mock_poll_context, cursor)

            assert exc_info.value.permanent is True

    @pytest.mark.asyncio
    async def test_rate_limit_includes_backoff(self, adapter, mock_poll_context):
        """Test rate limit errors include backoff seconds."""
        cursor = {"last_ts": None}

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            side_effect=PollAdapterError(
                "Slack rate limited", backoff_seconds=60
            ),
        ):
            with pytest.raises(PollAdapterError) as exc_info:
                await adapter.poll(mock_poll_context, cursor)

            assert exc_info.value.backoff_seconds == 60


@pytest.mark.unit
class TestNormalizeMessage:
    """Test message normalization."""

    def test_normalize_standard_message(self, adapter):
        """Test normalizing a standard user message."""
        message = {
            "ts": "1735630200.000000",
            "text": "Hello world!",
            "user": "U12345678",
            "username": "johndoe",
        }

        result = adapter._normalize_message(
            message,
            workspace_id="T01234567890",
            channel_id="C01234567890",
            bot_user_id=None,
        )

        assert result["message_id"] == "1735630200.000000"
        assert result["channel_id"] == "C01234567890"
        assert result["workspace_id"] == "T01234567890"
        assert result["text"] == "Hello world!"
        assert result["user"]["id"] == "U12345678"
        assert result["user"]["is_bot"] is False
        assert result["mentions_bot"] is False

    def test_normalize_thread_reply(self, adapter):
        """Test normalizing a thread reply message."""
        message = {
            "ts": "1735630300.000000",
            "text": "This is a reply",
            "user": "U12345678",
            "thread_ts": "1735630200.000000",
        }

        result = adapter._normalize_message(
            message,
            workspace_id="T01234567890",
            channel_id="C01234567890",
            bot_user_id=None,
        )

        assert result["thread_ts"] == "1735630200.000000"

    def test_normalize_bot_message(self, adapter):
        """Test normalizing a bot message."""
        message = {
            "ts": "1735630200.000000",
            "text": "Bot says hello",
            "bot_id": "B12345678",
            "username": "mybot",
        }

        result = adapter._normalize_message(
            message,
            workspace_id="T01234567890",
            channel_id="C01234567890",
            bot_user_id=None,
        )

        assert result["user"]["id"] == "B12345678"
        assert result["user"]["is_bot"] is True

    def test_normalize_message_with_bot_mention(self, adapter):
        """Test normalizing a message that mentions the bot."""
        message = {
            "ts": "1735630200.000000",
            "text": "Hey <@U_BOT_ID> can you help?",
            "user": "U12345678",
        }

        result = adapter._normalize_message(
            message,
            workspace_id="T01234567890",
            channel_id="C01234567890",
            bot_user_id="U_BOT_ID",
        )

        assert result["mentions_bot"] is True


@pytest.mark.unit
class TestTimestampComparison:
    """Test timestamp comparison."""

    def test_compare_timestamps_greater(self, adapter):
        """Test comparing timestamps where first is greater."""
        result = adapter._compare_timestamps("1735630200.000000", "1735630100.000000")
        assert result == 1

    def test_compare_timestamps_less(self, adapter):
        """Test comparing timestamps where first is less."""
        result = adapter._compare_timestamps("1735630100.000000", "1735630200.000000")
        assert result == -1

    def test_compare_timestamps_equal(self, adapter):
        """Test comparing equal timestamps."""
        result = adapter._compare_timestamps("1735630200.000000", "1735630200.000000")
        assert result == 0

    def test_compare_invalid_timestamps(self, adapter):
        """Test comparing invalid timestamps falls back to string comparison."""
        result = adapter._compare_timestamps("invalid", "also_invalid")
        # String comparison: "invalid" > "also_invalid"
        assert result == 1


@pytest.mark.unit
class TestIntegrationBootstrapAndPoll:
    """Test how bootstrap_cursor integrates with the polling lifecycle."""

    @pytest.mark.asyncio
    async def test_bootstrap_then_poll_only_fetches_new_messages(
        self, adapter, mock_poll_context
    ):
        """
        Integration test: bootstrap_cursor + poll should only process new messages.

        Scenario:
        1. Channel has 3 existing messages (ts: 100, 200, 300)
        2. bootstrap_cursor fetches most recent (300) and sets cursor
        3. poll() is called with cursor {"last_ts": "1735630300.000000"}
        4. Only messages with ts > 300 should be fetched (e.g., 400, 500)
        """
        # Step 1: Bootstrap fetches most recent message
        bootstrap_message = {
            "ts": "1735630300.000000",
            "text": "Most recent existing message",
            "user": "U12345678",
        }

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=[bootstrap_message],
        ):
            cursor = await adapter.bootstrap_cursor(mock_poll_context)

            assert cursor == {"last_ts": "1735630300.000000"}

        # Step 2: Simulate poll() with the bootstrapped cursor
        # New messages arrive: 400, 500
        new_messages = [
            {
                "ts": "1735630400.000000",
                "text": "New message 1",
                "user": "U12345678",
            },
            {
                "ts": "1735630500.000000",
                "text": "New message 2",
                "user": "U87654321",
            },
        ]

        with patch.object(
            adapter,
            "_fetch_slack_messages",
            new_callable=AsyncMock,
            return_value=new_messages,
        ) as mock_fetch:
            result = await adapter.poll(mock_poll_context, cursor)

            # Verify poll() called fetch with oldest_ts from cursor
            mock_fetch.assert_called_once_with(
                "xoxb-test-token-12345",
                "C01234567890",
                oldest_ts="1735630300.000000",
                max_results=50,
            )

            # Verify only new messages are in events
            assert len(result.events) == 2
            assert result.events[0].provider_event_id == "1735630400.000000"
            assert result.events[1].provider_event_id == "1735630500.000000"

            # Verify cursor advanced to newest message
            assert result.cursor["last_ts"] == "1735630500.000000"
