"""Unit tests for DiscordMessageReceivedAdapter bootstrap_cursor behavior."""

from unittest.mock import AsyncMock, Mock, patch
from datetime import datetime, timezone

import pytest

from seer.core.triggers.polling.adapters.discord_message_received import (
    DiscordMessageReceivedAdapter,
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
        "channel_id": "123456789",
        "guild_id": "987654321",
        "max_results": 50,
        "include_bot_messages": False,
        "only_mentions": False,
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
    """Create a DiscordMessageReceivedAdapter instance."""
    return DiscordMessageReceivedAdapter()


class TestBootstrapCursorWithExistingMessages:
    """Test bootstrap_cursor when channel has existing messages."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_single_message(self, adapter, mock_poll_context):
        """Test bootstrap_cursor returns most recent message ID when channel has one message."""
        mock_message = {
            "id": "999888777",
            "content": "Test message",
            "timestamp": "2024-01-15T10:30:00+00:00",
            "author": {"id": "111", "username": "testuser", "bot": False},
        }

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_message],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                assert cursor == {"last_message_id": "999888777"}

    @pytest.mark.asyncio
    async def test_bootstrap_with_multiple_messages_returns_first(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor with multiple messages returns first (most recent after reverse)."""
        # Note: _fetch_discord_messages reverses the order, so the first message
        # in the returned list is actually the oldest
        mock_messages = [
            {
                "id": "100",
                "content": "Oldest",
                "timestamp": "2024-01-15T10:00:00+00:00",
            },
            {
                "id": "200",
                "content": "Middle",
                "timestamp": "2024-01-15T10:30:00+00:00",
            },
            {
                "id": "300",
                "content": "Newest",
                "timestamp": "2024-01-15T11:00:00+00:00",
            },
        ]

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_messages[0]],  # Only fetches 1 with max_results=1
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                # Should use the first (oldest) message from the reversed list
                assert cursor == {"last_message_id": "100"}

    @pytest.mark.asyncio
    async def test_bootstrap_calls_fetch_with_correct_params(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor calls _fetch_discord_messages with correct parameters."""
        mock_message = {"id": "123", "content": "Test"}

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_xyz"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_message],
            ) as mock_fetch:
                await adapter.bootstrap_cursor(mock_poll_context)

                # Verify it was called with max_results=1 to fetch only most recent
                mock_fetch.assert_called_once_with(
                    "bot_token_xyz",  # bot_token
                    "123456789",  # channel_id from provider_config
                    last_message_id=None,  # last_message_id (None for first fetch)
                    max_results=1,  # max_results=1 to get only most recent
                )


class TestBootstrapCursorWithEmptyChannel:
    """Test bootstrap_cursor when channel has no messages."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_empty_list(self, adapter, mock_poll_context):
        """Test bootstrap_cursor returns None when channel has no messages."""
        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                assert cursor == {"last_message_id": None}

    @pytest.mark.asyncio
    async def test_bootstrap_with_none_response(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles None response from API."""
        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=None,
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                assert cursor == {"last_message_id": None}


class TestBootstrapCursorErrorHandling:
    """Test bootstrap_cursor error handling and fallback behavior."""

    @pytest.mark.asyncio
    async def test_bootstrap_handles_poll_adapter_error(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor catches PollAdapterError and returns None cursor."""
        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                side_effect=PollAdapterError("Rate limited", backoff_seconds=60),
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                # Should fall back to None cursor on error
                assert cursor == {"last_message_id": None}

    @pytest.mark.asyncio
    async def test_bootstrap_handles_auth_error(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles authentication errors gracefully."""
        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                side_effect=PollAdapterError(
                    "Discord authentication error", permanent=True
                ),
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                # Should fall back to None cursor even on permanent errors during bootstrap
                assert cursor == {"last_message_id": None}

    @pytest.mark.asyncio
    async def test_bootstrap_handles_missing_bot_token(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor propagates PollAdapterError from _get_bot_token."""
        with patch.object(
            adapter,
            "_get_bot_token",
            side_effect=PollAdapterError(
                "Discord bot token not configured", permanent=True
            ),
        ):
            # _get_bot_token raises error before we catch it in try/except
            # This should propagate up
            with pytest.raises(PollAdapterError, match="Discord bot token not configured"):
                await adapter.bootstrap_cursor(mock_poll_context)

    @pytest.mark.asyncio
    async def test_bootstrap_handles_missing_channel_id(
        self, adapter, mock_poll_context
    ):
        """Test bootstrap_cursor propagates error when channel_id is missing."""
        # Remove channel_id from config
        mock_poll_context.subscription.provider_config = {}

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            # _resolve_channel_id will raise error before we catch it
            with pytest.raises(PollAdapterError, match="channel_id is required"):
                await adapter.bootstrap_cursor(mock_poll_context)

    @pytest.mark.asyncio
    async def test_bootstrap_logs_warning_on_error(self, adapter, mock_poll_context):
        """Test bootstrap_cursor logs warning when falling back to None cursor."""
        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                side_effect=PollAdapterError("Test error"),
            ):
                with patch("seer.core.triggers.polling.adapters.discord_message_received.logger") as mock_logger:
                    cursor = await adapter.bootstrap_cursor(mock_poll_context)

                    # Should log warning about failed bootstrap
                    mock_logger.warning.assert_called_once()
                    assert "Failed to bootstrap cursor" in str(
                        mock_logger.warning.call_args
                    )
                    assert cursor == {"last_message_id": None}


class TestBootstrapCursorEdgeCases:
    """Test bootstrap_cursor edge cases and malformed data."""

    @pytest.mark.asyncio
    async def test_bootstrap_with_message_missing_id(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles message without 'id' field."""
        mock_message = {"content": "Test", "timestamp": "2024-01-15T10:00:00+00:00"}

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_message],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                # message.get("id") returns None, should be treated as None
                assert cursor == {"last_message_id": None}

    @pytest.mark.asyncio
    async def test_bootstrap_with_message_id_none(self, adapter, mock_poll_context):
        """Test bootstrap_cursor when message 'id' is explicitly None."""
        mock_message = {
            "id": None,
            "content": "Test",
            "timestamp": "2024-01-15T10:00:00+00:00",
        }

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_message],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                # Should handle None ID gracefully
                assert cursor == {"last_message_id": None}

    @pytest.mark.asyncio
    async def test_bootstrap_with_numeric_message_id(self, adapter, mock_poll_context):
        """Test bootstrap_cursor handles numeric message IDs (Discord snowflakes)."""
        mock_message = {
            "id": "1234567890123456789",  # Discord snowflake as string
            "content": "Test",
        }

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[mock_message],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                assert cursor == {"last_message_id": "1234567890123456789"}


class TestBootstrapIntegrationWithPollLifecycle:
    """Test how bootstrap_cursor integrates with the polling lifecycle."""

    @pytest.mark.asyncio
    async def test_bootstrap_then_poll_only_fetches_new_messages(
        self, adapter, mock_poll_context
    ):
        """
        Integration test: bootstrap_cursor + poll should only process new messages.

        Scenario:
        1. Channel has 3 existing messages (IDs: 100, 200, 300)
        2. bootstrap_cursor fetches most recent (300) and sets cursor
        3. poll() is called with cursor {"last_message_id": "300"}
        4. Only messages with ID > 300 should be fetched (e.g., 400, 500)
        """
        # Step 1: Bootstrap fetches most recent message
        bootstrap_message = {
            "id": "300",
            "content": "Most recent existing message",
            "timestamp": "2024-01-15T10:00:00+00:00",
            "author": {"id": "user1", "username": "user1", "bot": False},
        }

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter,
                "_fetch_discord_messages",
                new_callable=AsyncMock,
                return_value=[bootstrap_message],
            ):
                cursor = await adapter.bootstrap_cursor(mock_poll_context)

                assert cursor == {"last_message_id": "300"}

        # Step 2: Simulate poll() with the bootstrapped cursor
        # New messages arrive: 400, 500
        new_messages = [
            {
                "id": "400",
                "content": "New message 1",
                "timestamp": "2024-01-15T11:00:00+00:00",
                "channel_id": "123456789",
                "author": {"id": "user2", "username": "user2", "bot": False},
            },
            {
                "id": "500",
                "content": "New message 2",
                "timestamp": "2024-01-15T12:00:00+00:00",
                "channel_id": "123456789",
                "author": {"id": "user3", "username": "user3", "bot": False},
            },
        ]

        with patch.object(adapter, "_get_bot_token", return_value="bot_token_123"):
            with patch.object(
                adapter, "_resolve_guild_id", return_value="987654321"
            ):
                with patch.object(
                    adapter, "_resolve_max_results", return_value=50
                ):
                    with patch.object(
                        adapter, "_resolve_include_bot_messages", return_value=False
                    ):
                        with patch.object(
                            adapter, "_resolve_only_mentions", return_value=False
                        ):
                            with patch.object(
                                adapter,
                                "_fetch_discord_messages",
                                new_callable=AsyncMock,
                                return_value=new_messages,
                            ) as mock_fetch:
                                with patch.object(
                                    adapter,
                                    "_get_bot_user_id",
                                    new_callable=AsyncMock,
                                    return_value="bot123",
                                ):
                                    result = await adapter.poll(
                                        mock_poll_context, cursor
                                    )

                                    # Verify poll() called fetch with after: "300"
                                    mock_fetch.assert_called_once_with(
                                        "bot_token_123",
                                        "123456789",
                                        "300",  # last_message_id from cursor
                                        50,
                                    )

                                    # Verify only new messages are in events
                                    assert len(result.events) == 2
                                    assert (
                                        result.events[0].provider_event_id == "400"
                                    )
                                    assert (
                                        result.events[1].provider_event_id == "500"
                                    )

                                    # Verify cursor advanced to newest message
                                    assert result.cursor["last_message_id"] == "500"
