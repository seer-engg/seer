from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from seer.config import config
from seer.core.triggers.polling.adapters.base import (
    PollAdapter,
    PollAdapterError,
    PollContext,
    PolledEvent,
    PollResult,
    register_adapter,
)
from seer.logger import get_logger

logger = get_logger(__name__)

DISCORD_API_BASE = "https://discord.com/api/v10"
MAX_MESSAGES_PER_POLL = 100
DEFAULT_MAX_RESULTS = 50


def _parse_discord_timestamp(timestamp_str: Optional[str]) -> Optional[datetime]:
    """Parse Discord ISO8601 timestamp string to datetime."""
    if not timestamp_str:
        return None
    try:
        # Discord timestamps are ISO8601 with optional timezone
        # Remove the 'Z' suffix and parse
        if timestamp_str.endswith("Z"):
            timestamp_str = timestamp_str[:-1] + "+00:00"
        return datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning("Failed to parse Discord timestamp: %s", timestamp_str)
        return None


class DiscordMessageReceivedAdapter(PollAdapter):
    """Poll Discord channel for new messages using message ID cursor strategy."""

    trigger_key = "poll.discord.message_received"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """Initialize cursor - fetch recent messages on first poll."""
        # Start with None to fetch the most recent messages
        # We'll track the newest message ID we've seen
        return {"last_message_id": None}

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """Poll Discord channel for new messages."""
        bot_token = self._get_bot_token()
        channel_id = self._resolve_channel_id(ctx)
        guild_id = self._resolve_guild_id(ctx)
        max_results = self._resolve_max_results(ctx)
        include_bot_messages = self._resolve_include_bot_messages(ctx)
        only_mentions = self._resolve_only_mentions(ctx)

        last_message_id = cursor.get("last_message_id")

        try:
            messages = await self._fetch_discord_messages(bot_token, channel_id, last_message_id, max_results)

            if not messages:
                return PollResult(
                    events=[],
                    cursor={"last_message_id": last_message_id},
                    has_more=False,
                )

            # Process messages and create events
            process_config = {
                "guild_id": guild_id,
                "bot_token": bot_token,
                "last_message_id": last_message_id,
                "include_bot_messages": include_bot_messages,
                "only_mentions": only_mentions,
            }
            polled_events, new_last_message_id = await self._process_messages(messages, process_config)

            has_more = len(messages) >= max_results

            return PollResult(
                events=polled_events,
                cursor={"last_message_id": new_last_message_id or last_message_id},
                has_more=has_more,
            )
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Discord polling failure")
            raise PollAdapterError(
                "Unexpected Discord polling failure", detail={"error": str(exc)}
            ) from exc

    async def _fetch_discord_messages(
        self, bot_token: str, channel_id: str, last_message_id: Optional[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch messages from Discord API."""
        if last_message_id:
            params = {"after": last_message_id, "limit": max_results}
        else:
            params = {"limit": max_results}

        headers = {
            "Authorization": f"Bot {bot_token}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
                headers=headers,
                params=params,
            )
            await self._raise_for_status(resp)
            messages = resp.json()

            if not messages or not isinstance(messages, list):
                return []

            # Discord returns messages in reverse chronological order (newest first)
            # We want to process oldest first to maintain chronological order
            messages.reverse()
            return messages

    async def _process_messages(
        self,
        messages: List[Dict[str, Any]],
        process_config: Dict[str, Any],
    ) -> tuple[List[PolledEvent], Optional[str]]:
        """Process messages and create polled events."""
        polled_events: List[PolledEvent] = []
        new_last_message_id = process_config["last_message_id"]

        # Get bot user ID to check mentions
        bot_user_id = await self._get_bot_user_id(process_config["bot_token"])

        for message in messages:
            msg_id = message.get("id")
            if not msg_id:
                continue

            # Update cursor to newest message ID
            if new_last_message_id is None or self._compare_message_ids(msg_id, new_last_message_id) > 0:
                new_last_message_id = msg_id

            # Apply filters
            if not self._should_include_message(
                message, bot_user_id, process_config["include_bot_messages"], process_config["only_mentions"]
            ):
                continue

            # Create event for this message
            event = self._create_polled_event(message, process_config["guild_id"], bot_user_id)
            polled_events.append(event)

        return polled_events, new_last_message_id

    def _create_polled_event(
        self, message: Dict[str, Any], guild_id: Optional[str], bot_user_id: Optional[str]
    ) -> PolledEvent:
        """Create a PolledEvent from a Discord message."""
        msg_id = message.get("id")
        normalized_payload = self._normalize_message(message, guild_id, bot_user_id)

        # Parse timestamp
        timestamp_str = message.get("timestamp")
        occurred_at = _parse_discord_timestamp(timestamp_str)
        if not occurred_at:
            # Fallback to current time if timestamp parsing fails
            occurred_at = datetime.now(timezone.utc)

        return PolledEvent(
            payload=normalized_payload,
            raw=message,
            provider_event_id=msg_id,
            occurred_at=occurred_at,
        )

    def _normalize_message(
        self, message: Dict[str, Any], guild_id: Optional[str], bot_user_id: Optional[str]
    ) -> Dict[str, Any]:
        """Normalize Discord message to standard payload format."""
        author = message.get("author") or {}
        mentions = message.get("mentions") or []

        # Check if bot is mentioned
        mentions_bot = False
        if bot_user_id:
            mentions_bot = any(
                mention.get("id") == bot_user_id for mention in mentions if isinstance(mention, dict)
            )

        return {
            "message_id": message.get("id"),
            "channel_id": message.get("channel_id"),
            "guild_id": guild_id or message.get("guild_id"),
            "author": {
                "id": author.get("id"),
                "username": author.get("username"),
                "discriminator": author.get("discriminator"),
                "bot": author.get("bot", False),
            },
            "content": message.get("content"),
            "timestamp": message.get("timestamp"),
            "edited_timestamp": message.get("edited_timestamp"),
            "mentions_bot": mentions_bot,
            "attachments": message.get("attachments", []),
            "embeds": message.get("embeds", []),
        }

    def _should_include_message(
        self,
        message: Dict[str, Any],
        bot_user_id: Optional[str],
        include_bot_messages: bool,
        only_mentions: bool,
    ) -> bool:
        """Check if message should be included based on filters."""
        author = message.get("author") or {}
        is_bot = author.get("bot", False)

        # Filter out bot messages if configured
        if not include_bot_messages and is_bot:
            return False

        # Only include messages that mention the bot if configured
        if only_mentions:
            if not bot_user_id:
                return False
            mentions = message.get("mentions") or []
            return any(
                mention.get("id") == bot_user_id
                for mention in mentions
                if isinstance(mention, dict)
            )

        return True

    def _compare_message_ids(self, msg_id1: str, msg_id2: str) -> int:
        """Compare Discord message IDs (snowflakes) as integers."""
        try:
            id1 = int(msg_id1)
            id2 = int(msg_id2)
            if id1 > id2:
                return 1
            if id1 < id2:
                return -1
            return 0
        except (ValueError, TypeError):
            # Fallback to string comparison
            if msg_id1 > msg_id2:
                return 1
            if msg_id1 < msg_id2:
                return -1
            return 0

    async def _get_bot_user_id(self, bot_token: str) -> Optional[str]:
        """Get bot's user ID from Discord API."""
        try:
            headers = {
                "Authorization": f"Bot {bot_token}",
                "Content-Type": "application/json",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{DISCORD_API_BASE}/users/@me", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("id")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Reason: Non-critical helper - we want to catch all errors and continue polling without bot_user_id
            logger.warning("Failed to fetch bot user ID: %s", exc)
        return None

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Handle HTTP errors from Discord API."""
        if response.status_code < 400:
            return
        detail = {"status": response.status_code, "body": response.text[:500]}
        if response.status_code in {401, 403}:
            raise PollAdapterError(
                "Discord authentication error", permanent=True, detail=detail
            )
        if response.status_code == 404:
            raise PollAdapterError(
                "Discord channel not found or inaccessible", permanent=True, detail=detail
            )
        if response.status_code == 429:
            raise PollAdapterError(
                "Discord rate limited", backoff_seconds=60, detail=detail
            )
        raise PollAdapterError("Discord API error", detail=detail)

    def _get_bot_token(self) -> str:
        """Get Discord bot token from config."""
        if not config.discord_bot_token:
            raise PollAdapterError(
                "Discord bot token not configured",
                permanent=True,
                detail={"error": "discord_bot_token missing from config"},
            )
        return config.discord_bot_token

    def _resolve_channel_id(self, ctx: PollContext) -> str:
        """Resolve channel_id from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        channel_id = config_dict.get("channel_id")
        if not channel_id:
            raise PollAdapterError(
                "channel_id is required in provider_config",
                permanent=True,
                detail={"error": "Missing channel_id in subscription config"},
            )
        return str(channel_id)

    def _resolve_guild_id(self, ctx: PollContext) -> Optional[str]:
        """Resolve guild_id from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        guild_id = config_dict.get("guild_id")
        if guild_id:
            return str(guild_id)
        return None

    def _resolve_max_results(self, ctx: PollContext) -> int:
        """Resolve max_results from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        value = config_dict.get("max_results", DEFAULT_MAX_RESULTS)
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            numeric = DEFAULT_MAX_RESULTS
        return max(1, min(numeric, MAX_MESSAGES_PER_POLL))

    def _resolve_include_bot_messages(self, ctx: PollContext) -> bool:
        """Resolve include_bot_messages flag from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        return config_dict.get("include_bot_messages", False)

    def _resolve_only_mentions(self, ctx: PollContext) -> bool:
        """Resolve only_mentions flag from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        return config_dict.get("only_mentions", False)


register_adapter(DiscordMessageReceivedAdapter())
