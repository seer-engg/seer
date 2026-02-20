"""
Slack message polling adapter.

Polls a Slack channel for new messages using OAuth credentials.
Uses message timestamp (ts) as cursor for pagination.
"""
# pylint: disable=duplicate-code  # Reason: Discord and Slack polling adapters share similar config resolution patterns intentionally

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

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

SLACK_API_BASE = "https://slack.com/api"
MAX_MESSAGES_PER_POLL = 100
DEFAULT_MAX_RESULTS = 50


def _parse_slack_timestamp(ts: Optional[str]) -> Optional[datetime]:
    """
    Parse Slack message timestamp to datetime.

    Slack timestamps are in format "1234567890.123456" (Unix epoch with microseconds).
    """
    if not ts:
        return None
    try:
        # Slack ts format: "1234567890.123456" (seconds.microseconds)
        epoch_seconds = float(ts)
        return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
    except (ValueError, TypeError, OSError):
        logger.warning("Failed to parse Slack timestamp: %s", ts)
        return None


class SlackMessageReceivedAdapter(PollAdapter):
    """Poll Slack channel for new messages using message timestamp cursor strategy."""

    trigger_key = "poll.slack.message_received"

    async def bootstrap_cursor(self, ctx: PollContext) -> Dict[str, Any]:
        """Initialize cursor by fetching the most recent message timestamp to skip historical messages."""
        access_token = self._get_access_token(ctx)
        channel_id = self._resolve_channel_id(ctx)

        try:
            # Fetch only the single most recent message to establish the cursor
            messages = await self._fetch_slack_messages(
                access_token, channel_id, oldest_ts=None, max_results=1
            )

            if messages and len(messages) > 0:
                # Set cursor to the most recent message timestamp
                # Only messages AFTER this will be processed in future polls
                most_recent_ts = messages[0].get("ts")
                return {"last_ts": most_recent_ts}

            # No messages in channel yet - start with None
            # First message will be processed when it arrives
            return {"last_ts": None}

        except PollAdapterError:
            # If we can't fetch messages during bootstrap (permissions, rate limit, etc.)
            # start with None and let the first poll handle it
            logger.warning("Failed to bootstrap cursor for channel %s, starting with None", channel_id)
            return {"last_ts": None}

    async def poll(self, ctx: PollContext, cursor: Dict[str, Any]) -> PollResult:
        """Poll Slack channel for new messages."""
        access_token = self._get_access_token(ctx)
        channel_id = self._resolve_channel_id(ctx)
        workspace_id = self._resolve_workspace_id(ctx)
        max_results = self._resolve_max_results(ctx)
        include_bot_messages = self._resolve_include_bot_messages(ctx)
        only_app_mentions = self._resolve_only_app_mentions(ctx)

        last_ts = cursor.get("last_ts")

        try:
            messages = await self._fetch_slack_messages(
                access_token, channel_id, oldest_ts=last_ts, max_results=max_results
            )

            if not messages:
                return PollResult(
                    events=[],
                    cursor={"last_ts": last_ts},
                    has_more=False,
                )

            # Get bot user ID if we need to filter by mentions
            bot_user_id = None
            if only_app_mentions:
                bot_user_id = await self._get_bot_user_id(access_token)

            # Process messages and create events
            process_config = {
                "workspace_id": workspace_id,
                "channel_id": channel_id,
                "last_ts": last_ts,
                "include_bot_messages": include_bot_messages,
                "only_app_mentions": only_app_mentions,
                "bot_user_id": bot_user_id,
            }
            polled_events, new_last_ts = await self._process_messages(messages, process_config)

            has_more = len(messages) >= max_results

            return PollResult(
                events=polled_events,
                cursor={"last_ts": new_last_ts or last_ts},
                has_more=has_more,
            )
        except PollAdapterError:
            raise
        except Exception as exc:
            logger.exception("Unexpected Slack polling failure")
            raise PollAdapterError(
                "Unexpected Slack polling failure", detail={"error": str(exc)}
            ) from exc

    async def _fetch_slack_messages(
        self, access_token: str, channel_id: str, oldest_ts: Optional[str], max_results: int
    ) -> List[Dict[str, Any]]:
        """Fetch messages from Slack API using conversations.history."""
        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": max_results,
        }
        if oldest_ts:
            params["oldest"] = oldest_ts

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=utf-8",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{SLACK_API_BASE}/conversations.history",
                headers=headers,
                params=params,
            )
            await self._raise_for_status(resp)
            data = resp.json()

            # Check Slack's ok field
            if not data.get("ok"):
                await self._handle_slack_api_error(data)

            messages = data.get("messages", [])
            if not messages or not isinstance(messages, list):
                return []

            # Slack returns messages in reverse chronological order (newest first)
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
        new_last_ts = process_config["last_ts"]

        for message in messages:
            msg_ts = message.get("ts")
            if not msg_ts:
                continue

            # Update cursor to newest message timestamp
            if new_last_ts is None or self._compare_timestamps(msg_ts, new_last_ts) > 0:
                new_last_ts = msg_ts

            # Apply filters
            if not self._should_include_message(
                message,
                process_config["bot_user_id"],
                process_config["include_bot_messages"],
                process_config["only_app_mentions"],
            ):
                continue

            # Create event for this message
            event = self._create_polled_event(
                message,
                process_config["workspace_id"],
                process_config["channel_id"],
                process_config["bot_user_id"],
            )
            polled_events.append(event)

        return polled_events, new_last_ts

    def _create_polled_event(
        self,
        message: Dict[str, Any],
        workspace_id: Optional[str],
        channel_id: str,
        bot_user_id: Optional[str],
    ) -> PolledEvent:
        """Create a PolledEvent from a Slack message."""
        msg_ts = message.get("ts")
        normalized_payload = self._normalize_message(message, workspace_id, channel_id, bot_user_id)

        # Parse timestamp
        occurred_at = _parse_slack_timestamp(msg_ts)
        if not occurred_at:
            # Fallback to current time if timestamp parsing fails
            occurred_at = datetime.now(timezone.utc)

        return PolledEvent(
            payload=normalized_payload,
            raw=message,
            provider_event_id=msg_ts,
            occurred_at=occurred_at,
        )

    def _normalize_message(
        self,
        message: Dict[str, Any],
        workspace_id: Optional[str],
        channel_id: str,
        bot_user_id: Optional[str],
    ) -> Dict[str, Any]:
        """Normalize Slack message to standard payload format."""
        msg_ts = message.get("ts")
        user_id = message.get("user")
        bot_id = message.get("bot_id")

        # Determine if message is from a bot
        is_bot = bool(bot_id) or message.get("subtype") == "bot_message"

        # Check if bot is mentioned (Slack mentions look like <@U123456>)
        mentions_bot = False
        text = message.get("text", "")
        if bot_user_id and f"<@{bot_user_id}>" in text:
            mentions_bot = True

        # Parse timestamp to ISO format
        occurred_at = _parse_slack_timestamp(msg_ts)
        timestamp_iso = occurred_at.isoformat() if occurred_at else None

        return {
            "message_id": msg_ts,
            "channel_id": channel_id,
            "workspace_id": workspace_id,
            "user": {
                "id": user_id or bot_id,
                "username": message.get("username"),
                "is_bot": is_bot,
            },
            "text": text,
            "timestamp": timestamp_iso,
            "thread_ts": message.get("thread_ts"),
            "mentions_bot": mentions_bot,
            "attachments": message.get("attachments", []),
            "blocks": message.get("blocks", []),
        }

    def _should_include_message(
        self,
        message: Dict[str, Any],
        bot_user_id: Optional[str],
        include_bot_messages: bool,
        only_app_mentions: bool,
    ) -> bool:
        """Check if message should be included based on filters."""
        bot_id = message.get("bot_id")
        is_bot = bool(bot_id) or message.get("subtype") == "bot_message"

        # Filter out bot messages if configured
        if not include_bot_messages and is_bot:
            return False

        # Only include messages that mention the bot if configured
        if only_app_mentions:
            if not bot_user_id:
                return False
            text = message.get("text", "")
            if f"<@{bot_user_id}>" not in text:
                return False

        return True

    def _compare_timestamps(self, ts1: str, ts2: str) -> int:
        """Compare Slack timestamps as floats."""
        try:
            t1 = float(ts1)
            t2 = float(ts2)
            if t1 > t2:
                return 1
            if t1 < t2:
                return -1
            return 0
        except (ValueError, TypeError):
            # Fallback to string comparison
            if ts1 > ts2:
                return 1
            if ts1 < ts2:
                return -1
            return 0

    async def _get_bot_user_id(self, access_token: str) -> Optional[str]:
        """Get bot's user ID from Slack API."""
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json; charset=utf-8",
            }
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{SLACK_API_BASE}/auth.test", headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("ok"):
                        return data.get("user_id")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # Reason: Non-critical helper - we want to catch all errors and continue polling without bot_user_id
            logger.warning("Failed to fetch bot user ID: %s", exc)
        return None

    async def _raise_for_status(self, response: httpx.Response) -> None:
        """Handle HTTP errors from Slack API."""
        if response.status_code < 400:
            return
        detail = {"status": response.status_code, "body": response.text[:500]}
        if response.status_code in {401, 403}:
            raise PollAdapterError(
                "Slack authentication error", permanent=True, detail=detail
            )
        if response.status_code == 404:
            raise PollAdapterError(
                "Slack channel not found", permanent=True, detail=detail
            )
        if response.status_code == 429:
            # Try to get Retry-After header for backoff
            retry_after = response.headers.get("Retry-After")
            backoff = int(retry_after) if retry_after and retry_after.isdigit() else 60
            raise PollAdapterError(
                "Slack rate limited", backoff_seconds=backoff, detail=detail
            )
        raise PollAdapterError("Slack API error", detail=detail)

    async def _handle_slack_api_error(self, data: Dict[str, Any]) -> None:
        """Handle Slack API-level errors (ok: false)."""
        error = data.get("error", "unknown_error")
        detail = {"error": error, "response": data}

        # Permanent errors - subscription should be disabled
        permanent_errors = {
            "invalid_auth",
            "token_revoked",
            "account_inactive",
            "channel_not_found",
            "is_archived",
        }
        # Note: "not_in_channel" is intentionally NOT in permanent_errors
        # It's retryable so the trigger auto-recovers when bot is added to channel

        if error in permanent_errors:
            raise PollAdapterError(
                f"Slack API error: {error}", permanent=True, detail=detail
            )

        if error == "ratelimited":
            raise PollAdapterError(
                "Slack rate limited", backoff_seconds=60, detail=detail
            )

        raise PollAdapterError(f"Slack API error: {error}", detail=detail)

    def _get_access_token(self, ctx: PollContext) -> str:
        """Get OAuth access token from context."""
        if not ctx.access_token:
            raise PollAdapterError(
                "Slack access token not available",
                permanent=True,
                detail={"error": "No OAuth token in poll context"},
            )
        return ctx.access_token

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

    def _resolve_workspace_id(self, ctx: PollContext) -> Optional[str]:
        """Resolve workspace_id from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        workspace_id = config_dict.get("workspace_id")
        if workspace_id:
            return str(workspace_id)
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

    def _resolve_only_app_mentions(self, ctx: PollContext) -> bool:
        """Resolve only_app_mentions flag from subscription config."""
        config_dict = ctx.subscription.provider_config or {}
        return config_dict.get("only_app_mentions", False)


register_adapter(SlackMessageReceivedAdapter())
