"""
Polling-based trigger event browsers for Gmail, Discord, Slack, Google Calendar, and Google Sheets.

Extracted from trigger_event_browser to keep module sizes manageable.
These functions fetch live events from external APIs for trigger event browsing.
"""
# pylint: disable=too-many-lines # Reason: Further splitting would fragment related event browsing code across multiple modules; cohesion is preferred over arbitrary line limits
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from itertools import zip_longest
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.config import config
from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope
from seer.core.triggers.polling.adapters.gmail_email_received import (
    GmailEmailReceivedAdapter,
    MAX_MESSAGES_PER_POLL,
)
from seer.core.triggers.polling.adapters.discord_message_received import (
    DISCORD_API_BASE,
    DEFAULT_MAX_RESULTS as DISCORD_MAX_RESULTS,
    _parse_discord_timestamp,
)
from seer.core.triggers.polling.adapters.slack_message_received import (
    SLACK_API_BASE,
    DEFAULT_MAX_RESULTS as SLACK_MAX_RESULTS,
    SlackMessageReceivedAdapter,
    _parse_slack_timestamp,
)
from seer.core.triggers.polling.adapters.google_calendar_event_changed import (
    MAX_EVENTS_PER_POLL as CALENDAR_MAX_EVENTS,
    GoogleCalendarEventChangedAdapter,
    _normalize_event_with_type as _normalize_calendar_event,
)
from seer.core.triggers.polling.adapters.google_calendar_common import (
    CALENDAR_API_BASE,
    normalize_event as _normalize_calendar_event_common,
    parse_rfc3339 as _parse_rfc3339,
)
from seer.core.triggers.polling.adapters.google_calendar_event_start import (
    GoogleCalendarEventStartAdapter,
)
from seer.core.triggers.polling.adapters.google_sheets_row_added import (
    GoogleSheetsRowAddedAdapter,
    SHEETS_API_BASE,
    DEFAULT_SHEET_NAME,
)
from seer.database import User
from seer.logger import get_logger
from seer.tools.google.gmail.helpers import GMAIL_API_BASE, build_gmail_list_params
from seer.tools.oauth_manager import get_oauth_token

logger = get_logger(__name__)


# =============================================================================
# Gmail Event Browsing
# =============================================================================

async def list_gmail_events(
    user: User,
    provider_connection_id: int,
    options,
    gmail_adapter: GmailEmailReceivedAdapter,
) -> Dict[str, Any]:
    """List Gmail emails for browsing."""
    _, access_token = await get_oauth_token(
        user, connection_id=str(provider_connection_id)
    )

    label_ids = (options.filter_params or {}).get("label_ids", ["INBOX"])
    query = (options.filter_params or {}).get("query", "")
    page_size = min(options.page_size, MAX_MESSAGES_PER_POLL)

    params = build_gmail_list_params(page_size, label_ids, query)
    if options.page_token:
        params["pageToken"] = options.page_token

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            list_resp = await client.get(
                f"{GMAIL_API_BASE}/messages", headers=headers, params=params
            )
            if list_resp.status_code >= 400:
                _handle_gmail_error(list_resp)

            list_json = list_resp.json()
            messages, next_page_token = list_json.get("messages", []) or [], list_json.get("nextPageToken")

            if not messages:
                return {
                    "items": [],
                    "next_page_token": None,
                    "trigger_key": options.trigger_key,
                    "supports_search": True,
                }

            items: List[Dict[str, Any]] = []
            for message in messages[:page_size]:
                msg_id = message.get("id")
                if not msg_id:
                    continue

                msg_resp = await client.get(
                    f"{GMAIL_API_BASE}/messages/{msg_id}",
                    headers=headers,
                    params={
                        "format": "metadata",
                        "metadataHeaders": ["From", "To", "Subject", "Date"],
                    },
                )
                if msg_resp.status_code >= 400:
                    logger.warning(
                        "Failed to fetch message metadata",
                        extra={"message_id": msg_id, "status": msg_resp.status_code},
                    )
                    continue

                items.append(_build_gmail_event_item(
                    msg_data=msg_resp.json(),
                    trigger_key=options.trigger_key,
                    trigger_id=options.trigger_id,
                    provider_connection_id=provider_connection_id,
                    gmail_adapter=gmail_adapter,
                ))

            return {
                "items": items,
                "next_page_token": next_page_token,
                "trigger_key": options.trigger_key,
                "supports_search": True,
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list Gmail events")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Gmail messages: {str(exc)}"
        ) from exc


def _build_gmail_event_item(
    *,
    msg_data: Dict[str, Any],
    trigger_key: str,
    trigger_id: str,
    provider_connection_id: int,
    gmail_adapter: GmailEmailReceivedAdapter,
) -> Dict[str, Any]:
    """Build a TriggerEventItem from Gmail message data."""
    normalized_payload = gmail_adapter._normalize_message(msg_data)  # pylint: disable=protected-access # Reason: Reusing adapter's message normalization

    subject = normalized_payload.get("subject") or "(No subject)"
    from_info = normalized_payload.get("from", {})
    from_display = from_info.get("name") or from_info.get("email") or "Unknown"
    snippet = msg_data.get("snippet", "")
    internal_date_ms = int(msg_data.get("internalDate") or 0)

    occurred_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc)
    display_subtitle = occurred_at.strftime("%b %d, %Y %I:%M %p")

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=f"{subject} - from {from_display}",
            provider="gmail",
            provider_connection_id=provider_connection_id,
            payload=normalized_payload,
            raw=msg_data,
            occurred_at=occurred_at,
        )
    )

    return {
        "id": normalized_payload.get("message_id"),
        "display_title": f"{subject} - from {from_display}",
        "display_subtitle": display_subtitle,
        "preview": snippet[:200] if snippet else None,
        "envelope": envelope,
        "metadata": {
            "labels": normalized_payload.get("labels", []),
            "thread_id": normalized_payload.get("thread_id"),
        },
    }


def _handle_gmail_error(response: httpx.Response) -> None:
    """Handle Gmail API errors."""
    if response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Gmail authentication failed. Please reconnect your Google account.",
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail="Gmail access denied. Please ensure you have granted Gmail permissions.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Gmail rate limit exceeded. Please try again later.",
        )
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Gmail API error: {response.text[:200]}",
    )


# =============================================================================
# Discord Event Browsing
# =============================================================================

async def list_discord_events(
    options,
) -> Dict[str, Any]:
    """
    List Discord messages for browsing.

    Discord uses bot token authentication (not OAuth), so provider_connection_id
    is not used. The channel_id must be provided in filter_params.
    """
    bot_token = _get_discord_bot_token()
    channel_id = (options.filter_params or {}).get("channel_id")
    guild_id = (options.filter_params or {}).get("guild_id")

    if not channel_id:
        raise HTTPException(
            status_code=400,
            detail="channel_id is required in filter_params for Discord message browsing",
        )

    page_size = min(options.page_size, DISCORD_MAX_RESULTS)

    try:
        messages = await _fetch_discord_messages(bot_token, channel_id, page_size)

        if not messages:
            return {
                "items": [],
                "next_page_token": None,
                "trigger_key": options.trigger_key,
                "supports_search": False,
            }

        items: List[Dict[str, Any]] = []
        for message in messages:
            items.append(_build_discord_event_item(
                message=message,
                guild_id=guild_id,
                trigger_key=options.trigger_key,
                trigger_id=options.trigger_id,
            ))

        oldest_message_id = messages[-1].get("id") if messages else None

        return {
            "items": items,
            "next_page_token": oldest_message_id,
            "trigger_key": options.trigger_key,
            "supports_search": False,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list Discord events")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Discord messages: {str(exc)}"
        ) from exc


async def _fetch_discord_messages(
    bot_token: str, channel_id: str, max_results: int
) -> List[Dict[str, Any]]:
    """Fetch messages from Discord API."""
    headers = {
        "Authorization": f"Bot {bot_token}",
        "Content-Type": "application/json",
    }
    params = {"limit": max_results}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
            headers=headers,
            params=params,
        )
        _handle_discord_error(resp, channel_id)
        messages = resp.json()

        if not messages or not isinstance(messages, list):
            return []

        return messages


def _build_discord_event_item(
    *,
    message: Dict[str, Any],
    guild_id: Optional[str],
    trigger_key: str,
    trigger_id: str,
) -> Dict[str, Any]:
    """Build a TriggerEventItem from Discord message data."""
    author = message.get("author") or {}
    content = message.get("content", "")
    msg_id = message.get("id")

    username = author.get("username", "Unknown")
    display_title = f"{username}: {content[:80]}{'...' if len(content) > 80 else ''}"
    if not content:
        display_title = f"{username}: (attachment or embed)"

    timestamp_str = message.get("timestamp")
    occurred_at = _parse_discord_timestamp(timestamp_str)
    if not occurred_at:
        occurred_at = datetime.now(timezone.utc)
    display_subtitle = occurred_at.strftime("%b %d, %Y %I:%M %p")

    normalized_payload = {
        "message_id": msg_id,
        "channel_id": message.get("channel_id"),
        "guild_id": guild_id or message.get("guild_id"),
        "author": {
            "id": author.get("id"),
            "username": author.get("username"),
            "discriminator": author.get("discriminator"),
            "bot": author.get("bot", False),
        },
        "content": content,
        "timestamp": message.get("timestamp"),
        "edited_timestamp": message.get("edited_timestamp"),
        "attachments": message.get("attachments", []),
        "embeds": message.get("embeds", []),
    }

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=f"Discord message from {username}",
            provider="discord",
            provider_connection_id=None,
            payload=normalized_payload,
            raw=message,
            occurred_at=occurred_at,
        )
    )

    return {
        "id": msg_id,
        "display_title": display_title,
        "display_subtitle": display_subtitle,
        "preview": content[:200] if content else None,
        "envelope": envelope,
        "metadata": {
            "author_id": author.get("id"),
            "channel_id": message.get("channel_id"),
            "guild_id": guild_id or message.get("guild_id"),
        },
    }


def _get_discord_bot_token() -> str:
    """Get Discord bot token from config."""
    if not config.discord_bot_token:
        raise HTTPException(
            status_code=500,
            detail="Discord bot token not configured. Please configure DISCORD_BOT_TOKEN.",
        )
    return config.discord_bot_token


def _handle_discord_error(response: httpx.Response, channel_id: str) -> None:
    """Handle Discord API errors."""
    if response.status_code < 400:
        return
    if response.status_code in {401, 403}:
        raise HTTPException(
            status_code=response.status_code,
            detail="Discord authentication error. Please check bot token and permissions.",
        )
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Discord channel {channel_id} not found or bot doesn't have access.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Discord rate limit exceeded. Please try again later.",
        )
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Discord API error: {response.text[:200]}",
    )


# =============================================================================
# Slack Event Browsing
# =============================================================================

async def list_slack_events(
    user: User,
    provider_connection_id: int,
    options,
    slack_adapter: SlackMessageReceivedAdapter,
) -> Dict[str, Any]:
    """List Slack messages for browsing."""
    _, access_token = await get_oauth_token(
        user, connection_id=str(provider_connection_id)
    )

    channel_id = (options.filter_params or {}).get("channel_id")
    workspace_id = (options.filter_params or {}).get("workspace_id")

    if not channel_id:
        raise HTTPException(
            status_code=400,
            detail="channel_id is required in filter_params for Slack message browsing",
        )

    page_size = min(options.page_size, SLACK_MAX_RESULTS)

    try:
        messages = await _fetch_slack_messages(access_token, channel_id, page_size)

        if not messages:
            return {
                "items": [],
                "next_page_token": None,
                "trigger_key": options.trigger_key,
                "supports_search": False,
            }

        items: List[Dict[str, Any]] = []
        for message in messages:
            items.append(_build_slack_event_item(
                message=message,
                workspace_id=workspace_id,
                channel_id=channel_id,
                trigger_key=options.trigger_key,
                trigger_id=options.trigger_id,
                provider_connection_id=provider_connection_id,
                slack_adapter=slack_adapter,
            ))

        # Slack doesn't have standard pagination tokens; use oldest message ts
        oldest_message_ts = messages[-1].get("ts") if messages else None

        return {
            "items": items,
            "next_page_token": oldest_message_ts,
            "trigger_key": options.trigger_key,
            "supports_search": False,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list Slack events")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Slack messages: {str(exc)}"
        ) from exc


async def _fetch_slack_messages(
    access_token: str, channel_id: str, max_results: int
) -> List[Dict[str, Any]]:
    """Fetch messages from Slack API using conversations.history."""
    params: Dict[str, Any] = {
        "channel": channel_id,
        "limit": max_results,
    }

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
        _handle_slack_error(resp, channel_id)
        data = resp.json()

        # Check Slack's ok field
        if not data.get("ok"):
            error = data.get("error", "unknown_error")
            raise HTTPException(
                status_code=400,
                detail=f"Slack API error: {error}",
            )

        messages = data.get("messages", [])
        if not messages or not isinstance(messages, list):
            return []

        # Slack returns messages in reverse chronological order (newest first)
        # Keep this order for browsing (newest first is more intuitive)
        return messages


def _build_slack_event_item(
    *,
    message: Dict[str, Any],
    workspace_id: Optional[str],
    channel_id: str,
    trigger_key: str,
    trigger_id: str,
    provider_connection_id: int,
    slack_adapter: SlackMessageReceivedAdapter,
) -> Dict[str, Any]:
    """Build a TriggerEventItem from Slack message data."""
    # Reuse adapter's normalization logic
    normalized_payload = slack_adapter._normalize_message(message, workspace_id, channel_id, None)  # pylint: disable=protected-access # Reason: Reusing adapter's message normalization

    msg_ts = message.get("ts")
    text = message.get("text", "")
    user_info = normalized_payload.get("user", {})
    username = user_info.get("username") or user_info.get("id") or "Unknown"

    # Build display title
    if text:
        display_title = f"{username}: {text[:80]}{'...' if len(text) > 80 else ''}"
    else:
        display_title = f"{username}: (attachment or block)"

    # Parse timestamp
    occurred_at = _parse_slack_timestamp(msg_ts)
    if not occurred_at:
        occurred_at = datetime.now(timezone.utc)
    display_subtitle = occurred_at.strftime("%b %d, %Y %I:%M %p")

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=f"Slack message from {username}",
            provider="slack",
            provider_connection_id=provider_connection_id,
            payload=normalized_payload,
            raw=message,
            occurred_at=occurred_at,
        )
    )

    return {
        "id": msg_ts,
        "display_title": display_title,
        "display_subtitle": display_subtitle,
        "preview": text[:200] if text else None,
        "envelope": envelope,
        "metadata": {
            "user_id": user_info.get("id"),
            "channel_id": channel_id,
            "workspace_id": workspace_id,
        },
    }


def _handle_slack_error(response: httpx.Response, channel_id: str) -> None:
    """Handle Slack API HTTP errors."""
    if response.status_code < 400:
        return
    if response.status_code in {401, 403}:
        raise HTTPException(
            status_code=response.status_code,
            detail="Slack authentication error. Please reconnect your Slack workspace.",
        )
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Slack channel {channel_id} not found or bot doesn't have access.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Slack rate limit exceeded. Please try again later.",
        )
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Slack API error: {response.text[:200]}",
    )


# =============================================================================
# Google Calendar Event Browsing
# =============================================================================


async def list_google_calendar_events(
    user: User,
    provider_connection_id: int,
    options,
    _gcal_adapter: GoogleCalendarEventChangedAdapter,  # pylint: disable=unused-argument # Reason: Interface consistency with other list_*_events functions
) -> Dict[str, Any]:
    """List Google Calendar events for browsing."""
    _, access_token = await get_oauth_token(
        user, connection_id=str(provider_connection_id)
    )

    calendar_id = (options.filter_params or {}).get("calendar_id", "primary")
    query = (options.filter_params or {}).get("query", "")
    page_size = min(options.page_size, CALENDAR_MAX_EVENTS)

    params: Dict[str, Any] = {
        "maxResults": page_size,
        "singleEvents": True,
        "orderBy": "startTime",
    }
    if query:
        params["q"] = query
    if options.page_token:
        params["pageToken"] = options.page_token

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                headers=headers,
                params=params,
            )
            if resp.status_code >= 400:
                _handle_google_calendar_error(resp, calendar_id)

            data = resp.json()
            events = data.get("items", []) or []
            next_page_token = data.get("nextPageToken")

            if not events:
                return {
                    "items": [],
                    "next_page_token": None,
                    "trigger_key": options.trigger_key,
                    "supports_search": True,
                }

            items: List[Dict[str, Any]] = []
            for event in events:
                # Skip cancelled events
                if event.get("status") == "cancelled":
                    continue
                items.append(_build_google_calendar_event_item(
                    event=event,
                    calendar_id=calendar_id,
                    trigger_key=options.trigger_key,
                    trigger_id=options.trigger_id,
                    provider_connection_id=provider_connection_id,
                ))

            return {
                "items": items,
                "next_page_token": next_page_token,
                "trigger_key": options.trigger_key,
                "supports_search": True,
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list Google Calendar events")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Calendar events: {str(exc)}"
        ) from exc


def _build_google_calendar_event_item(
    *,
    event: Dict[str, Any],
    calendar_id: str,
    trigger_key: str,
    trigger_id: str,
    provider_connection_id: int,
) -> Dict[str, Any]:
    """Build a TriggerEventItem from Google Calendar event data."""
    # Reuse adapter's normalization
    normalized_payload = _normalize_calendar_event(event, calendar_id)

    summary = normalized_payload.get("summary") or "(No title)"
    start_info = normalized_payload.get("start", {})
    start_dt_str = start_info.get("datetime", "")

    # Parse start time for display
    occurred_at = _parse_rfc3339(event.get("updated")) or datetime.now(timezone.utc)

    # Format display subtitle from start time
    start_dt = _parse_rfc3339(start_dt_str)
    if start_dt:
        display_subtitle = start_dt.strftime("%b %d, %Y %I:%M %p")
    else:
        # All-day event - just show date
        display_subtitle = start_dt_str if start_dt_str else occurred_at.strftime("%b %d, %Y")

    location = normalized_payload.get("location") or ""
    description = normalized_payload.get("description") or ""
    preview = location if location else (description[:200] if description else None)

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=summary,
            provider="google_calendar",
            provider_connection_id=provider_connection_id,
            payload=normalized_payload,
            raw=event,
            occurred_at=occurred_at,
        )
    )

    return {
        "id": normalized_payload.get("event_id"),
        "display_title": summary,
        "display_subtitle": display_subtitle,
        "preview": preview,
        "envelope": envelope,
        "metadata": {
            "calendar_id": calendar_id,
            "event_type": normalized_payload.get("event_type"),
            "is_all_day": normalized_payload.get("is_all_day"),
            "status": normalized_payload.get("status"),
        },
    }


async def list_google_calendar_event_start_events(
    user: User,
    provider_connection_id: int,
    options,
    _gcal_start_adapter: GoogleCalendarEventStartAdapter,  # pylint: disable=unused-argument # Reason: Interface consistency with other list_*_events functions
) -> Dict[str, Any]:
    """List Google Calendar events for browsing with event_start trigger payload format."""
    _, access_token = await get_oauth_token(
        user, connection_id=str(provider_connection_id)
    )

    calendar_id = (options.filter_params or {}).get("calendar_id", "primary")
    query = (options.filter_params or {}).get("query", "")
    page_size = min(options.page_size, CALENDAR_MAX_EVENTS)

    # For event_start browsing, show events from now through the next 48 hours
    now = datetime.now(timezone.utc)
    time_min = now
    time_max = now + timedelta(hours=48)

    params: Dict[str, Any] = {
        "maxResults": page_size,
        "singleEvents": True,
        "orderBy": "startTime",
        "timeMin": time_min.isoformat(),
        "timeMax": time_max.isoformat(),
    }
    if query:
        params["q"] = query
    if options.page_token:
        params["pageToken"] = options.page_token

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{CALENDAR_API_BASE}/calendars/{calendar_id}/events",
                headers=headers,
                params=params,
            )
            if resp.status_code >= 400:
                _handle_google_calendar_error(resp, calendar_id)

            data = resp.json()
            events = data.get("items", []) or []
            next_page_token = data.get("nextPageToken")

            if not events:
                return {
                    "items": [],
                    "next_page_token": None,
                    "trigger_key": options.trigger_key,
                    "supports_search": True,
                }

            items: List[Dict[str, Any]] = []
            for event in events:
                if event.get("status") == "cancelled":
                    continue
                items.append(_build_google_calendar_event_start_item(
                    event=event,
                    calendar_id=calendar_id,
                    trigger_key=options.trigger_key,
                    trigger_id=options.trigger_id,
                    provider_connection_id=provider_connection_id,
                ))

            return {
                "items": items,
                "next_page_token": next_page_token,
                "trigger_key": options.trigger_key,
                "supports_search": True,
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to list Google Calendar events for event_start")
        raise HTTPException(
            status_code=500, detail=f"Failed to fetch Calendar events: {str(exc)}"
        ) from exc


def _build_google_calendar_event_start_item(
    *,
    event: Dict[str, Any],
    calendar_id: str,
    trigger_key: str,
    trigger_id: str,
    provider_connection_id: int,
) -> Dict[str, Any]:
    """Build a TriggerEventItem from Google Calendar event data with event_start payload format."""
    normalized_payload = _normalize_calendar_event_common(event, calendar_id)

    # Add event_start-specific fields (matching what the polling adapter produces)
    start_info = event.get("start", {})
    start_dt_str = start_info.get("dateTime") or start_info.get("date") or ""
    start_dt = _parse_rfc3339(start_dt_str)
    event_start_iso = start_dt.isoformat() if start_dt else start_dt_str

    normalized_payload.update({
        "trigger_type": "before_start",
        "offset_minutes": -15,
        "trigger_time": event_start_iso,
        "event_start": event_start_iso,
    })

    summary = normalized_payload.get("summary") or "(No title)"
    occurred_at = _parse_rfc3339(event.get("updated")) or datetime.now(timezone.utc)

    if start_dt:
        display_subtitle = start_dt.strftime("%b %d, %Y %I:%M %p")
    else:
        display_subtitle = start_dt_str if start_dt_str else occurred_at.strftime("%b %d, %Y")

    location = normalized_payload.get("location") or ""
    description = normalized_payload.get("description") or ""
    preview = location if location else (description[:200] if description else None)

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=summary,
            provider="google_calendar",
            provider_connection_id=provider_connection_id,
            payload=normalized_payload,
            raw=event,
            occurred_at=occurred_at,
        )
    )

    return {
        "id": normalized_payload.get("event_id"),
        "display_title": summary,
        "display_subtitle": display_subtitle,
        "preview": preview,
        "envelope": envelope,
        "metadata": {
            "calendar_id": calendar_id,
            "is_all_day": normalized_payload.get("is_all_day"),
            "status": normalized_payload.get("status"),
        },
    }


def _handle_google_calendar_error(response: httpx.Response, calendar_id: str) -> None:
    """Handle Google Calendar API errors."""
    if response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Google Calendar authentication failed. Please reconnect your Google account.",
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail="Google Calendar access denied. Please ensure you have granted Calendar permissions.",
        )
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Calendar '{calendar_id}' not found or not accessible.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Google Calendar rate limit exceeded. Please try again later.",
        )
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Google Calendar API error: {response.text[:200]}",
    )


# =============================================================================
# Google Sheets Event Browsing
# =============================================================================

SHEETS_MAX_BROWSE_ROWS = 50


async def list_google_sheets_events(
    user: User,
    provider_connection_id: int,
    options,
    _gsheets_adapter: GoogleSheetsRowAddedAdapter,  # pylint: disable=unused-argument # Reason: Interface consistency with other list_*_events functions
) -> Dict[str, Any]:
    """List recent rows from a Google Sheet for browsing."""
    _, access_token = await get_oauth_token(
        user, connection_id=str(provider_connection_id)
    )

    filter_params = options.filter_params or {}
    spreadsheet_id = filter_params.get("spreadsheet_id")
    sheet_name = filter_params.get("sheet_name", DEFAULT_SHEET_NAME)

    if not spreadsheet_id:
        raise HTTPException(
            status_code=400,
            detail="spreadsheet_id is required in filter_params for Google Sheets trigger browsing.",
        )

    headers_http = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{SHEETS_API_BASE}/{spreadsheet_id}/values/{sheet_name}",
                headers=headers_http,
            )
            if resp.status_code >= 400:
                _handle_google_sheets_error(resp, spreadsheet_id)

            all_rows = resp.json().get("values", [])

            if not all_rows:
                return {
                    "items": [],
                    "next_page_token": None,
                    "trigger_key": options.trigger_key,
                    "supports_search": False,
                }

            # Use first row as headers, remaining as data rows
            column_headers = [str(cell) for cell in all_rows[0]]
            data_rows = all_rows[1:]

            # Take last N rows (newest at bottom), reversed for display
            start_idx = max(0, len(data_rows) - SHEETS_MAX_BROWSE_ROWS)
            browse_slice = list(enumerate(data_rows[start_idx:], start=start_idx))
            browse_slice.reverse()

            items: List[Dict[str, Any]] = [
                _build_google_sheets_event_item(
                    row=row,
                    row_number=idx + 2,  # +1 for 0-based index, +1 for header row
                    column_headers=column_headers,
                    spreadsheet_id=spreadsheet_id,
                    sheet_name=sheet_name,
                    provider_connection_id=provider_connection_id,
                    trigger_id=options.trigger_id,
                    trigger_key=options.trigger_key,
                )
                for idx, row in browse_slice
            ]

            return {
                "items": items,
                "next_page_token": None,
                "trigger_key": options.trigger_key,
                "supports_search": False,
            }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected Google Sheets browsing failure")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to browse Google Sheets events: {str(exc)[:200]}",
        ) from exc


def _build_google_sheets_event_item(  # pylint: disable=too-many-arguments # Reason: All params are distinct sheet-row context needed for envelope construction
    *,
    row: List[Any],
    row_number: int,
    column_headers: List[str],
    spreadsheet_id: str,
    sheet_name: str,
    provider_connection_id: int,
    trigger_id: str,
    trigger_key: str,
) -> Dict[str, Any]:
    """Build a single browsable event item from a sheet row."""
    fields = dict(zip_longest(column_headers, row, fillvalue=None)) if column_headers else {}
    display_title = str(row[0]) if row else f"Row {row_number}"
    display_subtitle = f"Row {row_number}"
    preview = ", ".join(str(cell) for cell in row[:5])[:200] if row else None

    payload = {
        "row_number": row_number,
        "spreadsheet_id": spreadsheet_id,
        "sheet_name": sheet_name,
        "fields": fields if column_headers else None,
        "row_values": row if not column_headers else None,
    }

    occurred_at = datetime.now(timezone.utc)
    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=display_title,
            provider="google",
            provider_connection_id=provider_connection_id,
            payload=payload,
            raw={"row_number": row_number, "values": row},
            occurred_at=occurred_at,
        )
    )

    return {
        "id": f"{spreadsheet_id}:{sheet_name}:{row_number}",
        "display_title": display_title,
        "display_subtitle": display_subtitle,
        "preview": preview,
        "envelope": envelope,
        "metadata": {
            "spreadsheet_id": spreadsheet_id,
            "sheet_name": sheet_name,
            "row_number": row_number,
        },
    }


def _handle_google_sheets_error(response: httpx.Response, spreadsheet_id: str) -> None:
    """Handle Google Sheets API errors."""
    if response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Google Sheets authentication failed. Please reconnect your Google account.",
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=403,
            detail="Google Sheets access denied. Please ensure you have granted Sheets permissions.",
        )
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail=f"Spreadsheet '{spreadsheet_id}' not found or not accessible.",
        )
    if response.status_code == 429:
        raise HTTPException(
            status_code=429,
            detail="Google Sheets rate limit exceeded. Please try again later.",
        )
    raise HTTPException(
        status_code=response.status_code,
        detail=f"Google Sheets API error: {response.text[:200]}",
    )
