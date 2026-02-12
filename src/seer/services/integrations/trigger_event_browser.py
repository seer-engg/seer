"""
Trigger Event Browser for browsing real events from connected accounts.

This service allows users to browse actual trigger events (like Gmail emails)
from their connected accounts for workflow testing, instead of using hardcoded
sample data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope
from seer.core.triggers.polling.adapters.gmail_email_received import (
    GmailEmailReceivedAdapter,
    MAX_MESSAGES_PER_POLL,
)
from seer.database import User
from seer.logger import get_logger
from seer.tools.google.gmail.helpers import GMAIL_API_BASE, build_gmail_list_params
from seer.tools.oauth_manager import get_oauth_token

logger = get_logger(__name__)


@dataclass
class TriggerEventItem:
    """A browsable trigger event that can be selected for workflow testing."""

    id: str  # Unique event identifier (e.g., Gmail message_id)
    display_title: str  # Human-readable title (e.g., "Subject - from sender@email.com")
    display_subtitle: Optional[str]  # Secondary info (e.g., timestamp)
    preview: Optional[str]  # Preview text (e.g., email snippet)
    envelope: Dict[str, Any]  # Full trigger envelope ready for workflow execution
    metadata: Dict[str, Any]  # Additional metadata for display


@dataclass
class TriggerEventListOptions:
    """Options for listing trigger events."""

    trigger_key: str
    trigger_id: str = "trigger_test"  # Default trigger_id for testing
    page_size: int = 25
    page_token: Optional[str] = None  # For pagination (page token or offset)
    filter_params: Optional[Dict[str, Any]] = None  # Trigger-specific filters


# Map of trigger keys to their providers
TRIGGER_PROVIDER_MAP: Dict[str, str] = {
    "poll.gmail.email_received": "google",
    "poll.discord.message_received": "discord",
}


class TriggerEventBrowser:
    """
    Browse real trigger events from connected accounts.

    Supports:
    - Gmail emails (poll.gmail.email_received)
    - Discord messages (poll.discord.message_received) - future

    Uses existing PollAdapter logic to normalize event payloads.
    """

    def __init__(self, user: User):
        self.user = user
        self._gmail_adapter = GmailEmailReceivedAdapter()

    @staticmethod
    def get_supported_trigger_keys() -> List[str]:
        """Return list of trigger keys that support event browsing."""
        return list(TRIGGER_PROVIDER_MAP.keys())

    @staticmethod
    def get_provider_for_trigger(trigger_key: str) -> Optional[str]:
        """Get the OAuth provider for a trigger key."""
        return TRIGGER_PROVIDER_MAP.get(trigger_key)

    async def list_events(
        self,
        provider_connection_id: int,
        options: TriggerEventListOptions,
    ) -> Dict[str, Any]:
        """
        List trigger events from a connected account.

        Args:
            provider_connection_id: ID of the OAuthConnection to use
            options: Listing options (trigger_key, pagination, filters)

        Returns:
            {
                "items": List[TriggerEventItem as dict],
                "next_page_token": Optional[str],
                "trigger_key": str,
                "supports_search": bool,
            }
        """
        trigger_key = options.trigger_key

        if trigger_key == "poll.gmail.email_received":
            return await self._list_gmail_events(provider_connection_id, options)
        # Add more trigger types here as needed
        # elif trigger_key == "poll.discord.message_received":
        #     return await self._list_discord_events(provider_connection_id, options)

        raise ValueError(f"Trigger key '{trigger_key}' does not support event browsing")

    async def _list_gmail_events(
        self,
        provider_connection_id: int,
        options: TriggerEventListOptions,
    ) -> Dict[str, Any]:
        """List Gmail emails for browsing."""
        # Get OAuth token
        _, access_token = await get_oauth_token(
            self.user, connection_id=str(provider_connection_id)
        )

        # Parse filter params (inline to reduce local variable count)
        label_ids = (options.filter_params or {}).get("label_ids", ["INBOX"])
        query = (options.filter_params or {}).get("query", "")
        page_size = min(options.page_size, MAX_MESSAGES_PER_POLL)

        # Build Gmail API params
        params = build_gmail_list_params(page_size, label_ids, query)
        if options.page_token:
            params["pageToken"] = options.page_token

        headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # List messages
                list_resp = await client.get(
                    f"{GMAIL_API_BASE}/messages", headers=headers, params=params
                )
                if list_resp.status_code >= 400:
                    self._handle_gmail_error(list_resp)

                list_json = list_resp.json()
                messages, next_page_token = list_json.get("messages", []) or [], list_json.get("nextPageToken")

                if not messages:
                    return {
                        "items": [],
                        "next_page_token": None,
                        "trigger_key": options.trigger_key,
                        "supports_search": True,
                    }

                # Fetch metadata for each message
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

                    items.append(self._build_gmail_event_item(
                        msg_data=msg_resp.json(),
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
            logger.exception("Failed to list Gmail events")
            raise HTTPException(
                status_code=500, detail=f"Failed to fetch Gmail messages: {str(exc)}"
            ) from exc

    def _build_gmail_event_item(
        self,
        msg_data: Dict[str, Any],
        trigger_key: str,
        trigger_id: str,
        provider_connection_id: int,
    ) -> Dict[str, Any]:
        """Build a TriggerEventItem from Gmail message data."""
        # Reuse adapter's normalization logic
        normalized_payload = self._gmail_adapter._normalize_message(msg_data)  # pylint: disable=protected-access # Reason: Reusing adapter's message normalization

        # Extract display fields
        subject = normalized_payload.get("subject") or "(No subject)"
        from_info = normalized_payload.get("from", {})
        from_display = from_info.get("name") or from_info.get("email") or "Unknown"
        snippet = msg_data.get("snippet", "")
        internal_date_ms = int(msg_data.get("internalDate") or 0)

        # Format timestamp
        occurred_at = datetime.fromtimestamp(internal_date_ms / 1000, tz=timezone.utc)
        display_subtitle = occurred_at.strftime("%b %d, %Y %I:%M %p")

        # Build full trigger envelope
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

    def _handle_gmail_error(self, response: httpx.Response) -> None:
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


__all__ = [
    "TriggerEventBrowser",
    "TriggerEventItem",
    "TriggerEventListOptions",
    "TRIGGER_PROVIDER_MAP",
]
