"""
Trigger Event Browser for browsing real events from connected accounts.

This service allows users to browse actual trigger events (like Gmail emails,
Discord messages, webhooks, forms) from their connected accounts or stored
events for workflow testing, instead of using hardcoded sample data.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.core.triggers.events import TriggerEventEnvelopeInput, build_event_envelope
from seer.core.triggers.polling.adapters.gmail_email_received import GmailEmailReceivedAdapter
from seer.core.triggers.polling.adapters.slack_message_received import SlackMessageReceivedAdapter
from seer.database import User, TriggerEvent, TriggerSubscription
from seer.logger import get_logger
from seer.services.integrations.trigger_event_polling import (
    list_discord_events,
    list_gmail_events,
    list_slack_events,
)

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


@dataclass
class TriggerBrowsingConfig:
    """Configuration for how a trigger type supports event browsing."""

    provider: str
    mode: str  # "polling" (live API) or "persisted" (database)
    supports_search: bool = False


# Configuration for trigger event browsing
TRIGGER_BROWSING_CONFIG: Dict[str, TriggerBrowsingConfig] = {
    # Polling triggers - fetch live from external APIs
    "poll.gmail.email_received": TriggerBrowsingConfig("google", "polling", True),
    "poll.discord.message_received": TriggerBrowsingConfig("discord", "polling", False),
    "poll.slack.message_received": TriggerBrowsingConfig("slack", "polling", False),
    # Persisted triggers - query from TriggerEvent database table
    "webhook.generic": TriggerBrowsingConfig("generic", "persisted", False),
    "webhook.supabase.db_changes": TriggerBrowsingConfig("supabase", "persisted", False),
    "form.hosted": TriggerBrowsingConfig("form", "persisted", False),
    "form.hitl": TriggerBrowsingConfig("form", "persisted", False),
}

# Map of trigger keys to their providers (for backward compatibility)
TRIGGER_PROVIDER_MAP: Dict[str, str] = {
    key: cfg.provider for key, cfg in TRIGGER_BROWSING_CONFIG.items()
}


class TriggerEventBrowser:
    """
    Browse real trigger events from connected accounts or stored events.

    Supports two browsing modes:
    1. Polling triggers - fetch live events from external APIs:
       - Gmail emails (poll.gmail.email_received)
       - Discord messages (poll.discord.message_received)

    2. Persisted triggers - query stored events from TriggerEvent table:
       - Generic webhooks (webhook.generic)
       - Supabase DB changes (webhook.supabase.db_changes)
       - Hosted forms (form.hosted)
       - HITL forms (form.hitl)

    Uses existing PollAdapter logic to normalize event payloads.
    """

    def __init__(self, user: User):
        self.user = user
        self._gmail_adapter = GmailEmailReceivedAdapter()
        self._slack_adapter = SlackMessageReceivedAdapter()

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
        provider_connection_id: Optional[int] = None,
        subscription_id: Optional[int] = None,
        options: Optional[TriggerEventListOptions] = None,
    ) -> Dict[str, Any]:
        """
        List trigger events from a connected account or database.

        Args:
            provider_connection_id: ID of the OAuthConnection to use (for polling triggers)
            subscription_id: ID of the TriggerSubscription (for persisted triggers)
            options: Listing options (trigger_key, pagination, filters)

        Returns:
            {
                "items": List[TriggerEventItem as dict],
                "next_page_token": Optional[str],
                "trigger_key": str,
                "supports_search": bool,
            }
        """
        if options is None:
            raise ValueError("options is required")

        trigger_key = options.trigger_key
        browsing_config = TRIGGER_BROWSING_CONFIG.get(trigger_key)

        if browsing_config is None:
            raise ValueError(f"Trigger key '{trigger_key}' does not support event browsing")

        # Route based on browsing mode
        if browsing_config.mode == "polling":
            if provider_connection_id is None:
                raise ValueError(f"provider_connection_id is required for polling trigger '{trigger_key}'")
            return await self._dispatch_polling_trigger(trigger_key, provider_connection_id, options)

        if browsing_config.mode == "persisted":
            if subscription_id is None:
                raise ValueError(f"subscription_id is required for persisted trigger '{trigger_key}'")
            return await self._list_persisted_events(subscription_id, options)

        raise ValueError(f"Unknown browsing mode for trigger '{trigger_key}'")

    async def _dispatch_polling_trigger(
        self,
        trigger_key: str,
        provider_connection_id: int,
        options: TriggerEventListOptions,
    ) -> Dict[str, Any]:
        """Dispatch to the correct polling browser based on trigger key."""
        if trigger_key == "poll.gmail.email_received":
            return await list_gmail_events(self.user, provider_connection_id, options, self._gmail_adapter)
        if trigger_key == "poll.discord.message_received":
            return await list_discord_events(options)
        if trigger_key == "poll.slack.message_received":
            return await list_slack_events(self.user, provider_connection_id, options, self._slack_adapter)
        raise ValueError(f"Polling trigger '{trigger_key}' is not yet implemented")

    # =========================================================================
    # Persisted Events - Query from TriggerEvent database table
    # =========================================================================

    async def _list_persisted_events(
        self,
        subscription_id: int,
        options: TriggerEventListOptions,
    ) -> Dict[str, Any]:
        """
        List persisted trigger events from the database.

        Used for webhooks, forms, and other triggers that store events in the
        TriggerEvent table rather than fetching from external APIs.
        """
        # Verify subscription exists and belongs to user
        subscription = await TriggerSubscription.get_or_none(
            id=subscription_id, user=self.user
        )
        if not subscription:
            raise HTTPException(
                status_code=404,
                detail=f"Subscription {subscription_id} not found",
            )

        page_size = min(options.page_size, 50)

        # Parse offset from page_token
        offset = 0
        if options.page_token:
            try:
                offset = int(options.page_token)
            except ValueError:
                offset = 0

        # Query events
        events = await TriggerEvent.filter(
            subscription_id=subscription_id
        ).order_by("-received_at").offset(offset).limit(page_size + 1)

        # Check if there are more
        has_more = len(events) > page_size
        events = events[:page_size]

        # Build items
        items: List[Dict[str, Any]] = []
        for event in events:
            items.append(self._build_persisted_event_item(
                event=event,
                trigger_key=options.trigger_key,
                trigger_id=options.trigger_id,
            ))

        # Calculate next_page_token
        next_page_token = str(offset + page_size) if has_more else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "trigger_key": options.trigger_key,
            "supports_search": False,
        }

    def _build_persisted_event_item(
        self,
        event: TriggerEvent,
        trigger_key: str,
        trigger_id: str,
    ) -> Dict[str, Any]:
        """Build a TriggerEventItem from a persisted TriggerEvent."""
        event_data = event.event or {}

        # Determine display based on trigger type
        if trigger_key.startswith("webhook."):
            display_title, preview, provider = _extract_webhook_display(event_data, trigger_key)
        elif trigger_key.startswith("form."):
            display_title, preview, provider = _extract_form_display(event_data, trigger_key)
        else:
            display_title = f"Event #{event.id}"
            preview = str(event_data)[:200] if event_data else None
            provider = "generic"

        return _build_persisted_event_result(
            event=event,
            display_title=display_title,
            preview=preview,
            trigger_key=trigger_key,
            trigger_id=trigger_id,
            provider=provider,
        )


def _extract_webhook_display(event_data: Dict[str, Any], trigger_key: str) -> tuple[str, Optional[str], str]:
    """Extract display title and preview for webhook events."""
    display_title = "Webhook Event"
    preview = None

    if "type" in event_data:
        display_title = f"Webhook: {event_data['type']}"
    elif "event" in event_data:
        display_title = f"Webhook: {event_data['event']}"
    elif "action" in event_data:
        display_title = f"Webhook: {event_data['action']}"

    if trigger_key == "webhook.supabase.db_changes":
        table = event_data.get("table", "")
        op_type = event_data.get("type", "")
        display_title = f"Supabase: {op_type} on {table}" if table else f"Supabase: {op_type}"
        record = event_data.get("record", {})
        if record:
            preview = str(record)[:200]

    return display_title, preview, "webhook"


def _extract_form_display(event_data: Dict[str, Any], trigger_key: str) -> tuple[str, Optional[str], str]:
    """Extract display title and preview for form events."""
    display_title = "Form Submission"
    preview = None

    if trigger_key == "form.hitl":
        display_title = "HITL Response"
        response = event_data.get("response", event_data.get("answer", ""))
        if response:
            preview = str(response)[:200]
    else:
        form_fields = list(event_data.keys())
        if form_fields:
            first_field = form_fields[0]
            first_value = str(event_data[first_field])[:50]
            display_title = f"Form: {first_field}={first_value}..."
            preview = str(event_data)[:200]

    return display_title, preview, "form"


def _build_persisted_event_result(
    *,
    event: TriggerEvent,
    display_title: str,
    preview: Optional[str],
    trigger_key: str,
    trigger_id: str,
    provider: str,
) -> Dict[str, Any]:
    """Build a standard event result dict with envelope for a persisted event."""
    received_at = event.received_at or datetime.now(timezone.utc)
    display_subtitle = received_at.strftime("%b %d, %Y %I:%M %p")

    envelope = build_event_envelope(
        TriggerEventEnvelopeInput(
            trigger_id=trigger_id,
            trigger_key=trigger_key,
            title=display_title,
            provider=provider,
            provider_connection_id=None,
            payload=event.event or {},
            raw=event.raw_payload or {},
            occurred_at=event.occurred_at or received_at,
        )
    )

    return {
        "id": str(event.id),
        "display_title": display_title,
        "display_subtitle": display_subtitle,
        "preview": preview,
        "envelope": envelope,
        "metadata": {
            "event_id": event.id,
            "received_at": received_at.isoformat(),
        },
    }


__all__ = [
    "TriggerEventBrowser",
    "TriggerEventItem",
    "TriggerEventListOptions",
    "TriggerBrowsingConfig",
    "TRIGGER_BROWSING_CONFIG",
    "TRIGGER_PROVIDER_MAP",
]
