"""Polling adapters used by the trigger engine."""

from . import airtable_new_record_in_view  # noqa: F401
from . import cron_schedule  # noqa: F401
from . import discord_message_received  # noqa: F401
from . import gmail_email_received  # noqa: F401
from . import google_calendar_event_changed  # noqa: F401
from . import google_calendar_event_start  # noqa: F401
from . import slack_message_received  # noqa: F401

__all__ = [
    "airtable_new_record_in_view",
    "cron_schedule",
    "discord_message_received",
    "gmail_email_received",
    "google_calendar_event_changed",
    "google_calendar_event_start",
    "slack_message_received",
]
