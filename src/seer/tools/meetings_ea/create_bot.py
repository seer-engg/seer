"""meetingsEA tool: send a bot to join and record a meeting."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.create_bot")

# Maps user-friendly recording_mode to Attendee recording_settings
_RECORDING_MODE_MAP = {
    "speaker_view": {"format": "mp4", "view": "speaker_view"},
    "gallery_view": {"format": "mp4", "view": "gallery_view"},
    "audio_only": {"format": "mp3"},
}


class MeetingsEACreateBotTool(MeetingsEAToolBase):
    """Send a bot to join a video meeting and record/transcribe it."""

    name = "meetings_ea_create_bot"
    description = (
        "Send a meetingsEA bot to join a video meeting (Google Meet, Zoom, or Microsoft Teams). "
        "The bot will join the meeting, record audio/video, and transcribe the conversation."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "meeting_url": {
                    "type": "string",
                    "description": "The meeting URL (Google Meet, Zoom, or Microsoft Teams link).",
                },
                "bot_name": {
                    "type": "string",
                    "description": "Display name for the bot in the meeting.",
                    "default": "meetingsEA",
                },
                "recording_mode": {
                    "type": "string",
                    "enum": ["speaker_view", "gallery_view", "audio_only"],
                    "description": "Recording mode: speaker_view (default), gallery_view, or audio_only.",
                    "default": "speaker_view",
                },
                "transcription_enabled": {
                    "type": "boolean",
                    "description": "Whether to transcribe the meeting in real-time.",
                    "default": True,
                },
                "auto_leave_timeout_seconds": {
                    "type": "integer",
                    "description": "Automatically leave the meeting after this many seconds of no active participants (0 to disable).",
                    "default": 3600,
                    "minimum": 0,
                },
            },
            "required": ["meeting_url"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bot_id": {"type": "string", "description": "Unique bot identifier"},
                "state": {"type": "string", "description": "Current bot state"},
                "meeting_url": {"type": "string", "description": "The meeting URL"},
            },
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional["ResolvedCredentials"] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, credentials  # Platform API key, no OAuth

        meeting_url: str = arguments["meeting_url"]
        bot_name: str = arguments.get("bot_name", "meetingsEA")
        recording_mode: str = arguments.get("recording_mode", "speaker_view")
        _transcription_enabled: bool = arguments.get("transcription_enabled", True)  # noqa: F841  # Reserved for future use
        auto_leave_timeout: int = arguments.get("auto_leave_timeout_seconds", 3600)

        # Build Attendee API request body
        body: Dict[str, Any] = {
            "meeting_url": meeting_url,
            "bot_name": bot_name,
        }

        # Recording settings
        if recording_mode in _RECORDING_MODE_MAP:
            body["recording_settings"] = _RECORDING_MODE_MAP[recording_mode]

        # Transcription — omit to let Attendee auto-detect per platform

        # Auto-leave settings
        if auto_leave_timeout > 0:
            body["automatic_leave_settings"] = {
                "only_participant_in_meeting_timeout_seconds": auto_leave_timeout,
            }

        # Tag with org ID for isolation
        metadata: Dict[str, Any] = {}
        if context and context.organization_id is not None:
            metadata["seer_org_id"] = str(context.organization_id)
        body["metadata"] = metadata

        # Register webhook for real-time state updates (must be HTTPS for Attendee)
        webhook_url = self._get_webhook_url()
        if webhook_url and webhook_url.startswith("https://"):
            body["webhooks"] = [
                {
                    "url": webhook_url,
                    "triggers": ["bot.state_change", "transcript.update"],
                }
            ]

        result = await self._api_request("POST", "/api/v1/bots", json=body)

        return {
            "bot_id": result.get("id"),
            "state": result.get("state"),
            "meeting_url": result.get("meeting_url", meeting_url),
        }


__all__ = ["MeetingsEACreateBotTool"]
