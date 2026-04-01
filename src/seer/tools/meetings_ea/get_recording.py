"""meetingsEA tool: get recording download URL."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.get_recording")


class MeetingsEAGetRecordingTool(MeetingsEAToolBase):
    """Get a download URL for a meeting recording."""

    name = "meetings_ea_get_recording"
    description = (
        "Get a download URL for a meetingsEA meeting recording. "
        "The URL is short-lived (typically valid for ~30 minutes)."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bot_id": {
                    "type": "string",
                    "description": "The bot identifier returned by meetings_ea_create_bot.",
                },
            },
            "required": ["bot_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "Short-lived download URL for the recording"},
                "state": {"type": "string", "description": "Recording state (e.g., complete, in_progress)"},
                "recording_type": {"type": "string", "description": "Type of recording (audio_and_video, audio_only)"},
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
        _ = access_token, credentials

        bot_id: str = arguments["bot_id"]

        # Verify org ownership by fetching the bot first
        if context and context.organization_id is not None:
            bot = await self._api_request("GET", f"/api/v1/bots/{bot_id}")
            bot_metadata = bot.get("metadata") or {}
            bot_org_id = bot_metadata.get("seer_org_id")
            if bot_org_id and bot_org_id != str(context.organization_id):
                raise HTTPException(status_code=404, detail="Bot not found")

        result = await self._api_request("GET", f"/api/v1/bots/{bot_id}/recording")

        return {
            "url": result.get("url"),
            "state": result.get("state"),
            "recording_type": result.get("recording_type"),
        }


__all__ = ["MeetingsEAGetRecordingTool"]
