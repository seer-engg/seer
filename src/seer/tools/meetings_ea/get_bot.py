"""meetingsEA tool: get bot status and details."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.get_bot")


class MeetingsEAGetBotTool(MeetingsEAToolBase):
    """Get the current status and details of a meetingsEA bot."""

    name = "meetings_ea_get_bot"
    description = (
        "Get the current status and details of a meetingsEA bot, "
        "including its state, meeting URL, and event history."
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
                "bot_id": {"type": "string"},
                "state": {"type": "string", "description": "Current bot state (e.g., joining, joined_recording, ended)"},
                "meeting_url": {"type": "string"},
                "events": {
                    "type": "array",
                    "description": "State transition history",
                    "items": {"type": "object"},
                },
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
        result = await self._api_request("GET", f"/api/v1/bots/{bot_id}")

        # Verify org ownership
        if context and context.organization_id is not None:
            bot_metadata = result.get("metadata") or {}
            bot_org_id = bot_metadata.get("seer_org_id")
            if bot_org_id and bot_org_id != str(context.organization_id):
                raise HTTPException(status_code=404, detail="Bot not found")

        return {
            "bot_id": result.get("id"),
            "state": result.get("state"),
            "meeting_url": result.get("meeting_url"),
            "events": result.get("events", []),
        }


__all__ = ["MeetingsEAGetBotTool"]
