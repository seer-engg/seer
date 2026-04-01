"""meetingsEA tool: tell bot to leave meeting."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.leave_meeting")


class MeetingsEALeaveMeetingTool(MeetingsEAToolBase):
    """Tell a meetingsEA bot to leave the meeting."""

    name = "meetings_ea_leave_meeting"
    description = "Tell a meetingsEA bot to leave the meeting it is currently in."

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
                "state": {"type": "string", "description": "Bot state after leave command"},
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

        # Verify org ownership
        if context and context.organization_id is not None:
            bot = await self._api_request("GET", f"/api/v1/bots/{bot_id}")
            bot_metadata = bot.get("metadata") or {}
            bot_org_id = bot_metadata.get("seer_org_id")
            if bot_org_id and bot_org_id != str(context.organization_id):
                raise HTTPException(status_code=404, detail="Bot not found")

        result = await self._api_request("POST", f"/api/v1/bots/{bot_id}/leave")

        return {
            "bot_id": result.get("id", bot_id),
            "state": result.get("state", "leaving"),
        }


__all__ = ["MeetingsEALeaveMeetingTool"]
