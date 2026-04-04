"""meetingsEA tool: list bots with optional filters."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

from seer.logger import get_logger
from seer.tools.meetings_ea.base import MeetingsEAToolBase

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.tools.credential_resolver import ResolvedCredentials

logger = get_logger("tools.meetings_ea.list_bots")


class MeetingsEAListBotsTool(MeetingsEAToolBase):
    """List meetingsEA bots, optionally filtering by meeting URL or status."""

    name = "meetings_ea_list_bots"
    description = (
        "List meetingsEA bots for your organization. "
        "Optionally filter by meeting URL or bot status."
    )

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "meeting_url": {
                    "type": "string",
                    "description": "Filter by meeting URL.",
                },
                "status": {
                    "type": "string",
                    "description": "Filter by bot state (e.g., joining, joined_recording, ended).",
                },
            },
            "required": [],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "bots": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "bot_id": {"type": "string"},
                            "state": {"type": "string"},
                            "meeting_url": {"type": "string"},
                        },
                    },
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

        params: Dict[str, str] = {}
        if arguments.get("meeting_url"):
            params["meeting_url"] = arguments["meeting_url"]
        if arguments.get("status"):
            params["states"] = arguments["status"]

        result = await self._api_request("GET", "/api/v1/bots", params=params or None)

        # Attendee returns a list or paginated result
        bots_raw = result if isinstance(result, list) else result.get("results", result.get("bots", []))

        # Filter by org
        org_id = str(context.organization_id) if context and context.organization_id is not None else None
        bots = []
        for bot in bots_raw:
            if org_id:
                bot_metadata = bot.get("metadata") or {}
                if bot_metadata.get("seer_org_id") != org_id:
                    continue
            bots.append({
                "bot_id": bot.get("id"),
                "state": bot.get("state"),
                "meeting_url": bot.get("meeting_url"),
            })

        return {"bots": bots}


__all__ = ["MeetingsEAListBotsTool"]
