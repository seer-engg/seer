"""
Slack reaction operations - adding emoji reactions to messages.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.slack.base import SlackAPIClient

logger = get_logger("shared.tools.slack.reactions")


class SlackAddReactionTool(SlackAPIClient):
    """Add an emoji reaction to a Slack message."""

    name = "slack_add_reaction"
    description = "Add an emoji reaction to a message in Slack."
    required_scopes = ["reactions:write"]
    integration_type = "slack"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Slack workspace (team) ID",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Slack channel ID where the message is located",
                },
                "timestamp": {
                    "type": "string",
                    "description": "Message timestamp (ts) to react to",
                },
                "emoji": {
                    "type": "string",
                    "description": "Emoji name without colons (e.g., 'thumbsup', 'heart', 'rocket')",
                },
            },
            "required": ["workspace_id", "channel_id", "timestamp", "emoji"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "workspace_id": {
                "resource_type": "workspace",
                "display_field": "name",
                "value_field": "resource_id",
                "search_enabled": True,
                "filter": {"provider": "slack", "resource_type": "workspace"},
            },
            "channel_id": {
                "resource_type": "channel",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "workspace_id",
                "filter": {"provider": "slack", "resource_type": "channel"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean", "description": "Success status"},
            },
            "required": ["ok"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        channel_id = str(arguments.get("channel_id") or "")
        timestamp = str(arguments.get("timestamp") or "")
        emoji = str(arguments.get("emoji") or "").strip(":")  # Remove colons if present

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")
        if not channel_id:
            raise HTTPException(status_code=400, detail="Parameter 'channel_id' is required")
        if not timestamp:
            raise HTTPException(status_code=400, detail="Parameter 'timestamp' is required")
        if not emoji:
            raise HTTPException(status_code=400, detail="Parameter 'emoji' is required")

        logger.info(
            "Adding Slack reaction: channel_id=%s, ts=%s, emoji=%s",
            channel_id,
            timestamp,
            emoji
        )

        return await self._make_request(
            "POST",
            "reactions.add",
            credentials=credentials,
            json_body={
                "channel": channel_id,
                "timestamp": timestamp,
                "name": emoji,
            }
        )
