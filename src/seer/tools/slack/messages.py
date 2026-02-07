"""
Slack message operations - sending channel messages and direct messages.
"""

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.slack.base import SlackAPIClient

logger = get_logger("shared.tools.slack.messages")


class SlackSendChannelMessageTool(SlackAPIClient):
    """Send a message to a Slack channel."""

    name = "slack_send_channel_message"
    description = "Send a message to a Slack channel. Bot must be added to the channel."
    required_scopes = ["chat:write", "channels:read"]
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
                    "description": "Slack channel ID (e.g., C01234567890)",
                },
                "text": {
                    "type": "string",
                    "description": "Message text (supports Slack markdown formatting)",
                    "maxLength": 40000,
                },
                "thread_ts": {
                    "type": "string",
                    "description": "Thread timestamp to reply to (optional)",
                },
                "unfurl_links": {
                    "type": "boolean",
                    "description": "Enable unfurling of links in message",
                    "default": True,
                },
            },
            "required": ["workspace_id", "channel_id", "text"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for workspace_id and channel_id parameters."""
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
                "channel": {"type": "string", "description": "Channel ID where message was sent"},
                "ts": {"type": "string", "description": "Message timestamp (unique ID)"},
                "message": {"type": "object", "description": "Message object with text, user, etc."},
            },
            "required": ["ok", "channel", "ts"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        channel_id = str(arguments.get("channel_id") or "")
        text = arguments.get("text")
        thread_ts = arguments.get("thread_ts")
        unfurl_links = arguments.get("unfurl_links", True)

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")
        if not channel_id:
            raise HTTPException(status_code=400, detail="Parameter 'channel_id' is required")
        if not text:
            raise HTTPException(status_code=400, detail="Parameter 'text' is required")

        logger.info(
            "Sending Slack channel message: workspace_id=%s, channel_id=%s",
            workspace_id,
            channel_id
        )

        body: Dict[str, Any] = {
            "channel": channel_id,
            "text": text,
            "unfurl_links": unfurl_links,
        }
        if thread_ts:
            body["thread_ts"] = thread_ts

        return await self._make_request(
            "POST",
            "chat.postMessage",
            credentials=credentials,
            json_body=body
        )


class SlackSendDirectMessageTool(SlackAPIClient):
    """Send a direct message to a Slack user."""

    name = "slack_send_direct_message"
    description = "Send a direct message (DM) to a Slack user."
    required_scopes = ["chat:write", "im:write", "users:read"]
    integration_type = "slack"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Slack workspace (team) ID",
                },
                "user_id": {
                    "type": "string",
                    "description": "Slack user ID to send the message to (e.g., U01234567890)",
                },
                "text": {
                    "type": "string",
                    "description": "Message text (supports Slack markdown formatting)",
                    "maxLength": 40000,
                },
            },
            "required": ["workspace_id", "user_id", "text"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for workspace_id and user_id parameters."""
        return {
            "workspace_id": {
                "resource_type": "workspace",
                "display_field": "name",
                "value_field": "resource_id",
                "search_enabled": True,
                "filter": {"provider": "slack", "resource_type": "workspace"},
            },
            "user_id": {
                "resource_type": "user",
                "display_field": "real_name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "workspace_id",
                "filter": {"provider": "slack", "resource_type": "user"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean", "description": "Success status"},
                "channel": {"type": "string", "description": "DM channel ID"},
                "ts": {"type": "string", "description": "Message timestamp"},
                "message": {"type": "object", "description": "Message object"},
            },
            "required": ["ok", "channel", "ts"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        user_id = str(arguments.get("user_id") or "")
        text = arguments.get("text")

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="Parameter 'user_id' is required")
        if not text:
            raise HTTPException(status_code=400, detail="Parameter 'text' is required")

        logger.info("Sending Slack DM to user_id=%s", user_id)

        # Step 1: Open/get DM channel with user
        open_resp = await self._make_request(
            "POST",
            "conversations.open",
            credentials=credentials,
            json_body={"users": user_id}
        )
        dm_channel_id = open_resp.get("channel", {}).get("id")

        if not dm_channel_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to open DM channel: no channel ID in response"
            )

        # Step 2: Send message to DM channel
        return await self._make_request(
            "POST",
            "chat.postMessage",
            credentials=credentials,
            json_body={
                "channel": dm_channel_id,
                "text": text,
            }
        )
