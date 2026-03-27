"""
Slack message operations - sending channel messages and direct messages.
"""
# pylint: disable=duplicate-code  # Reason: Resource picker patterns are intentionally shared across Slack tools

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.slack.base import SlackAPIClient
from seer.utils.rich_text import Platform, RichTextConverter

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.slack.messages")

# Shared converter instance for Slack tools
_rich_text_converter = RichTextConverter()


class SlackSendChannelMessageTool(SlackAPIClient):
    """Send a message to a Slack channel."""

    name = "slack_send_channel_message"
    description = "Send a message to a Slack channel. If the bot isn't in the channel, use slack_join_channel first."
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
                    "description": "Message text (supports formatting: bold, italic, strikethrough, links, lists, quotes, code)",
                    "maxLength": 40000,
                    "x-ui-type": "rich_text",
                    "x-rich-text-output": "markdown",
                    "x-rich-text-features": [
                        "bold", "italic", "strikethrough", "link",
                        "bulletList", "orderedList", "blockquote", "code", "codeBlock",
                    ],
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
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context  # unused but required for interface consistency
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

        # Convert markdown to Slack mrkdwn format
        formatted_text = _rich_text_converter.convert(text, Platform.SLACK)

        body: Dict[str, Any] = {
            "channel": channel_id,
            "text": formatted_text,
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
                    "description": "Message text (supports formatting: bold, italic, strikethrough, links, lists, quotes, code)",
                    "maxLength": 40000,
                    "x-ui-type": "rich_text",
                    "x-rich-text-output": "markdown",
                    "x-rich-text-features": [
                        "bold", "italic", "strikethrough", "link",
                        "bulletList", "orderedList", "blockquote", "code", "codeBlock",
                    ],
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
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        _ = access_token, context  # unused but required for interface consistency
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

        # Convert markdown to Slack mrkdwn format
        formatted_text = _rich_text_converter.convert(text, Platform.SLACK)

        # Step 2: Send message to DM channel
        return await self._make_request(
            "POST",
            "chat.postMessage",
            credentials=credentials,
            json_body={
                "channel": dm_channel_id,
                "text": formatted_text,
            }
        )
