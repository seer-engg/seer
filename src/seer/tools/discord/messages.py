"""
Discord message operations - sending channel messages and direct messages.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.discord.base import DISCORD_API_BASE, DiscordAPIClient
from seer.utils.rich_text import Platform, RichTextConverter

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.discord.messages")

# Shared converter instance for Discord tools
_rich_text_converter = RichTextConverter()


class DiscordSendChannelMessageTool(DiscordAPIClient):
    """Send a message to a Discord channel."""

    name = "discord_send_channel_message"
    description = "Send a message to a Discord channel in a server. Requires bot to be installed in the server."
    required_scopes = ["bot"]
    integration_type = "discord"

    # Required permissions: VIEW_CHANNEL (1024) | SEND_MESSAGES (2048) = 3072
    required_permissions = 3072

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "guild_id": {
                    "type": "string",
                    "description": "Discord server (guild) ID where the channel is located",
                },
                "channel_id": {
                    "type": "string",
                    "description": "Discord channel ID to send the message to",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Message content (supports formatting: bold, italic, underline,"
                        " strikethrough, lists, quotes, code). Max 2000 characters."
                    ),
                    "maxLength": 2000,
                    "x-ui-type": "rich_text",
                    "x-rich-text-platform": "discord",
                },
                "embed": {
                    "type": "object",
                    "description": "Optional rich embed object. See Discord API documentation for embed structure.",
                    "default": None,
                },
            },
            "required": ["guild_id", "channel_id"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        """Enable resource picker for guild_id and channel_id parameters."""
        return {
            "guild_id": {
                "resource_type": "guild",
                "display_field": "name",
                "value_field": "resource_id",  # guild_id is stored in resource_id
                "search_enabled": True,
                "filter": {"provider": "discord", "resource_type": "guild"},
            },
            "channel_id": {
                "resource_type": "channel",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "depends_on": "guild_id",  # Channels depend on guild selection
                "filter": {"provider": "discord", "resource_type": "channel"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Message ID"},
                "channel_id": {"type": "string", "description": "Channel ID where message was sent"},
                "guild_id": {"type": ["string", "null"], "description": "Guild ID (if sent in a guild)"},
                "author": {"type": "object", "description": "Message author information"},
                "content": {"type": ["string", "null"], "description": "Message content (null if only embed)"},
                "timestamp": {"type": "string", "description": "ISO8601 timestamp"},
                "edited_timestamp": {"type": ["string", "null"], "description": "ISO8601 timestamp if edited"},
                "tts": {"type": "boolean", "description": "Whether this was a TTS message"},
                "mention_everyone": {"type": "boolean", "description": "Whether @everyone was mentioned"},
                "mentions": {"type": "array", "description": "Array of user objects mentioned"},
                "attachments": {"type": "array", "description": "Array of attachment objects"},
                "embeds": {"type": "array", "description": "Array of embed objects"},
            },
            "required": ["id", "channel_id", "timestamp"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        # access_token and context unused but required for interface consistency
        _ = access_token, context
        guild_id = str(arguments.get("guild_id") or "")
        channel_id = str(arguments.get("channel_id") or "")
        content = arguments.get("content")
        embed = arguments.get("embed")

        if not guild_id:
            raise HTTPException(status_code=400, detail="Parameter 'guild_id' is required")
        if not channel_id:
            raise HTTPException(status_code=400, detail="Parameter 'channel_id' is required")
        if not content and not embed:
            raise HTTPException(
                status_code=400,
                detail="Either 'content' or 'embed' must be provided"
            )

        # Validate content length
        if content and len(content) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Message content cannot exceed 2000 characters"
            )

        # Build request body with formatted content
        body: Dict[str, Any] = {}
        if content:
            # Convert markdown to Discord format
            formatted_content = _rich_text_converter.convert(str(content), Platform.DISCORD)
            body["content"] = formatted_content
        if embed:
            body["embed"] = embed

        logger.info(
            "Sending Discord channel message: guild_id=%s, channel_id=%s",
            guild_id,
            channel_id
        )

        # Send message to channel
        resp = await self._make_request(
            "POST",
            f"{DISCORD_API_BASE}/channels/{channel_id}/messages",
            credentials=credentials,
            json_body=body
        )

        return resp.json()


class DiscordSendDirectMessageTool(DiscordAPIClient):
    """Send a direct message to a Discord user."""

    name = "discord_send_direct_message"
    description = "Send a direct message (DM) to a Discord user. The bot must share a server with the user."
    required_scopes = ["bot"]
    integration_type = "discord"

    # Required permissions: SEND_MESSAGES (2048) only
    required_permissions = 2048

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "string",
                    "description": "Discord user ID to send the message to",
                },
                "content": {
                    "type": "string",
                    "description": (
                        "Message content (supports formatting: bold, italic, underline,"
                        " strikethrough, lists, quotes, code). Max 2000 characters."
                    ),
                    "maxLength": 2000,
                    "x-ui-type": "rich_text",
                    "x-rich-text-platform": "discord",
                },
                "embed": {
                    "type": "object",
                    "description": "Optional rich embed object. See Discord API documentation for embed structure.",
                    "default": None,
                },
            },
            "required": ["user_id"],
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Message ID"},
                "channel_id": {"type": "string", "description": "DM channel ID"},
                "author": {"type": "object", "description": "Message author information"},
                "recipient": {"type": "object", "description": "DM recipient user information"},
                "content": {"type": ["string", "null"], "description": "Message content (null if only embed)"},
                "timestamp": {"type": "string", "description": "ISO8601 timestamp"},
                "edited_timestamp": {"type": ["string", "null"], "description": "ISO8601 timestamp if edited"},
                "tts": {"type": "boolean", "description": "Whether this was a TTS message"},
                "mentions": {"type": "array", "description": "Array of user objects mentioned"},
                "attachments": {"type": "array", "description": "Array of attachment objects"},
                "embeds": {"type": "array", "description": "Array of embed objects"},
            },
            "required": ["id", "channel_id", "timestamp"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        *,
        credentials: Optional[ResolvedCredentials] = None,
        context: Optional["WorkflowRuntimeContext"] = None,
    ) -> Dict[str, Any]:
        # access_token and context unused but required for interface consistency
        _ = access_token, context
        user_id = str(arguments.get("user_id") or "")
        content = arguments.get("content")
        embed = arguments.get("embed")

        if not user_id:
            raise HTTPException(status_code=400, detail="Parameter 'user_id' is required")
        if not content and not embed:
            raise HTTPException(
                status_code=400,
                detail="Either 'content' or 'embed' must be provided"
            )

        # Validate content length
        if content and len(content) > 2000:
            raise HTTPException(
                status_code=400,
                detail="Message content cannot exceed 2000 characters"
            )

        logger.info("Sending Discord DM to user_id=%s", user_id)

        # Step 1: Create or get DM channel
        create_dm_body = {"recipient_id": user_id}
        create_dm_resp = await self._make_request(
            "POST",
            f"{DISCORD_API_BASE}/users/@me/channels",
            credentials=credentials,
            json_body=create_dm_body
        )
        dm_channel = create_dm_resp.json()
        dm_channel_id = dm_channel.get("id")

        if not dm_channel_id:
            raise HTTPException(
                status_code=500,
                detail="Failed to create DM channel: no channel ID in response"
            )

        # Step 2: Send message to DM channel with formatted content
        message_body: Dict[str, Any] = {}
        if content:
            # Convert markdown to Discord format
            formatted_content = _rich_text_converter.convert(str(content), Platform.DISCORD)
            message_body["content"] = formatted_content
        if embed:
            message_body["embed"] = embed

        resp = await self._make_request(
            "POST",
            f"{DISCORD_API_BASE}/channels/{dm_channel_id}/messages",
            credentials=credentials,
            json_body=message_body
        )

        return resp.json()
