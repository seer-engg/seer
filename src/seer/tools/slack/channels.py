"""
Slack channel operations - listing channels and getting message history.
"""
# pylint: disable=duplicate-code  # Reason: Resource picker patterns are intentionally shared across Slack tools

from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.slack.base import SlackAPIClient

logger = get_logger("shared.tools.slack.channels")


class SlackListChannelsTool(SlackAPIClient):
    """List channels in a Slack workspace."""

    name = "slack_list_channels"
    description = "List public and private channels in a Slack workspace that the bot has access to."
    required_scopes = ["channels:read", "groups:read"]
    integration_type = "slack"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Slack workspace (team) ID",
                },
                "types": {
                    "type": "string",
                    "description": "Channel types to include (comma-separated: public_channel, private_channel)",
                    "default": "public_channel,private_channel",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of channels to return",
                    "default": 100,
                    "maximum": 1000,
                },
            },
            "required": ["workspace_id"],
        }

    def get_resource_pickers(self) -> Dict[str, Any]:
        return {
            "workspace_id": {
                "resource_type": "workspace",
                "display_field": "name",
                "value_field": "resource_id",
                "search_enabled": True,
                "filter": {"provider": "slack", "resource_type": "workspace"},
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ok": {"type": "boolean"},
                "channels": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "is_private": {"type": "boolean"},
                            "is_archived": {"type": "boolean"},
                            "num_members": {"type": "integer"},
                        },
                    },
                },
            },
            "required": ["ok", "channels"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        types = arguments.get("types", "public_channel,private_channel")
        limit = arguments.get("limit", 100)

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")

        logger.info("Listing Slack channels: workspace_id=%s", workspace_id)

        return await self._make_request(
            "GET",
            "conversations.list",
            credentials=credentials,
            params={
                "types": types,
                "limit": limit,
                "exclude_archived": "true",
            }
        )


class SlackGetChannelHistoryTool(SlackAPIClient):
    """Get message history from a Slack channel."""

    name = "slack_get_channel_history"
    description = "Get recent messages from a Slack channel."
    required_scopes = ["channels:history", "groups:history"]
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
                    "description": "Slack channel ID",
                },
                "limit": {
                    "type": "integer",
                    "description": "Number of messages to return",
                    "default": 20,
                    "maximum": 100,
                },
                "oldest": {
                    "type": "string",
                    "description": "Only messages after this timestamp",
                },
                "latest": {
                    "type": "string",
                    "description": "Only messages before this timestamp",
                },
            },
            "required": ["workspace_id", "channel_id"],
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
                "ok": {"type": "boolean"},
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {"type": "string"},
                            "user": {"type": "string"},
                            "text": {"type": "string"},
                            "ts": {"type": "string"},
                        },
                    },
                },
                "has_more": {"type": "boolean"},
            },
            "required": ["ok", "messages"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        channel_id = str(arguments.get("channel_id") or "")
        limit = arguments.get("limit", 20)
        oldest = arguments.get("oldest")
        latest = arguments.get("latest")

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")
        if not channel_id:
            raise HTTPException(status_code=400, detail="Parameter 'channel_id' is required")

        logger.info(
            "Getting Slack channel history: channel_id=%s, limit=%s",
            channel_id,
            limit
        )

        params: Dict[str, Any] = {
            "channel": channel_id,
            "limit": limit,
        }
        if oldest:
            params["oldest"] = oldest
        if latest:
            params["latest"] = latest

        return await self._make_request(
            "GET",
            "conversations.history",
            credentials=credentials,
            params=params
        )


class SlackJoinChannelTool(SlackAPIClient):
    """Join a public Slack channel."""

    name = "slack_join_channel"
    description = """Join a public Slack channel. Required before sending messages to channels the bot hasn't joined yet.
    Only works for public channels - private channels require an invite."""
    required_scopes = ["channels:join"]
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
                    "description": "Slack channel ID to join (e.g., C01234567890)",
                },
            },
            "required": ["workspace_id", "channel_id"],
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
                "channel": {
                    "type": "object",
                    "description": "Channel information",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "is_channel": {"type": "boolean"},
                        "is_member": {"type": "boolean"},
                    },
                },
            },
            "required": ["ok", "channel"],
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[ResolvedCredentials] = None,
    ) -> Dict[str, Any]:
        workspace_id = str(arguments.get("workspace_id") or "")
        channel_id = str(arguments.get("channel_id") or "")

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")
        if not channel_id:
            raise HTTPException(status_code=400, detail="Parameter 'channel_id' is required")

        logger.info("Joining Slack channel: workspace_id=%s, channel_id=%s", workspace_id, channel_id)

        return await self._make_request(
            "POST",
            "conversations.join",
            credentials=credentials,
            json_body={"channel": channel_id}
        )
