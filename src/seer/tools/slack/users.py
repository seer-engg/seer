"""
Slack user operations - listing workspace members.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.credential_resolver import ResolvedCredentials
from seer.tools.slack.base import SlackAPIClient

if TYPE_CHECKING:
    from seer.core.runtime.context import WorkflowRuntimeContext

logger = get_logger("shared.tools.slack.users")


class SlackListUsersTool(SlackAPIClient):
    """List users in a Slack workspace."""

    name = "slack_list_users"
    description = "List members in a Slack workspace."
    required_scopes = ["users:read"]
    integration_type = "slack"

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "workspace_id": {
                    "type": "string",
                    "description": "Slack workspace (team) ID",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of users to return",
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
                "members": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "real_name": {"type": "string"},
                            "is_admin": {"type": "boolean"},
                            "is_bot": {"type": "boolean"},
                        },
                    },
                },
            },
            "required": ["ok", "members"],
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
        limit = arguments.get("limit", 100)

        if not workspace_id:
            raise HTTPException(status_code=400, detail="Parameter 'workspace_id' is required")

        logger.info("Listing Slack users: workspace_id=%s", workspace_id)

        return await self._make_request(
            "GET",
            "users.list",
            credentials=credentials,
            params={"limit": limit}
        )
