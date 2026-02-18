"""Slack resource provider for browsing workspaces, channels, and users."""
# pylint: disable=too-many-arguments
# Reason: Resource provider list_resources has many filter parameters
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.database import IntegrationResource, OAuthConnection
from seer.services.integrations.providers.slack import SlackProvider
from seer.services.integrations.resource_providers.base import ResourceContext, ResourceProvider
from seer.services.integrations.resource_providers.utils import parse_offset
from seer.logger import get_logger

logger = get_logger(__name__)


class SlackResourceProvider(ResourceProvider):
    """
    Resource provider for Slack.

    Supports:
    - workspace: Database-backed list of Slack workspaces the user has connected
    - channel: API-backed list of channels in a specific workspace (requires workspace_id)
    - user: API-backed list of users in a specific workspace (requires workspace_id)
    """

    provider = "slack"
    resource_configs: Dict[str, Dict[str, Any]] = {
        "workspace": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "source": "database",
        },
        "channel": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "depends_on": "workspace_id",
            "source": "api",
        },
        "user": {
            "display_field": "real_name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "depends_on": "workspace_id",
            "source": "api",
        },
    }

    async def list_resources(
        self,
        *,
        access_token: Optional[str] = None,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
        context: Optional[ResourceContext] = None,
    ) -> Dict[str, Any]:
        """
        List Slack resources (workspaces, channels, or users).
        """
        if not context or not context.user:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Missing user context",
                detail="Slack resource provider requires user context",
                status=400
            )
        assert context is not None
        assert context.user is not None

        if resource_type == "workspace":
            return await self._list_workspaces(
                user=context.user,
                query=query,
                page_token=page_token,
                page_size=page_size,
            )
        if resource_type == "channel":
            workspace_id = (depends_on_values or {}).get("workspace_id")
            if not workspace_id:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing required parameter",
                    detail="workspace_id is required to list Slack channels. Please select a workspace first.",
                    status=400
                )
            return await self._list_channels(
                user=context.user,
                workspace_id=str(workspace_id),
                query=query,
                page_token=page_token,
                page_size=page_size,
            )
        if resource_type == "user":
            workspace_id = (depends_on_values or {}).get("workspace_id")
            if not workspace_id:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing required parameter",
                    detail="workspace_id is required to list Slack users. Please select a workspace first.",
                    status=400
                )
            return await self._list_users(
                user=context.user,
                workspace_id=str(workspace_id),
                query=query,
                page_token=page_token,
                page_size=page_size,
            )

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Slack resource type '{resource_type}'"
        )

    async def _list_workspaces(
        self,
        user: Any,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List Slack workspaces from database."""
        from seer.api.integrations.services import list_integration_resources  # pylint: disable=import-outside-toplevel  # Avoid circular import

        resources = await list_integration_resources(
            user,
            provider="slack",
            resource_type="workspace",
        )

        filtered_resources = resources
        if query:
            q_lower = query.lower()
            filtered_resources = [
                r for r in resources
                if q_lower in (r.name or "").lower() or q_lower in (r.resource_id or "").lower()
            ]

        offset = parse_offset(page_token)
        paged_resources = filtered_resources[offset:offset + page_size]

        items = [
            {
                "id": r.resource_id,
                "name": r.name or f"Slack Workspace {r.resource_id}",
                "display_name": r.name or f"Slack Workspace {r.resource_id}",
                "type": "workspace",
                "metadata": r.resource_metadata or {},
            }
            for r in paged_resources
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_resources) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }

    async def get_bot_token_for_workspace(self, user: Any, workspace_id: str) -> str:
        """Get bot token for a specific workspace (public API)."""
        return await self._get_bot_token_for_workspace_internal(user, workspace_id)

    async def _get_bot_token_for_workspace_internal(self, user: Any, workspace_id: str) -> str:
        """Get bot token for a specific workspace (internal)."""
        # Verify user has access to this workspace
        workspace_resource = await IntegrationResource.get_or_none(
            user=user,
            provider="slack",
            resource_type="workspace",
            resource_id=workspace_id,
            status="active"
        )
        if not workspace_resource:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Workspace not found",
                detail=f"Slack workspace {workspace_id} not found or you don't have access to it",
                status=404
            )

        # Get the OAuth connection for this workspace
        connection = await OAuthConnection.get_or_none(
            user=user,
            provider="slack",
            status="active"
        )
        if not connection or not connection.access_token_enc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="No Slack connection",
                detail="No active Slack connection found. Please reconnect Slack.",
                status=401
            )

        # Return the token (stored as plain text despite the name)
        return connection.access_token_enc

    async def _list_channels(
        self,
        user: Any,
        workspace_id: str,
        *,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List Slack channels for a specific workspace using Slack API."""
        bot_token = await self._get_bot_token_for_workspace_internal(user, workspace_id)

        provider_impl = SlackProvider()
        try:
            channels = await provider_impl.fetch_channels(
                access_token=bot_token,
                types="public_channel,private_channel",
            )
        except HTTPException as exc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch channels",
                detail=exc.detail,
                status=exc.status_code
            )

        filtered_channels: List[Dict[str, Any]] = channels
        if query:
            q_lower = query.lower()
            filtered_channels = [
                ch for ch in channels
                if q_lower in (ch.get("name") or "").lower()
            ]

        offset = parse_offset(page_token)
        paged_channels = filtered_channels[offset:offset + page_size]

        items = [
            {
                "id": ch.get("id", ""),
                "name": ch.get("name") or f"Channel {ch.get('id', '')}",
                "display_name": f"#{ch.get('name', '')}" if ch.get("name") else f"Channel {ch.get('id', '')}",
                "type": "channel",
                "metadata": {
                    "channel_id": ch.get("id", ""),
                    "channel_name": ch.get("name"),
                    "is_private": ch.get("is_private", False),
                    "is_archived": ch.get("is_archived", False),
                    "is_member": ch.get("is_member", False),
                    "workspace_id": workspace_id,
                },
            }
            for ch in paged_channels
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_channels) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }

    async def _list_users(
        self,
        user: Any,
        workspace_id: str,
        *,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List Slack users for a specific workspace using Slack API."""
        bot_token = await self._get_bot_token_for_workspace_internal(user, workspace_id)

        provider_impl = SlackProvider()
        try:
            users = await provider_impl.fetch_users(access_token=bot_token)
        except HTTPException as exc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch users",
                detail=exc.detail,
                status=exc.status_code
            )

        filtered_users: List[Dict[str, Any]] = users
        if query:
            q_lower = query.lower()
            filtered_users = [
                u for u in users
                if q_lower in (u.get("name") or "").lower()
                or q_lower in (u.get("real_name") or "").lower()
                or q_lower in (u.get("profile", {}).get("display_name") or "").lower()
            ]

        offset = parse_offset(page_token)
        paged_users = filtered_users[offset:offset + page_size]

        items = [
            {
                "id": u.get("id", ""),
                "name": u.get("real_name") or u.get("name") or f"User {u.get('id', '')}",
                "display_name": u.get("real_name") or u.get("profile", {}).get("display_name") or u.get("name", ""),
                "type": "user",
                "metadata": {
                    "user_id": u.get("id", ""),
                    "username": u.get("name"),
                    "real_name": u.get("real_name"),
                    "email": u.get("profile", {}).get("email"),
                    "workspace_id": workspace_id,
                },
            }
            for u in paged_users
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_users) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }
