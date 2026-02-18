"""Browser resource provider for browsing browser profiles."""
# pylint: disable=too-many-arguments
# Reason: Resource provider list_resources has many filter parameters
from __future__ import annotations

from typing import Any, Dict, Optional

from seer.api.core.errors import INTEGRATION_PROBLEM, raise_problem
from seer.services.browser import BrowserProfileManager
from seer.services.integrations.resource_providers.base import ResourceContext, ResourceProvider
from seer.services.integrations.resource_providers.utils import parse_offset
from seer.logger import get_logger

logger = get_logger(__name__)


class BrowserResourceProvider(ResourceProvider):
    """
    Resource provider for browser profiles.

    Supports:
    - browser_profile: Database-backed list of browser profiles the user has created
    """

    provider = "browser"
    resource_configs: Dict[str, Dict[str, Any]] = {
        "browser_profile": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "source": "database",
        },
    }

    async def list_resources(  # pylint: disable=unused-argument  # Reason: Part of ResourceProvider interface
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
        List browser resources (profiles).
        """
        if not context or not context.user:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Missing user context",
                detail="Browser resource provider requires user context",
                status=400
            )
        assert context is not None
        assert context.user is not None

        if resource_type == "browser_profile":
            return await self._list_profiles(
                user=context.user,
                query=query,
                page_token=page_token,
                page_size=page_size,
            )

        raise_problem(
            type_uri=INTEGRATION_PROBLEM,
            title="Unsupported resource type",
            detail=f"Unsupported browser resource type '{resource_type}'",
            status=400
        )
        raise AssertionError("raise_problem should have raised")  # Unreachable, for type checker

    async def _list_profiles(
        self,
        user: Any,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List browser profiles from database."""
        manager = BrowserProfileManager()
        profiles = await manager.list_profiles(user)

        # Filter by search query
        filtered_profiles = profiles
        if query:
            q_lower = query.lower()
            filtered_profiles = [
                p for p in profiles
                if q_lower in (p.get("name") or "").lower()
            ]

        # Paginate
        offset = parse_offset(page_token)
        paged_profiles = filtered_profiles[offset:offset + page_size]

        # Format as resource items
        items = [
            {
                "id": p["id"],
                "name": p["name"],
                "display_name": p["name"],
                "type": "browser_profile",
                "metadata": {
                    "logged_in_domains": p.get("logged_in_domains", []),
                    "created_at": p.get("created_at"),
                    "last_used_at": p.get("last_used_at"),
                },
            }
            for p in paged_profiles
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_profiles) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }
