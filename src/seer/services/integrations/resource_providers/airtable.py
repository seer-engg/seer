"""Airtable resource provider for browsing bases and tables."""
# pylint: disable=too-many-arguments,too-many-positional-arguments
# Reason: Resource provider list_resources and helper methods have many filter parameters
from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import HTTPException

from seer.api.core.errors import INTEGRATION_PROBLEM, VALIDATION_PROBLEM, raise_problem
from seer.database import OAuthConnection
from seer.services.integrations.resource_providers.base import ResourceContext, ResourceProvider
from seer.services.integrations.resource_providers.utils import parse_offset
from seer.logger import get_logger

logger = get_logger(__name__)

# Airtable API base URL
AIRTABLE_API_BASE = "https://api.airtable.com/v0"


class AirtableResourceProvider(ResourceProvider):
    """
    Resource provider for Airtable.

    Supports:
    - base: API-backed list of Airtable bases accessible to the user
    - table: API-backed list of tables in a specific base (requires base_id)
    """

    provider = "airtable"
    resource_configs: Dict[str, Dict[str, Any]] = {
        "base": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "source": "api",
        },
        "table": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "depends_on": "base_id",
            "source": "api",
        },
        "view": {
            "display_field": "name",
            "value_field": "id",
            "supports_search": True,
            "supports_hierarchy": False,
            "depends_on": "table_id",
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
        List Airtable resources (bases or tables).
        """
        if not context or not context.user:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Missing user context",
                detail="Airtable resource provider requires user context",
                status=400
            )
        assert context is not None
        assert context.user is not None

        if resource_type == "base":
            return await self._list_bases(
                user=context.user,
                query=query,
                page_token=page_token,
                page_size=page_size,
            )
        if resource_type == "table":
            base_id = (depends_on_values or {}).get("base_id")
            if not base_id:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing required parameter",
                    detail="base_id is required to list Airtable tables. Please select a base first.",
                    status=400
                )
            return await self._list_tables(
                user=context.user,
                base_id=str(base_id),
                query=query,
                page_token=page_token,
                page_size=page_size,
            )
        if resource_type == "view":
            base_id = (depends_on_values or {}).get("base_id")
            table_id = (depends_on_values or {}).get("table_id")
            if not base_id or not table_id:
                raise_problem(
                    type_uri=VALIDATION_PROBLEM,
                    title="Missing required parameter",
                    detail="base_id and table_id are required to list Airtable views. Please select a base and table first.",
                    status=400
                )
            return await self._list_views(
                user=context.user,
                base_id=str(base_id),
                table_id=str(table_id),
                query=query,
                page_token=page_token,
                page_size=page_size,
            )

        raise HTTPException(
            status_code=400,
            detail=f"Unsupported Airtable resource type '{resource_type}'"
        )

    async def _get_access_token(self, user: Any) -> str:
        """Get access token for the user's Airtable connection."""
        connection = await OAuthConnection.filter(
            user=user,
            provider="airtable",
            status="active",
        ).order_by("-updated_at").first()
        if not connection or not connection.access_token_enc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="No Airtable connection",
                detail="No active Airtable connection found. Please connect your Airtable account.",
                status=401
            )

        return connection.access_token_enc

    async def _list_bases(
        self,
        user: Any,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List Airtable bases using the API."""
        access_token = await self._get_access_token(user)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{AIRTABLE_API_BASE}/meta/bases",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch bases",
                detail=f"Airtable API error: {exc.response.status_code}",
                status=exc.response.status_code
            )

        bases: List[Dict[str, Any]] = data.get("bases", [])

        # Filter by query if provided
        filtered_bases = bases
        if query:
            q_lower = query.lower()
            filtered_bases = [
                b for b in bases
                if q_lower in (b.get("name") or "").lower()
            ]

        # Apply pagination
        offset = parse_offset(page_token)
        paged_bases = filtered_bases[offset:offset + page_size]

        items = [
            {
                "id": b.get("id", ""),
                "name": b.get("name") or f"Base {b.get('id', '')}",
                "display_name": b.get("name") or f"Base {b.get('id', '')}",
                "type": "base",
                "metadata": {
                    "base_id": b.get("id", ""),
                    "base_name": b.get("name"),
                    "permission_level": b.get("permissionLevel"),
                },
            }
            for b in paged_bases
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_bases) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }

    async def _list_tables(
        self,
        user: Any,
        base_id: str,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List tables in an Airtable base using the API."""
        access_token = await self._get_access_token(user)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{AIRTABLE_API_BASE}/meta/bases/{base_id}/tables",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise_problem(
                    type_uri=INTEGRATION_PROBLEM,
                    title="Base not found",
                    detail=f"Airtable base '{base_id}' not found or you don't have access to it",
                    status=404
                )
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch tables",
                detail=f"Airtable API error: {exc.response.status_code}",
                status=exc.response.status_code
            )

        tables: List[Dict[str, Any]] = data.get("tables", [])

        # Filter by query if provided
        filtered_tables = tables
        if query:
            q_lower = query.lower()
            filtered_tables = [
                t for t in tables
                if q_lower in (t.get("name") or "").lower()
            ]

        # Apply pagination
        offset = parse_offset(page_token)
        paged_tables = filtered_tables[offset:offset + page_size]

        items = [
            {
                "id": t.get("id", ""),
                "name": t.get("name") or f"Table {t.get('id', '')}",
                "display_name": t.get("name") or f"Table {t.get('id', '')}",
                "type": "table",
                "metadata": {
                    "table_id": t.get("id", ""),
                    "table_name": t.get("name"),
                    "description": t.get("description"),
                    "primary_field_id": t.get("primaryFieldId"),
                    "base_id": base_id,
                    "field_count": len(t.get("fields", [])),
                    "view_count": len(t.get("views", [])),
                },
            }
            for t in paged_tables
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(filtered_tables) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }

    async def _list_views(
        self,
        user: Any,
        base_id: str,
        table_id: str,
        query: Optional[str],
        page_token: Optional[str],
        page_size: int,
    ) -> Dict[str, Any]:
        """List views in an Airtable table using the API.

        Views are included in the table schema response from the tables endpoint.
        """
        access_token = await self._get_access_token(user)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(
                    f"{AIRTABLE_API_BASE}/meta/bases/{base_id}/tables",
                    headers={"Authorization": f"Bearer {access_token}"},
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise_problem(
                    type_uri=INTEGRATION_PROBLEM,
                    title="Base not found",
                    detail=f"Airtable base '{base_id}' not found or you don't have access to it",
                    status=404
                )
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Failed to fetch views",
                detail=f"Airtable API error: {exc.response.status_code}",
                status=exc.response.status_code
            )

        # Find the specific table and extract its views
        tables: List[Dict[str, Any]] = data.get("tables", [])
        target_table = next((t for t in tables if t.get("id") == table_id), None)

        if not target_table:
            raise_problem(
                type_uri=INTEGRATION_PROBLEM,
                title="Table not found",
                detail=f"Airtable table '{table_id}' not found in base '{base_id}'",
                status=404
            )

        views: List[Dict[str, Any]] = target_table.get("views", [])

        # Filter by query if provided
        if query:
            q_lower = query.lower()
            views = [v for v in views if q_lower in (v.get("name") or "").lower()]

        # Apply pagination
        offset = parse_offset(page_token)
        paged_views = views[offset:offset + page_size]

        items = [
            {
                "id": v.get("id", ""),
                "name": v.get("name") or f"View {v.get('id', '')}",
                "display_name": v.get("name") or f"View {v.get('id', '')}",
                "type": "view",
                "metadata": {
                    "view_id": v.get("id", ""),
                    "view_name": v.get("name"),
                    "view_type": v.get("type"),
                    "base_id": base_id,
                    "table_id": table_id,
                },
            }
            for v in paged_views
        ]

        next_page_token = str(offset + page_size) if offset + page_size < len(views) else None

        return {
            "items": items,
            "next_page_token": next_page_token,
            "supports_search": True,
            "supports_hierarchy": False,
        }
