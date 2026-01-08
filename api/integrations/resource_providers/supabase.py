from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from api.integrations.constants import (
    SUPABASE_RESOURCE_PROVIDER,
    SUPABASE_RESOURCE_TYPE_PROJECT,
)
from api.integrations.providers import get_integration_provider
from api.integrations.resource_providers.base import ResourceProvider
from shared.logger import get_logger

logger = get_logger("api.integrations.resource_providers.supabase")


class SupabaseResourceProvider(ResourceProvider):
    provider = SUPABASE_RESOURCE_PROVIDER
    aliases = ["supabase", "supabase_mgmt"]
    resource_configs: Dict[str, Dict[str, Any]] = {
        "supabase_project": {
            "display_field": "name",
            "value_field": "ref",
            "supports_hierarchy": False,
            "supports_search": True,
        },
    }

    async def list_resources(
        self,
        *,
        access_token: str,
        resource_type: str,
        query: Optional[str],
        parent_id: Optional[str],
        page_token: Optional[str],
        page_size: int,
        filter_params: Optional[Dict[str, Any]],
        depends_on_values: Optional[Dict[str, str]],
    ) -> Dict[str, Any]:
        if resource_type != "supabase_project":
            raise HTTPException(status_code=400, detail=f"Unsupported Supabase resource type '{resource_type}'")

        provider = get_integration_provider(SUPABASE_RESOURCE_PROVIDER)
        if not provider:
            return {"items": [], "error": "Supabase provider unavailable", "next_page_token": None}

        try:
            projects = await provider.list_remote_resources(
                access_token=access_token,
                resource_type=SUPABASE_RESOURCE_TYPE_PROJECT,
            )
        except HTTPException as exc:
            return {"items": [], "error": exc.detail, "next_page_token": None}
        except Exception as exc:
            logger.exception("Error listing Supabase projects: %s", exc)
            return {"items": [], "error": str(exc), "next_page_token": None}

        if query:
            query_lower = query.lower()
            projects = [
                project
                for project in projects
                if query_lower in (project.get("name") or "").lower()
                or query_lower in (project.get("ref") or "").lower()
            ]

        total = len(projects)
        start_index = 0
        if page_token:
            try:
                start_index = int(page_token)
            except ValueError:
                start_index = 0
        end_index = start_index + page_size
        sliced = projects[start_index:end_index]
        next_page = str(end_index) if end_index < total else None

        items = []
        for project in sliced:
            ref = project.get("ref") or project.get("project_ref")
            project_id = project.get("id") or project.get("project_id")
            items.append(
                {
                    "id": ref or str(project_id),
                    "name": project.get("name") or ref or project_id,
                    "display_name": project.get("name") or ref or project_id,
                    "type": "project",
                    "project_id": project_id,
                    "project_ref": ref,
                    "organization_id": project.get("organization_id") or project.get("org_id"),
                    "region": project.get("region"),
                    "status": project.get("status"),
                    "rest_url": project.get("api_url") or project.get("restUrl"),
                }
            )

        return {
            "items": items,
            "next_page_token": next_page,
            "supports_hierarchy": False,
            "supports_search": True,
            "total_count": total,
        }
