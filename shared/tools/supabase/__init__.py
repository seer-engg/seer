from __future__ import annotations

from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from shared.logger import get_logger
from shared.tools.base import BaseTool, ResourcePickerConfig, register_tool

logger = get_logger("shared.tools.supabase")


class SupabaseTableQueryTool(BaseTool):
    name = "supabase_table_query"
    description = "Query a Supabase table via the REST interface (read-only)."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {
        "provider": "supabase",
        "resource_type": "project",
        "required": True,
    }

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {
                    "type": "integer",
                    "description": "Persisted Supabase project resource ID.",
                },
                "table": {
                    "type": "string",
                    "description": "Table or view name to query.",
                },
                "select": {
                    "type": "string",
                    "description": "Columns to select (PostgREST syntax). Defaults to '*'.",
                    "default": "*",
                },
                "filters": {
                    "type": "object",
                    "description": "Column filters mapping to literal values (eq).",
                    "additionalProperties": {"type": "string"},
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                    "description": "Maximum rows to return (default 100).",
                    "default": 100,
                },
                "order_by": {
                    "type": "string",
                    "description": "Column ordering, e.g., 'created_at.desc' or 'id.asc'.",
                },
            },
            "required": ["integration_resource_id", "table"],
        }

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {
            "integration_resource_id": {
                "resource_type": "supabase_project",
                "display_field": "name",
                "value_field": "id",
                "search_enabled": True,
                "endpoint": "/integrations/supabase/resources/bindings",
            }
        }

    def get_output_schema(self) -> Dict[str, Any]:
        return {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        }

    async def execute(
        self,
        access_token: Optional[str],
        arguments: Dict[str, Any],
        credentials: Optional[Any] = None,
    ) -> Any:
        if not credentials or not credentials.resource:
            raise HTTPException(status_code=400, detail="Supabase project binding is required.")
        resource = credentials.resource
        service_key = credentials.secrets.get("supabase_service_role_key")
        if not service_key:
            raise HTTPException(status_code=400, detail="Supabase project is missing service role key.")

        table = arguments["table"]
        select = arguments.get("select") or "*"
        limit = arguments.get("limit", 100)
        order_by = arguments.get("order_by")
        filters = arguments.get("filters") or {}

        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(
                status_code=400,
                detail="Supabase project metadata is missing rest_url. Please re-bind the project.",
            )

        params: Dict[str, Any] = {
            "select": select,
            "limit": limit,
        }
        if order_by:
            params["order"] = order_by

        for column, value in filters.items():
            params[column] = f"eq.{value}"

        headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Accept": "application/json",
        }

        url = f"{rest_url.rstrip('/')}/{table}"
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url, headers=headers, params=params)
                if response.status_code >= 400:
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"Supabase query failed: {response.text[:500]}",
                    )
                return response.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase query error", extra={"table": table})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


def _resolve_rest_url(resource) -> Optional[str]:
    metadata = resource.resource_metadata or {}
    rest_url = metadata.get("rest_url") or metadata.get("api_url") or metadata.get("restUrl")
    project_ref = resource.resource_key or metadata.get("project_ref")
    if rest_url:
        return rest_url
    if project_ref:
        return f"https://{project_ref}.supabase.co/rest/v1"
    return None


def register_supabase_tools() -> None:
    register_tool(SupabaseTableQueryTool())


__all__ = ["register_supabase_tools", "SupabaseTableQueryTool"]


register_supabase_tools()