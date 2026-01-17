from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.supabase.common import (
    _apply_eq_filters,
    _require_project_and_key,
    _resolve_rest_url,
    _service_headers,
)

logger = get_logger("shared.tools.supabase.database")

# assuming you already have:
# - BaseTool
# - ResourcePickerConfig
# - logger
# - credentials.resource / credentials.secrets
# and your existing _resolve_rest_url(resource)


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


# -----------------------------
# Database (PostgREST) - WRITE
# -----------------------------

class SupabaseTableInsertTool(BaseTool):
    name = "supabase_table_insert"
    description = "Insert rows into a Supabase table via PostgREST."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {"provider": "supabase", "resource_type": "project", "required": True}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer", "description": "Persisted Supabase project resource ID."},
                "table": {"type": "string", "description": "Table name."},
                "rows": {
                    "type": "array",
                    "description": "Rows to insert.",
                    "items": {"type": "object", "additionalProperties": True},
                    "minItems": 1,
                },
                "return_rows": {
                    "type": "boolean",
                    "description": "Return inserted rows (Prefer: return=representation).",
                    "default": True,
                },
            },
            "required": ["integration_resource_id", "table", "rows"],
        }

    def get_resource_pickers(self) -> Dict[str, "ResourcePickerConfig"]:
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
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing rest_url. Please re-bind.")

        table = arguments["table"]
        rows = arguments["rows"]
        return_rows = bool(arguments.get("return_rows", True))

        headers = _service_headers(
            service_key,
            extra={
                "Content-Type": "application/json",
                "Prefer": "return=representation" if return_rows else "return=minimal",
            },
        )
        url = f"{rest_url.rstrip('/')}/{table}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=rows)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Supabase insert failed: {resp.text[:500]}")
                return resp.json() if return_rows else {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase insert error", extra={"table": table})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseTableUpsertTool(BaseTool):
    name = "supabase_table_upsert"
    description = "Upsert rows into a Supabase table via PostgREST (merge duplicates)."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {"provider": "supabase", "resource_type": "project", "required": True}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "table": {"type": "string"},
                "rows": {"type": "array", "items": {"type": "object", "additionalProperties": True}, "minItems": 1},
                "on_conflict": {
                    "type": "string",
                    "description": "Comma-separated unique columns for upsert (PostgREST on_conflict). Example: 'email' or 'org_id,email'.",
                },
                "return_rows": {"type": "boolean", "default": True},
            },
            "required": ["integration_resource_id", "table", "rows"],
        }

    def get_resource_pickers(self) -> Dict[str, "ResourcePickerConfig"]:
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
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing rest_url. Please re-bind.")

        table = arguments["table"]
        rows = arguments["rows"]
        on_conflict = arguments.get("on_conflict")
        return_rows = bool(arguments.get("return_rows", True))

        params: Dict[str, Any] = {}
        if on_conflict:
            params["on_conflict"] = on_conflict

        prefer_parts = ["resolution=merge-duplicates", "return=representation" if return_rows else "return=minimal"]
        headers = _service_headers(
            service_key,
            extra={"Content-Type": "application/json", "Prefer": ",".join(prefer_parts)},
        )
        url = f"{rest_url.rstrip('/')}/{table}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, params=params, json=rows)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Supabase upsert failed: {resp.text[:500]}")
                return resp.json() if return_rows else {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase upsert error", extra={"table": table})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseTableUpdateTool(BaseTool):
    name = "supabase_table_update"
    description = "Update rows in a Supabase table via PostgREST (PATCH)."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {"provider": "supabase", "resource_type": "project", "required": True}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "table": {"type": "string"},
                "patch": {"type": "object", "additionalProperties": True, "description": "Partial object of columns to update."},
                "filters": {
                    "type": "object",
                    "description": "Column filters mapping to literal values (eq).",
                    "additionalProperties": {"type": "string"},
                },
                "return_rows": {"type": "boolean", "default": True},
            },
            "required": ["integration_resource_id", "table", "patch", "filters"],
        }

    def get_resource_pickers(self) -> Dict[str, "ResourcePickerConfig"]:
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
        return {"type": "array", "items": {"type": "object", "additionalProperties": True}}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing rest_url. Please re-bind.")

        table = arguments["table"]
        patch = arguments["patch"]
        filters = arguments["filters"]
        return_rows = bool(arguments.get("return_rows", True))

        params: Dict[str, Any] = {}
        _apply_eq_filters(params, filters)

        headers = _service_headers(
            service_key,
            extra={
                "Content-Type": "application/json",
                "Prefer": "return=representation" if return_rows else "return=minimal",
            },
        )
        url = f"{rest_url.rstrip('/')}/{table}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.patch(url, headers=headers, params=params, json=patch)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Supabase update failed: {resp.text[:500]}")
                return resp.json() if return_rows else {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase update error", extra={"table": table})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseTableDeleteTool(BaseTool):
    name = "supabase_table_delete"
    description = "Delete rows from a Supabase table via PostgREST (DELETE)."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {"provider": "supabase", "resource_type": "project", "required": True}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "table": {"type": "string"},
                "filters": {"type": "object", "additionalProperties": {"type": "string"}, "description": "eq filters"},
                "return_rows": {"type": "boolean", "default": False},
            },
            "required": ["integration_resource_id", "table", "filters"],
        }

    def get_resource_pickers(self) -> Dict[str, "ResourcePickerConfig"]:
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
            "type": "object",
            "additionalProperties": True,
            "description": "Either {ok:true} or deleted rows (if return_rows).",
        }

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing rest_url. Please re-bind.")

        table = arguments["table"]
        filters = arguments["filters"]
        return_rows = bool(arguments.get("return_rows", False))

        params: Dict[str, Any] = {}
        _apply_eq_filters(params, filters)

        headers = _service_headers(
            service_key,
            extra={"Prefer": "return=representation" if return_rows else "return=minimal"},
        )
        url = f"{rest_url.rstrip('/')}/{table}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(url, headers=headers, params=params)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Supabase delete failed: {resp.text[:500]}")
                return resp.json() if return_rows else {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase delete error", extra={"table": table})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseRpcCallTool(BaseTool):
    name = "supabase_rpc_call"
    description = "Call a Postgres function via PostgREST RPC endpoint."
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource = {"provider": "supabase", "resource_type": "project", "required": True}

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "function": {"type": "string", "description": "Postgres function name."},
                "args": {"type": "object", "additionalProperties": True, "description": "JSON args passed to the function."},
            },
            "required": ["integration_resource_id", "function"],
        }

    def get_resource_pickers(self) -> Dict[str, "ResourcePickerConfig"]:
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
        return {"type": "object", "additionalProperties": True}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        rest_url = _resolve_rest_url(resource)
        if not rest_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing rest_url. Please re-bind.")

        fn = arguments["function"]
        args = arguments.get("args") or {}

        headers = _service_headers(service_key, extra={"Content-Type": "application/json"})
        url = f"{rest_url.rstrip('/')}/rpc/{fn}"

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=args)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Supabase RPC failed: {resp.text[:500]}")
                # RPC can return scalar/array/object depending on function
                return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase rpc error", extra={"function": fn})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")
