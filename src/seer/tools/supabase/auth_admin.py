from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.supabase.common import (
    _require_project_and_key,
    _resolve_auth_url,
    _service_headers,
)

logger = get_logger("shared.tools.supabase.auth_admin")


# -----------------------------
# Auth Admin (/auth/v1) - server-side only
# -----------------------------

class SupabaseAuthAdminListUsersTool(BaseTool):
    name = "supabase_auth_admin_list_users"
    description = "List users via Supabase Auth Admin API (server-side)."
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
                "page": {"type": "integer", "minimum": 1, "default": 1},
                "per_page": {"type": "integer", "minimum": 1, "maximum": 1000, "default": 50},
            },
            "required": ["integration_resource_id"],
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
        auth_url = _resolve_auth_url(resource)
        if not auth_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing auth URL. Please re-bind.")

        params = {"page": arguments.get("page", 1), "per_page": arguments.get("per_page", 50)}
        url = f"{auth_url.rstrip('/')}/admin/users"
        headers = _service_headers(service_key)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url, headers=headers, params=params)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"List users failed: {resp.text[:500]}")
                return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase auth admin list users error")
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseAuthAdminCreateUserTool(BaseTool):
    name = "supabase_auth_admin_create_user"
    description = "Create a user via Supabase Auth Admin API (server-side)."
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
                "email": {"type": "string"},
                "password": {"type": "string"},
                "email_confirm": {"type": "boolean", "default": True},
                "user_metadata": {"type": "object", "additionalProperties": True},
                "app_metadata": {"type": "object", "additionalProperties": True},
            },
            "required": ["integration_resource_id", "email"],
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
        auth_url = _resolve_auth_url(resource)
        if not auth_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing auth URL. Please re-bind.")

        payload: Dict[str, Any] = {
            "email": arguments["email"],
            "email_confirm": bool(arguments.get("email_confirm", True)),
        }
        if arguments.get("password"):
            payload["password"] = arguments["password"]
        if arguments.get("user_metadata"):
            payload["user_metadata"] = arguments["user_metadata"]
        if arguments.get("app_metadata"):
            payload["app_metadata"] = arguments["app_metadata"]

        url = f"{auth_url.rstrip('/')}/admin/users"
        headers = _service_headers(service_key, extra={"Content-Type": "application/json"})

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Create user failed: {resp.text[:500]}")
                return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase auth admin create user error", extra={"email": arguments.get("email")})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")


class SupabaseAuthAdminDeleteUserTool(BaseTool):
    name = "supabase_auth_admin_delete_user"
    description = "Delete a user via Supabase Auth Admin API (server-side)."
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
                "user_id": {"type": "string", "description": "User UUID."},
            },
            "required": ["integration_resource_id", "user_id"],
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
        auth_url = _resolve_auth_url(resource)
        if not auth_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing auth URL. Please re-bind.")

        user_id = arguments["user_id"]
        url = f"{auth_url.rstrip('/')}/admin/users/{user_id}"
        headers = _service_headers(service_key)

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.delete(url, headers=headers)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Delete user failed: {resp.text[:500]}")
                # Some responses are empty; normalize.
                if resp.headers.get("content-type", "").startswith("application/json"):
                    return resp.json()
                return {"ok": True}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase auth admin delete user error", extra={"user_id": user_id})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")
