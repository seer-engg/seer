from typing import Any, Dict, Optional

from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.supabase.common import (
    SupabaseProjectTool,
    _service_request_json_or_ok,
    _require_project_and_key,
    _resolve_auth_url,
)

logger = get_logger("shared.tools.supabase.auth_admin")


# -----------------------------
# Auth Admin (/auth/v1) - server-side only
# -----------------------------

class SupabaseAuthAdminListUsersTool(SupabaseProjectTool):
    name = "supabase_auth_admin_list_users"
    description = "List users via Supabase Auth Admin API (server-side)."

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

    def get_output_schema(self) -> Dict[str, Any]:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, access_token: Optional[str], arguments: Dict[str, Any], credentials: Optional[Any] = None) -> Any:
        resource, service_key = _require_project_and_key(credentials)
        auth_url = _resolve_auth_url(resource)
        if not auth_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing auth URL. Please re-bind.")

        params = {"page": arguments.get("page", 1), "per_page": arguments.get("per_page", 50)}
        url = f"{auth_url.rstrip('/')}/admin/users"

        return await _service_request_json_or_ok(
            "GET", service_key, url, params=params, logger_obj=logger, error_detail="List users failed"
        )


class SupabaseAuthAdminCreateUserTool(SupabaseProjectTool):
    name = "supabase_auth_admin_create_user"
    description = "Create a user via Supabase Auth Admin API (server-side)."

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
        return await _service_request_json_or_ok(
            "POST", service_key, url, json_body=payload, logger_obj=logger, error_detail="Create user failed"
        )


class SupabaseAuthAdminDeleteUserTool(SupabaseProjectTool):
    name = "supabase_auth_admin_delete_user"
    description = "Delete a user via Supabase Auth Admin API (server-side)."

    def get_parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "integration_resource_id": {"type": "integer"},
                "user_id": {"type": "string", "description": "User UUID."},
            },
            "required": ["integration_resource_id", "user_id"],
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
        return await _service_request_json_or_ok(
            "DELETE", service_key, url, logger_obj=logger, error_detail="Delete user failed", ok_fallback={"ok": True}
        )
