from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.logger import get_logger
from seer.tools.base import BaseTool, ResourcePickerConfig
from seer.tools.supabase.common import (
    _require_project_and_key,
    _resolve_functions_url,
    _service_headers,
)

logger = get_logger("shared.tools.supabase.edge_functions")


# -----------------------------
# Edge Functions (/functions/v1)
# -----------------------------


class SupabaseFunctionInvokeTool(BaseTool):
    name = "supabase_function_invoke"
    description = "Invoke a Supabase Edge Function (POST)."
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
                "function": {"type": "string", "description": "Edge function name."},
                "body": {"type": "object", "additionalProperties": True, "description": "JSON body passed to the function."},
                "headers": {"type": "object", "additionalProperties": {"type": "string"}, "description": "Optional extra headers."},
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
        fn_url = _resolve_functions_url(resource)
        if not fn_url:
            raise HTTPException(status_code=400, detail="Supabase project metadata is missing functions URL. Please re-bind.")

        fn = arguments["function"]
        body = arguments.get("body")
        extra_headers = arguments.get("headers") or {}

        headers = _service_headers(service_key, extra={"Content-Type": "application/json"})
        headers.update(extra_headers)

        url = f"{fn_url.rstrip('/')}/{fn}"

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(url, headers=headers, json=body)
                if resp.status_code >= 400:
                    raise HTTPException(status_code=resp.status_code, detail=f"Function invoke failed: {resp.text[:500]}")
                # Functions can return arbitrary JSON/text
                ctype = resp.headers.get("content-type", "")
                if ctype.startswith("application/json"):
                    return resp.json()
                return {"text": resp.text}
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Supabase function invoke error", extra={"function": fn})
            raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}")
