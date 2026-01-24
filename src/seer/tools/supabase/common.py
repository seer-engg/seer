from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from seer.tools.base import BaseTool, DefaultResourceRequirement, ResourcePickerConfig

SUPABASE_PROJECT_PICKER: ResourcePickerConfig = {
    "resource_type": "supabase_project",
    "display_field": "name",
    "value_field": "id",
    "search_enabled": True,
    "endpoint": "/integrations/supabase/resources/bindings",
}
SUPABASE_DEFAULT_RESOURCE: DefaultResourceRequirement = {
    "provider": "supabase",
    "resource_type": "project",
    "required": True,
}


def _resolve_project_ref(resource) -> Optional[str]:
    metadata = getattr(resource, "resource_metadata", None) or {}
    return getattr(resource, "resource_key", None) or metadata.get("project_ref") or metadata.get("projectRef")


def _resolve_auth_url(resource) -> Optional[str]:
    metadata = getattr(resource, "resource_metadata", None) or {}
    auth_url = metadata.get("auth_url") or metadata.get("authUrl")
    if auth_url:
        return auth_url.rstrip("/")
    ref = _resolve_project_ref(resource)
    return f"https://{ref}.supabase.co/auth/v1" if ref else None


def _resolve_storage_url(resource) -> Optional[str]:
    metadata = getattr(resource, "resource_metadata", None) or {}
    storage_url = metadata.get("storage_url") or metadata.get("storageUrl")
    if storage_url:
        return storage_url.rstrip("/")
    ref = _resolve_project_ref(resource)
    return f"https://{ref}.supabase.co/storage/v1" if ref else None


def _resolve_functions_url(resource) -> Optional[str]:
    metadata = getattr(resource, "resource_metadata", None) or {}
    fn_url = metadata.get("functions_url") or metadata.get("functionsUrl")
    if fn_url:
        return fn_url.rstrip("/")
    ref = _resolve_project_ref(resource)
    return f"https://{ref}.supabase.co/functions/v1" if ref else None


def _service_headers(service_key: str, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    # Supabase REST APIs accept `apikey` and `Authorization: Bearer <key>`
    headers = {
        "apikey": service_key,
        "Authorization": f"Bearer {service_key}",
        "Accept": "application/json",
    }
    if extra:
        headers.update(extra)
    return headers


def _require_project_and_key(credentials) -> tuple[Any, str]:
    if not credentials or not getattr(credentials, "resource", None):
        raise HTTPException(status_code=400, detail="Supabase project binding is required.")
    service_key = (getattr(credentials, "secrets", None) or {}).get("supabase_service_role_key")
    if not service_key:
        raise HTTPException(status_code=400, detail="Supabase project is missing service role key.")
    return credentials.resource, service_key


def _apply_eq_filters(params: Dict[str, Any], filters: Dict[str, Any]) -> None:
    # Keep consistent with your existing approach: eq only.
    for col, val in (filters or {}).items():
        params[col] = f"eq.{val}"


def _resolve_rest_url(resource) -> Optional[str]:
    metadata = resource.resource_metadata or {}
    rest_url = metadata.get("rest_url") or metadata.get("api_url") or metadata.get("restUrl")
    project_ref = resource.resource_key or metadata.get("project_ref")
    if rest_url:
        return rest_url
    if project_ref:
        return f"https://{project_ref}.supabase.co/rest/v1"
    return None


# pylint: disable=too-many-arguments  # Reason: Helper centralizes varied request options for Supabase APIs
async def _request_json_or_ok(
    method: str,
    url: str,
    *,
    headers: Dict[str, str],
    logger_obj,
    error_detail: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    timeout: float = 30.0,
    ok_fallback: Optional[Any] = None,
    content: Optional[bytes] = None,
) -> Any:
    """
    Perform a Supabase HTTP request with consistent error handling and JSON parsing.
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
                content=content,
            )
            if response.status_code >= 400:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"{error_detail}: {response.text[:500]}",
                )
            if response.headers.get("content-type", "").startswith("application/json"):
                return response.json()
            if ok_fallback is not None:
                return ok_fallback
            return {}
    except HTTPException:
        raise
    except Exception as exc:
        logger_obj.exception("%s error", error_detail, extra={"url": url})
        raise HTTPException(status_code=500, detail=f"Supabase request failed: {str(exc)}") from exc


# pylint: disable=too-many-arguments  # Reason: Wrapper exposes common Supabase request knobs
async def _service_request_json_or_ok(
    method: str,
    service_key: str,
    url: str,
    *,
    logger_obj,
    error_detail: str,
    params: Optional[Dict[str, Any]] = None,
    json_body: Optional[Any] = None,
    timeout: float = 30.0,
    ok_fallback: Optional[Any] = None,
    extra_headers: Optional[Dict[str, str]] = None,
    content: Optional[bytes] = None,
) -> Any:
    headers = _service_headers(service_key, extra=extra_headers)
    return await _request_json_or_ok(
        method,
        url,
        headers=headers,
        params=params,
        json_body=json_body,
        timeout=timeout,
        ok_fallback=ok_fallback,
        content=content,
        logger_obj=logger_obj,
        error_detail=error_detail,
    )


class SupabaseProjectTool(BaseTool):
    """
    Shared base for Supabase tools that operate on a project binding.

    Provides consistent integration metadata and resource picker wiring for
    the persisted Supabase project resource.
    """
    integration_type = "supabase"
    provider = "supabase"
    required_scopes: list[str] = []
    required_secrets = ["supabase_service_role_key"]
    default_resource: DefaultResourceRequirement = SUPABASE_DEFAULT_RESOURCE

    def get_resource_pickers(self) -> Dict[str, ResourcePickerConfig]:
        return {"integration_resource_id": dict(SUPABASE_PROJECT_PICKER)}
