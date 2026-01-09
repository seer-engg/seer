from typing import Any, Dict, Optional

from fastapi import HTTPException


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
