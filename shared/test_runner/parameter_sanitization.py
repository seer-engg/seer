"""Parameter sanitization utilities, including Asana-specific logic."""
import os
from typing import Dict, Any, Optional

from shared.test_runner.variable_injection import get_field


_ASANA_WORKSPACE_ENV_KEYS = ("ASANA_WORKSPACE_ID", "ASANA_DEFAULT_WORKSPACE_GID")
_ASANA_PROJECT_ENV_KEYS = ("ASANA_PROJECT_ID", "ASANA_DEFAULT_PROJECT_GID")


def get_env_value(*keys: str) -> str | None:
    """Get the first non-empty environment variable value from the given keys."""
    for key in keys:
        val = os.getenv(key)
        if val:
            trimmed = val.strip()
            if trimmed:
                return trimmed
    return None


def extract_resource_id(resource: Any) -> str | None:
    """Extract resource ID from different data structures."""
    if not isinstance(resource, dict):
        return None
    
    # Try 'id' field first (common)
    resource_id = resource.get('id')
    if resource_id:
        return str(resource_id)
    
    # Try 'gid' field (Asana-specific)
    gid = resource.get('gid')
    if gid:
        return str(gid)
    
    return None


def resolve_workspace_gid(current: Any, mcp_resources: Dict[str, Any]) -> str | None:
    """Resolves a workspace GID in priority order:
    1. If current is a dict with 'id' or 'gid', use that
    2. If current is a string, return it
    3. If current is [resource:key], resolve it
    4. Fallback to ENV
    5. Fallback to all resource IDs of type 'workspace'
    """
    # 1. dict
    if isinstance(current, dict):
        extracted = extract_resource_id(current)
        if extracted:
            return extracted
    
    # 2. string
    if isinstance(current, str):
        current_str = current.strip()
        # Might be a direct GID
        if current_str:
            return current_str
    
    # 4. ENV fallback
    env_workspace = get_env_value(*_ASANA_WORKSPACE_ENV_KEYS)
    if env_workspace:
        return env_workspace
    
    # 5. scan mcp_resources for type=workspace
    for key, resource in mcp_resources.items():
        if not isinstance(resource, dict):
            continue
        res_type = resource.get('type')
        if res_type and res_type.lower() == 'workspace':
            extracted = extract_resource_id(resource)
            if extracted:
                return extracted
    
    return None


def resolve_project_gid(current: Any, mcp_resources: Dict[str, Any]) -> str | None:
    """Resolves a project GID in priority order:
    1. If current is a dict with 'id' or 'gid', use that
    2. If current is a string, return it
    3. If current is [resource:key], resolve it
    4. Fallback to ENV
    5. Fallback to all resource IDs of type 'project'
    """
    # 1. dict
    if isinstance(current, dict):
        extracted = extract_resource_id(current)
        if extracted:
            return extracted
    
    # 2. string
    if isinstance(current, str):
        current_str = current.strip()
        if current_str:
            return current_str
    
    # 4. ENV fallback
    env_project = get_env_value(*_ASANA_PROJECT_ENV_KEYS)
    if env_project:
        return env_project
    
    # 5. scan mcp_resources for type=project
    for key, resource in mcp_resources.items():
        if not isinstance(resource, dict):
            continue
        res_type = resource.get('type')
        if res_type and res_type.lower() == 'project':
            extracted = extract_resource_id(resource)
            if extracted:
                return extracted
    
    return None


def sanitize_tool_params(
    tool_name: str,
    params: Dict[str, Any],
    mcp_resources: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Sanitizes tool parameters, with special handling for Asana tools.
    
    This function contains Asana-specific business logic that ideally should
    be moved to an Asana-specific adapter or plugin in the future.
    """
    sanitized = dict(params)

    # Generic: Convert comma-separated opt_fields string to list
    if "opt_fields" in sanitized and isinstance(sanitized["opt_fields"], str):
        raw = sanitized["opt_fields"].strip()
        if raw:
            fields = [field.strip() for field in raw.split(",") if field.strip()]
            sanitized["opt_fields"] = fields or [raw]

    # Asana-specific logic
    if tool_name.startswith("asana_"):
        if "workspace_gid" in sanitized:
            resolved_workspace = resolve_workspace_gid(
                sanitized.get("workspace_gid"), mcp_resources
            )
            if resolved_workspace:
                sanitized["workspace_gid"] = resolved_workspace
        elif "workspace" in sanitized and isinstance(sanitized["workspace"], dict):
            resolved_workspace = resolve_workspace_gid(
                sanitized.get("workspace"), mcp_resources
            )
            if resolved_workspace:
                sanitized["workspace"] = resolved_workspace

        if "project_gid" in sanitized:
            resolved_project = resolve_project_gid(
                sanitized.get("project_gid"), mcp_resources
            )
            if resolved_project:
                sanitized["project_gid"] = resolved_project

    return sanitized

