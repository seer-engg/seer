"""
Tool name normalization utilities.
Handles conversion of various tool name formats to canonical forms.
"""
import re


def canonicalize_tool_name(raw_tool: str, service_hint: str | None = None) -> str:
    """
    Normalize various tool name spellings into lookup-friendly keys.
    
    Args:
        raw_tool: The raw tool name (e.g., "ASANA_CREATE_TASK", "asana.create_task")
        service_hint: Optional service name to prefix if tool has no service prefix
        
    Returns:
        Canonical tool name (e.g., "asana_create_task")
        
    Examples:
        >>> canonicalize_tool_name("ASANA_CREATE_TASK")
        'asana_create_task'
        >>> canonicalize_tool_name("asana.create_task")
        'asana_create_task'
        >>> canonicalize_tool_name("create_task", service_hint="asana")
        'asana_create_task'
    """
    if not raw_tool:
        return raw_tool

    normalized = raw_tool.strip()
    if not normalized:
        return normalized

    lowered = normalized.lower()
    if lowered.startswith("system."):
        return lowered

    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace(".", "_")
    normalized = normalized.strip("_")
    normalized = normalized.lower()

    if "_" in normalized:
        return normalized

    if service_hint:
        prefix = service_hint.strip().lower()
        if prefix:
            return f"{prefix}_{normalized}"

    return normalized

