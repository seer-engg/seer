"""Utility functions for resource providers."""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional

from fastapi import HTTPException


def parse_depends_on(depends_on: Optional[str], *, error_detail: str = "Invalid depends_on JSON") -> Dict[str, Any]:
    """
    Parse depends_on JSON string into a dictionary.

    Args:
        depends_on: JSON string with dependent parameter values
        error_detail: Error message to use if parsing fails

    Returns:
        Parsed dictionary (empty if depends_on is None)

    Raises:
        HTTPException: If JSON parsing fails
    """
    if not depends_on:
        return {}
    try:
        parsed = json.loads(depends_on)
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=error_detail) from exc
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail=error_detail)
    return parsed


def parse_offset(page_token: Optional[str]) -> int:
    """
    Parse page_token string into integer offset.

    Args:
        page_token: String-encoded offset (e.g., "50")

    Returns:
        Integer offset (0 if page_token is None or invalid)
    """
    if not page_token:
        return 0
    try:
        return int(page_token)
    except ValueError:
        return 0


def extract_name(entry: Any, keys: Iterable[str]) -> Optional[str]:
    """
    Extract name from entry using multiple possible keys.

    Args:
        entry: Entry to extract name from (string, dict, or other)
        keys: Ordered list of keys to try (first match wins)

    Returns:
        Extracted name string, or None if not found
    """
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        for key in keys:
            value = entry.get(key)
            if value:
                return str(value)
    return None


def filter_entries(
    raw_entries: List[Any],
    *,
    name_keys: Iterable[str],
    query: Optional[str],
    skip_system: bool = False
) -> List[str]:
    """
    Filter entries by name and optionally skip system entries.

    Args:
        raw_entries: List of raw entries (strings or dicts)
        name_keys: Keys to extract name from (for dict entries)
        query: Optional search query (case-insensitive substring match)
        skip_system: Whether to skip system entries (information_schema, pg_*)

    Returns:
        List of filtered entry names
    """
    filtered: List[str] = []
    for entry in raw_entries:
        name = extract_name(entry, name_keys)
        if not name:
            continue
        if skip_system and (name == "information_schema" or name.startswith("pg_")):
            continue
        filtered.append(name)

    if query:
        lowered = query.lower()
        filtered = [name for name in filtered if lowered in name.lower()]

    return filtered


def paginate_items(
    names: List[str],
    *,
    page_size: int,
    offset: int,
    item_type: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Paginate a list of names into standard resource response format.

    Args:
        names: Full list of names to paginate
        page_size: Number of items per page
        offset: Starting offset for this page
        item_type: Type identifier for items (e.g., "schema", "table")
        description: Optional description to add to each item
        metadata: Optional metadata dict to merge into each item

    Returns:
        Standard resource response dict with items, next_page_token, and metadata
    """
    paged = names[offset:offset + page_size]
    items = []
    for name in paged:
        item: Dict[str, Any] = {
            "id": name,
            "name": name,
            "display_name": name,
            "type": item_type,
        }
        if description:
            item["description"] = description
        if metadata:
            item.update(metadata)
        items.append(item)

    next_page_token = str(offset + page_size) if offset + page_size < len(names) else None

    return {
        "items": items,
        "next_page_token": next_page_token,
        "supports_search": True,
        "supports_hierarchy": False,
    }


def resolve_resource_id(
    integration_resource_id: Optional[int],
    depends_on_values: Dict[str, Any],
) -> int:
    """
    Resolve integration_resource_id from explicit parameter or depends_on values.

    Args:
        integration_resource_id: Explicit resource ID (takes precedence)
        depends_on_values: Dependent values that may contain integration_resource_id

    Returns:
        Resolved resource ID

    Raises:
        HTTPException: If resource ID cannot be resolved
    """
    if integration_resource_id is not None:
        return integration_resource_id

    candidate = depends_on_values.get("integration_resource_id")
    if candidate is None:
        raise HTTPException(status_code=400, detail="integration_resource_id is required")

    try:
        return int(candidate)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="integration_resource_id is required") from exc
