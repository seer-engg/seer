"""Helpers for presenting MCP resource hints to LLMs."""
from __future__ import annotations

from typing import Any, Dict, List


def format_resource_hints(mcp_resources: Dict[str, Any]) -> str:
    """Render stored MCP resources into a human-readable bullet list."""

    if not mcp_resources:
        return "None provided. Prefer using [var:...] or [resource:...] tokens for runtime values."

    lines: List[str] = []
    for name in sorted(mcp_resources.keys()):
        payload = mcp_resources[name]
        if not isinstance(payload, dict):
            continue
        data = payload.get("data")
        if isinstance(data, dict):
            payload = data
        identifier = payload.get("id") or payload.get("gid")
        descriptor = payload.get("name") or payload.get("full_name")
        if identifier or descriptor:
            pretty = f"- {name}:"
            if descriptor:
                pretty += f" name={descriptor}"
            if identifier:
                pretty += f" id={identifier}"
            lines.append(pretty)
    return "\n".join(lines) if lines else "Resources available but no IDs detected."
