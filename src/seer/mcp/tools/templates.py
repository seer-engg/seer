"""
MCP template tools for retrieving workflow templates.

Uses shared template search logic from seer.tools.template_shared.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json

from seer.mcp.server import mcp
from seer.tools.template_shared import search_templates, list_all_templates
from seer.logger import get_logger

logger = get_logger(__name__)


@mcp.tool()
async def get_workflow_template(query: str) -> str:
    """
    Retrieve workflow templates by name or tags to use as starting points.

    Use this when you need a pre-built workflow pattern that can be customized.
    Templates include common patterns like "supabase signup to email", "slack notification", etc.

    Args:
        query: Template name or tag to search for (e.g., "supabase gmail", "welcome", "slack notification")

    Returns:
        JSON with matching template(s) including full spec that can be customized
    """
    try:
        result = search_templates(query)
        return json.dumps(result, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error retrieving template: %s", e)
        return json.dumps({
            "query": query,
            "matches": [],
            "error": str(e)
        })


@mcp.tool()
async def list_workflow_templates() -> str:
    """
    List all available workflow templates.

    Use this to see all pre-built workflow patterns that can be used as starting points.

    Returns:
        JSON with list of all templates including names, descriptions, and tags
    """
    try:
        result = list_all_templates()
        return json.dumps(result, indent=2)
    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing templates: %s", e)
        return json.dumps({
            "templates": [],
            "error": str(e)
        })
