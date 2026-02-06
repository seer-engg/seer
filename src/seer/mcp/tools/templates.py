"""
MCP template tools for retrieving workflow templates.

Reuses existing template logic from seer.agents.nexus.schema_context.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

import json
from typing import Any, Dict, List

from seer.mcp.server import mcp
from seer.agents.nexus.schema_context import get_workflow_templates
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
        templates = get_workflow_templates()
        query_lower = query.lower()

        matches: List[Dict[str, Any]] = []
        for template in templates:
            name = template.get("name", "").lower()
            tags = [t.lower() for t in template.get("tags", [])]
            description = template.get("description", "").lower()

            # Match if query appears in name, tags, or description
            if (query_lower in name or
                any(query_lower in tag for tag in tags) or
                query_lower in description):
                matches.append(template)

        if not matches:
            available_templates = [
                {"name": t.get("name"), "tags": t.get("tags", [])}
                for t in templates
            ]
            return json.dumps({
                "query": query,
                "matches": [],
                "message": f"No templates found matching '{query}'",
                "available_templates": available_templates,
                "suggestion": "Try searching with integration names (gmail, supabase, slack) or action words (welcome, notification, report)"
            })

        # Return matches with full specs
        results = []
        for match in matches:
            results.append({
                "name": match.get("name"),
                "description": match.get("description"),
                "tags": match.get("tags"),
                "customization_guide": match.get("customization_guide"),
                "spec": match.get("spec")
            })

        return json.dumps({
            "query": query,
            "matches": results,
            "count": len(results),
            "message": f"Found {len(results)} template(s) matching '{query}'"
        }, indent=2)

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
        templates = get_workflow_templates()

        template_list = []
        for template in templates:
            template_list.append({
                "name": template.get("name"),
                "description": template.get("description"),
                "tags": template.get("tags", []),
            })

        return json.dumps({
            "templates": template_list,
            "total": len(template_list),
        }, indent=2)

    except Exception as e:  # pylint: disable=broad-exception-caught # Reason: Return friendly JSON error
        logger.exception("Error listing templates: %s", e)
        return json.dumps({
            "templates": [],
            "error": str(e)
        })
