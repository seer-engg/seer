"""
Shared template search logic for workflow templates.
Used by both Nexus agent tools and MCP tools.
"""
from typing import Any, Dict, List

from seer.agents.nexus.schema_context import get_workflow_templates


def search_templates(query: str) -> Dict[str, Any]:
    """
    Search workflow templates by name, tags, or description.

    Args:
        query: Template name or tag to search for (e.g., "supabase gmail", "welcome")

    Returns:
        Dict with matches, count, message, and optionally available_templates/suggestion
    """
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
            matches.append({
                "name": template.get("name"),
                "description": template.get("description"),
                "tags": template.get("tags"),
                "customization_guide": template.get("customization_guide"),
                "spec": template.get("spec")
            })

    if not matches:
        available_templates = [
            {"name": t.get("name"), "tags": t.get("tags", [])}
            for t in templates
        ]
        return {
            "query": query,
            "matches": [],
            "message": f"No templates found matching '{query}'",
            "available_templates": available_templates,
            "suggestion": "Try searching with integration names (gmail, supabase, slack) or action words (welcome, notification, report)"
        }

    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "message": f"Found {len(matches)} template(s) matching '{query}'"
    }


def list_all_templates() -> Dict[str, Any]:
    """
    List all available workflow templates.

    Returns:
        Dict with templates list and total count
    """
    templates = get_workflow_templates()

    template_list = [
        {
            "name": template.get("name"),
            "description": template.get("description"),
            "tags": template.get("tags", []),
        }
        for template in templates
    ]

    return {
        "templates": template_list,
        "total": len(template_list),
    }
