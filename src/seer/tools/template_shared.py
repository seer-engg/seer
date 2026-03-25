"""
Shared template search logic for workflow templates.
Used by both Nexus agent tools and MCP tools.
"""
from __future__ import annotations

from typing import Any, Dict, List

from seer.database.template_models import WorkflowTemplate


async def search_templates(query: str) -> Dict[str, Any]:
    """
    Search workflow templates by name, tags, or description.

    Args:
        query: Template name or tag to search for (e.g., "supabase gmail", "welcome")

    Returns:
        Dict with matches, count, message, and optionally available_templates/suggestion
    """
    query_lower = query.lower()
    query_words = query_lower.split()

    # Fetch all templates (small table, filter in Python for flexible matching)
    templates = await WorkflowTemplate.all()

    matches: List[Dict[str, Any]] = []
    for template in templates:
        name = template.name.lower()
        tags = [t.lower() for t in (template.tags or [])]
        description = template.description.lower()
        searchable = f"{name} {' '.join(tags)} {description}"

        # Match if any query word appears in searchable text
        if any(word in searchable for word in query_words):
            matches.append({
                "name": template.name,
                "description": template.description,
                "tags": template.tags or [],
                "slug": template.slug,
                "spec": template.spec,
                "category": template.category.value if template.category else None,
            })

    if not matches:
        all_templates = await WorkflowTemplate.all()
        available_templates = [
            {"name": t.name, "tags": t.tags or []}
            for t in all_templates
        ]
        return {
            "query": query,
            "matches": [],
            "message": f"No templates found matching '{query}'",
            "available_templates": available_templates,
            "suggestion": "Try searching with integration names (gmail, supabase, slack) or action words (welcome, notification, report)",
        }

    return {
        "query": query,
        "matches": matches,
        "count": len(matches),
        "message": f"Found {len(matches)} template(s) matching '{query}'",
    }


async def list_all_templates() -> Dict[str, Any]:
    """
    List all available workflow templates.

    Returns:
        Dict with templates list and total count
    """
    templates = await WorkflowTemplate.all()

    template_list = [
        {
            "name": t.name,
            "description": t.description,
            "tags": t.tags or [],
            "slug": t.slug,
            "category": t.category.value if t.category else None,
        }
        for t in templates
    ]

    return {
        "templates": template_list,
        "total": len(template_list),
    }
