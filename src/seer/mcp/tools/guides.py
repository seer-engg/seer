"""
MCP tools for retrieving workflow documentation and guides.

Provides on-demand access to workflow building documentation that agents
can call when they need detailed guidance.
"""
# pylint: disable=cyclic-import # Reason: mcp server module registers tools via imports

from __future__ import annotations

from typing import Optional

from seer.mcp.server import mcp
from seer.mcp.tracking import track_mcp_tool
from seer.prompts import (
    get_primitive_blocks_guide,
    get_graph_structure_guide,
    get_skill_guide,
    list_available_skills,
)
from seer.logger import get_logger

logger = get_logger(__name__)


def _get_integration_guide(integration: str) -> str:
    """Get integration-specific guide or list available integrations."""
    if integration.lower() == "list":
        skills = list_available_skills()
        if not skills:
            return "No integration guides available."
        return "# Available Integration Guides\n\n" + "\n".join(f"- {s}" for s in sorted(skills))

    guide = get_skill_guide(integration.lower())
    if guide is None:
        skills = list_available_skills()
        available = ", ".join(skills) if skills else "none"
        return f"Integration guide not found: {integration}\n\nAvailable: {available}"
    return guide


@mcp.tool()
@track_mcp_tool("get_workflow_guide")
async def get_workflow_guide(
    section: Optional[str] = None,
    integration: Optional[str] = None,
) -> str:
    """
    Get comprehensive workflow building documentation.

    Returns detailed guidance for building Seer workflows. Call this when you need
    to understand workflow structure, block types, or integration-specific patterns.

    Args:
        section: Optional section to retrieve:
                 - "blocks" - Block types reference (tool, llm, if, for_each)
                 - "graph" - Graph structure and edge types
                 - None (default) - Returns both blocks and graph guides
        integration: Optional integration name for integration-specific guide.
                     Use "list" to see available integrations.
                     Examples: "gmail", "slack", "supabase"

    Returns:
        Markdown documentation with schemas, examples, and best practices.

    Examples:
        get_workflow_guide()                    # Full blocks + graph guide
        get_workflow_guide(section="blocks")    # Just block types
        get_workflow_guide(integration="gmail") # Gmail tools and patterns
        get_workflow_guide(integration="list")  # List available integrations
    """
    # If integration requested, return integration-specific guide
    if integration:
        return _get_integration_guide(integration)

    # Return section-specific or combined guide
    if section == "blocks":
        return get_primitive_blocks_guide()
    if section == "graph":
        return get_graph_structure_guide()

    # Return combined guide (default)
    blocks = get_primitive_blocks_guide()
    graph = get_graph_structure_guide()
    return f"{blocks}\n\n---\n\n{graph}"
