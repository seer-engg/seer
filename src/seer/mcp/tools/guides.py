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
    get_nexus_system_prompt,
)
from seer.agents.nexus.schema_context import generate_trigger_reference
from seer.logger import get_logger

logger = get_logger(__name__)


def get_started_impl() -> str:
    """
    Get the base Nexus system prompt for workflow building.

    This returns the foundational instructions without detailed block/graph/trigger
    documentation. Call get_workflow_guide() separately for detailed references.

    Returns:
        Base system prompt with core principles, tool discovery, and validation checklist.
    """
    return get_nexus_system_prompt()


@mcp.tool()
@track_mcp_tool("get_started")
async def get_started() -> str:
    """
    Get foundational instructions for building Seer workflows.

    Returns the base system prompt with core principles for workflow building:
    - Tool and trigger discovery patterns
    - Clarification question best practices
    - WorkflowSpec v2 schema overview
    - OAuth account selection flow
    - Validation checklist

    For detailed documentation about specific topics, call get_workflow_guide():
    - get_workflow_guide() - Full blocks + graph + triggers guide
    - get_workflow_guide(section="blocks") - Block types reference
    - get_workflow_guide(section="graph") - Graph structure and compilation
    - get_workflow_guide(section="triggers") - Trigger specification
    - get_workflow_guide(integration="gmail") - Integration-specific patterns

    Returns:
        Base workflow building instructions as Markdown.
    """
    return get_started_impl()


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
                 - "blocks" - Block types reference (tool, agent, mcp, if, for_each, etc.)
                 - "graph" - Graph structure and edge types
                 - "triggers" - Trigger specification and required fields
                 - None (default) - Returns blocks, graph, and trigger guides
        integration: Optional integration name for integration-specific guide.
                     Use "list" to see available integrations.
                     Examples: "gmail", "slack", "supabase"

    Returns:
        Markdown documentation with schemas, examples, and best practices.

    Examples:
        get_workflow_guide()                      # Full blocks + graph + triggers guide
        get_workflow_guide(section="blocks")      # Just block types
        get_workflow_guide(section="triggers")    # Trigger spec and required fields
        get_workflow_guide(integration="gmail")   # Gmail tools and patterns
        get_workflow_guide(integration="list")    # List available integrations
    """
    # If integration requested, return integration-specific guide
    if integration:
        return _get_integration_guide(integration)

    # Return section-specific or combined guide
    if section == "blocks":
        return get_primitive_blocks_guide()
    if section == "graph":
        return get_graph_structure_guide()
    if section == "triggers":
        return generate_trigger_reference()

    # Return combined guide (default)
    blocks = get_primitive_blocks_guide()
    graph = get_graph_structure_guide()
    triggers = generate_trigger_reference()
    return f"{blocks}\n\n---\n\n{graph}\n\n---\n\n{triggers}"
