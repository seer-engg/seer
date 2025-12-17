"""
Unified Agent Context for shared state across agents.

This module defines AgentContext, which contains all shared state
that is passed between agents (Eval → Codex → Eval).

By extracting shared fields into a single context object, we:
1. Eliminate duplication between EvalAgentState and CodexState
2. Make handoffs between agents cleaner
3. Provide a single source of truth for shared data
4. Make it easier to add new shared fields

Usage:
    # In Eval Agent
    context = AgentContext(
        user_context=user_ctx,
        github_context=github_ctx,
        ...
    )
    state = EvalAgentState(context=context, ...)
    
    # In handoff to Codex
    codex_input = CodexInput(
        context=state.context,  # Just pass the whole context
        ...
    )
"""
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, ConfigDict
from shared.tools import ToolEntry


if TYPE_CHECKING:
    from shared.schema import UserContext, GithubContext, SandboxContext

class IntegrationItem(BaseModel):
    """
    Integration item information
    """

    id: Optional[str] = None
    name: str
    mode: Optional[str] = None
class Integration(BaseModel):
    """
    Integration information
    """
    github: Optional[IntegrationItem] = None
    googledrive: Optional[IntegrationItem] = None
    asana: Optional[IntegrationItem] = None
    gmail: Optional[IntegrationItem] = None
    sandbox: Optional[IntegrationItem] = None

class AgentContext(BaseModel):
    """
    Immutable shared context for all agents.
    
    This contains all state that is shared between the Eval Agent and Codex Agent.
    When one agent hands off to another, this context is passed along.
    
    Fields:
        user_context: User preferences and configuration
        github_context: GitHub repository information
        sandbox_context: E2B sandbox connection details
        target_agent_version: Version number of target agent being evaluated
        mcp_services: List of MCP service names to load tools from
        mcp_resources: Dict of created MCP resources (for cleanup)
    """
    
    # User information
    user_context: Optional["UserContext"] = Field(
        default=None,
        description="User preferences and configuration"
    )
    
    # GitHub repository information
    github_context: Optional["GithubContext"] = Field(
        default=None,
        description="GitHub repository being worked on"
    )
    
    # Sandbox environment
    sandbox_context: Optional["SandboxContext"] = Field(
        default=None,
        description="E2B sandbox connection details"
    )

    agent_name: str = Field(
        default="",
        description="The name of the agent"
    )
    
    # Target agent version tracking
    target_agent_version: int = Field(
        default=0,
        description="Version number of the target agent being evaluated/developed"
    )
    
    # MCP (Model Context Protocol) configuration
    mcp_services: List[str] = Field(
        default_factory=list,
        description="List of MCP service names (e.g., ['asana', 'github'])"
    )
    
    mcp_resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="MCP resources created during evaluation (for cleanup)"
    )

    # Functional requirements for the target agent
    functional_requirements: List[str] = Field(
        default_factory=list,
        description="Functional requirements for the target agent, aligned with the user"
    )

    # Tools
    tool_entries: Dict[str, ToolEntry] = Field(
        default_factory=dict, 
        description="Subset of tools selected for this evaluation run"
    )
    integrations: Integration = Field(
        default=Integration(),
        description="Integrations selected for this evaluation run"
    )
    user_id: str = Field(
        default="",
        description="The ID of the user"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


__all__ = ["AgentContext"]

