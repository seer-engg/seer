"""
Validate OAuth connection requirements for tool nodes in a workflow.

This module handles multi-account scenarios where users may have multiple OAuth
connections for the same provider (e.g., personal + work Gmail accounts).

Validation rules:
- If a tool requires OAuth and user has 0 connections: error
- If a tool requires OAuth and user has exactly 1 connection: auto-select (backward compatible)
- If a tool requires OAuth and user has >1 connections:
  - If node specifies connection_id: validate it exists and is active
  - If node does NOT specify connection_id: error (ambiguous selection)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional

from seer.core.errors import ErrorCode, NodeError, ValidationPhaseError
from seer.core.schema.models import ToolNode, WorkflowSpec
from seer.logger import get_logger
from seer.tools.base import get_tool

if TYPE_CHECKING:
    from seer.database import OAuthConnection, User

logger = get_logger(__name__)


@dataclass
class ConnectionValidationResult:
    """Result of connection validation for a workflow."""

    errors: List[NodeError] = field(default_factory=list)
    resolved_connections: Dict[str, int] = field(default_factory=dict)
    """Maps node_id -> connection_id for auto-resolved single-account cases."""


async def validate_tool_connections(
    spec: WorkflowSpec,
    user: "User",
    organization_id: Optional[int] = None,
) -> ConnectionValidationResult:
    """
    Validate that all tool nodes have valid OAuth connections.

    This validation ensures:
    1. Tools requiring OAuth have at least one matching connection
    2. Multi-account scenarios require explicit connection_id selection
    3. Specified connection_ids exist and are active

    Args:
        spec: Workflow specification to validate
        user: User whose connections to check
        organization_id: Optional organization ID to include shared connections

    Returns:
        ConnectionValidationResult with errors and auto-resolved connections
    """
    # Import here to avoid circular dependency
    from seer.database import OAuthConnection  # pylint: disable=import-outside-toplevel
    from tortoise.expressions import Q  # pylint: disable=import-outside-toplevel  # Reason: avoid import overhead when not needed

    result = ConnectionValidationResult()

    # Collect all tool nodes that require OAuth
    oauth_tools = _collect_oauth_tool_nodes(spec)
    if not oauth_tools:
        return result

    # Group nodes by provider
    nodes_by_provider: Dict[str, List[tuple[ToolNode, str]]] = {}
    for node, provider in oauth_tools:
        nodes_by_provider.setdefault(provider, []).append((node, provider))

    # Fetch user's connections + organization-shared connections
    query = Q(user=user, status="active")
    if organization_id:
        query |= Q(shared_with_organization_id=organization_id, status="active")

    connections = await OAuthConnection.filter(query).all()
    connections_by_provider: Dict[str, List[OAuthConnection]] = {}
    for conn in connections:
        connections_by_provider.setdefault(conn.provider, []).append(conn)

    # Validate each provider's nodes
    for provider, nodes in nodes_by_provider.items():
        provider_connections = connections_by_provider.get(provider, [])

        for node, _ in nodes:
            _validate_node_connection(
                node=node,
                provider=provider,
                connections=provider_connections,
                result=result,
            )

    return result


def _collect_oauth_tool_nodes(spec: WorkflowSpec) -> List[tuple[ToolNode, str]]:
    """
    Collect all tool nodes that require OAuth authentication.

    Returns:
        List of (ToolNode, provider) tuples for nodes requiring OAuth
    """
    oauth_nodes: List[tuple[ToolNode, str]] = []

    for node in spec.nodes:
        if not isinstance(node, ToolNode):
            continue

        tool = get_tool(node.tool)
        if tool is None:
            # Tool not found - this is handled by other validation
            continue

        if not tool.required_scopes:
            # Tool doesn't require OAuth
            continue

        provider = tool.provider
        if not provider:
            # No provider - can't validate connections
            logger.warning(
                "Tool '%s' has required_scopes but no provider defined",
                node.tool,
            )
            continue

        oauth_nodes.append((node, provider))

    return oauth_nodes


def _validate_node_connection(
    node: ToolNode,
    provider: str,
    connections: List["OAuthConnection"],
    result: ConnectionValidationResult,
) -> None:
    """
    Validate a single tool node's connection requirements.

    Updates result with errors or resolved connection.
    """
    connection_count = len(connections)

    # Case 1: No connections for this provider
    if connection_count == 0:
        result.errors.append(NodeError(
            code=ErrorCode.VALIDATION_ERROR,
            message=f"No {provider} account connected. Please connect an account first.",
            node_id=node.id,
            location="connection_id",
        ))
        return

    # Case 2: Node specifies connection_id - validate it exists
    if node.connection_id is not None:
        matching_conn = next(
            (c for c in connections if c.id == node.connection_id),
            None,
        )
        if matching_conn is None:
            # Check if it's an inactive or non-existent connection
            account_list = ", ".join(
                c.provider_account_id or f"ID:{c.id}"
                for c in connections
            )
            result.errors.append(NodeError(
                code=ErrorCode.VALIDATION_ERROR,
                message=(
                    f"OAuth connection '{node.connection_id}' not found or inactive. "
                    f"Available {provider} accounts: {account_list}"
                ),
                node_id=node.id,
                location="connection_id",
            ))
        # If found, connection_id is already set on the node - no action needed
        return

    # Case 3: Single connection - auto-select (backward compatible)
    if connection_count == 1:
        result.resolved_connections[node.id] = connections[0].id
        return

    # Case 4: Multiple connections, no selection - error
    account_list = ", ".join(
        c.provider_account_id or f"ID:{c.id}"
        for c in connections
    )
    result.errors.append(NodeError(
        code=ErrorCode.VALIDATION_ERROR,
        message=(
            f"Multiple {provider} accounts connected ({account_list}). "
            f"Please select which account to use for tool '{node.tool}'."
        ),
        node_id=node.id,
        location="connection_id",
    ))


async def validate_connections_and_raise(
    spec: WorkflowSpec,
    user: "User",
    organization_id: Optional[int] = None,
) -> Dict[str, int]:
    """
    Validate connections and raise ValidationPhaseError if any errors found.

    Args:
        spec: Workflow specification to validate
        user: User whose connections to check
        organization_id: Optional organization ID to include shared connections

    Returns:
        Dict mapping node_id -> connection_id for auto-resolved connections

    Raises:
        ValidationPhaseError: If any connection validation errors found
    """
    result = await validate_tool_connections(spec, user, organization_id)

    if result.errors:
        messages = []
        for err in result.errors:
            if err.node_id:
                messages.append(f"{err.node_id}: {err.message}")
            else:
                messages.append(err.message)
        raise ValidationPhaseError("\n".join(messages), errors=result.errors)

    return result.resolved_connections


__all__ = [
    "ConnectionValidationResult",
    "validate_tool_connections",
    "validate_connections_and_raise",
]
