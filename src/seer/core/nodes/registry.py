"""
Node type registry for workflow node implementations.

Similar to ToolRegistry, this provides a centralized registry for node types.
Each node type registers itself on module import, making it available for
runtime execution and type checking.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional

if TYPE_CHECKING:
    from seer.core.nodes.base import BaseNodeType


class NodeTypeRegistry:
    """
    Registry for node type implementations.

    Each node type (tool, agent, if, for_each, etc.) registers its implementation
    here. The runtime and compiler look up node types by their type literal
    to dispatch execution and type registration.
    """

    def __init__(self) -> None:
        self._node_types: Dict[str, "BaseNodeType"] = {}

    def register(self, node_type: "BaseNodeType") -> None:
        """
        Register a node type implementation.

        Args:
            node_type: The node type implementation to register
        """
        self._node_types[node_type.type_literal] = node_type

    def get(self, type_literal: str) -> Optional["BaseNodeType"]:
        """
        Get a node type implementation by its type literal.

        Args:
            type_literal: The node type identifier ('tool', 'agent', etc.)

        Returns:
            The node type implementation, or None if not found
        """
        return self._node_types.get(type_literal)

    def all_types(self) -> Dict[str, "BaseNodeType"]:
        """Return all registered node types."""
        return dict(self._node_types)

    def type_literals(self) -> list[str]:
        """Return list of all registered type literals."""
        return list(self._node_types.keys())


# Global singleton registry
node_type_registry = NodeTypeRegistry()


def register_node_type(node_type: "BaseNodeType") -> "BaseNodeType":
    """
    Register a node type in the global registry.

    Can be used as a decorator or function:

        @register_node_type
        class MyNodeType(BaseNodeType):
            ...

    Or:

        register_node_type(MyNodeType())

    Args:
        node_type: The node type implementation to register

    Returns:
        The same node type (for use as decorator)
    """
    node_type_registry.register(node_type)
    return node_type
