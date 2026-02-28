"""
Node type implementations for workflow execution.

This package provides a "one file per node type" pattern where each node type
(tool, llm, if, for_each, hitl, browser, mcp) has its own file containing:
- Execution logic
- Type registration
- Routing logic (for control flow nodes)

The Pydantic model classes remain in schema/models.py (canonical location) to
avoid circular imports, and are re-exported here for convenience.

All node types auto-register with the node_type_registry on import.

Usage:
    from seer.core.nodes import node_type_registry
    from seer.core.nodes import ToolNode, LLMNode, Node  # Re-exported models
"""

from seer.core.nodes.base import (
    BaseNodeType,
    NodeExecutionContext,
    TypeRegistrationContext,
    RoutingResult,
    build_trace_entry,
    build_error_trace,
    # Shared utilities for node implementations
    build_eval_context,
    get_trace_key,
    evaluate_inputs,
    write_error_trace,
)
from seer.core.nodes.registry import (
    NodeTypeRegistry,
    node_type_registry,
    register_node_type,
)

# Import node type implementations to trigger auto-registration
# Each module registers its node type when imported
from seer.core.nodes.tool_node import ToolNodeType
from seer.core.nodes.llm_node import LLMNodeType
from seer.core.nodes.mcp_node import MCPNodeType
from seer.core.nodes.if_node import IfNodeType
from seer.core.nodes.for_each_node import ForEachNodeType
from seer.core.nodes.hitl_node import HITLNodeType
from seer.core.nodes.browser_node import BrowserNodeType
from seer.core.nodes.image_gen_node import ImageGenNodeType
from seer.core.nodes.agent_node import AgentNodeType

# Re-export Pydantic model classes from schema/models.py (canonical location)
from seer.core.schema.models import (
    # Node models
    ToolNode,
    LLMNode,
    MCPNode,
    IfNode,
    ForEachNode,
    HITLNode,
    BrowserNode,
    ImageGenNode,
    AgentNode,
    Node,
    # HITL supporting types
    HITLInputType,
    HITLInputField,
    HITLInputOption,
    HITLDisplayItem,
    HITLDeliveryChannel,
    DeliveryChannelType,
    GmailDeliveryConfig,
)

__all__ = [
    # Base classes and utilities
    "BaseNodeType",
    "NodeExecutionContext",
    "TypeRegistrationContext",
    "RoutingResult",
    "build_trace_entry",
    "build_error_trace",
    # Shared utilities for node implementations
    "build_eval_context",
    "get_trace_key",
    "evaluate_inputs",
    "write_error_trace",
    # Registry
    "NodeTypeRegistry",
    "node_type_registry",
    "register_node_type",
    # Node type implementations
    "ToolNodeType",
    "LLMNodeType",
    "MCPNodeType",
    "IfNodeType",
    "ForEachNodeType",
    "HITLNodeType",
    "BrowserNodeType",
    "ImageGenNodeType",
    "AgentNodeType",
    # Node models (Pydantic classes) - re-exported from schema/models.py
    "ToolNode",
    "LLMNode",
    "MCPNode",
    "IfNode",
    "ForEachNode",
    "HITLNode",
    "BrowserNode",
    "ImageGenNode",
    "AgentNode",
    "Node",
    # HITL supporting types
    "HITLInputType",
    "HITLInputField",
    "HITLInputOption",
    "HITLDisplayItem",
    "HITLDeliveryChannel",
    "DeliveryChannelType",
    "GmailDeliveryConfig",
]
