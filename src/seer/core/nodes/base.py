"""
Abstract base class for node type implementations.

Each node type (tool, llm, if, for_each, etc.) implements this interface,
centralizing model definition, execution, type registration, and routing
in a single file per node type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional

from seer.core.schema.models import NodeBase

if TYPE_CHECKING:
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.registry.mcp_client_registry import MCPClientRegistry
    from seer.core.registry.tool_registry import ToolRegistry
    from seer.core.runtime.context import WorkflowRuntimeContext
    from seer.core.runtime.state import WorkflowState
    from seer.core.schema.models import Edge, Node
    from seer.core.schema.schema_registry import SchemaRegistry


@dataclass
class NodeExecutionContext:
    """Context passed to node execution methods."""
    state: "WorkflowState"
    config: Mapping[str, Any]
    locals_ctx: Mapping[str, Any] | None
    runtime_context: Optional["WorkflowRuntimeContext"]
    # Loop maps for trace key generation
    loop_body_map: Dict[str, str] | None = None
    nested_loop_parents: Dict[str, str] | None = None
    # Trigger data for expression evaluation
    trigger: Mapping[str, Any] | None = None


@dataclass
class TypeRegistrationContext:
    """Context for type environment registration."""
    schema_registry: "SchemaRegistry"
    tool_registry: "ToolRegistry"
    mcp_client_registry: Optional["MCPClientRegistry"] = None


@dataclass
class RoutingResult:
    """Result of routing logic for control flow nodes."""
    is_conditional: bool
    router_func: Any  # Callable[[dict], str]
    path_map: Dict[str, str]


class BaseNodeType(ABC):
    """
    Abstract base class for node type implementations.

    Each node type (tool, agent, if, for_each, hitl, browser, mcp) implements
    this interface. This centralizes all node-related logic:
    - Pydantic model class for validation
    - Execution logic (sync and async)
    - Type environment registration
    - Optional: routing for control flow nodes
    - Optional: loop detection for ForEachNode
    """

    @property
    @abstractmethod
    def type_literal(self) -> str:
        """Return the node type identifier ('tool', 'agent', 'if', etc.)."""

    @property
    @abstractmethod
    def model_class(self) -> type[NodeBase]:
        """Return the Pydantic model class for this node type."""

    @abstractmethod
    async def execute_async(
        self,
        node: "Node",
        ctx: NodeExecutionContext,
        services: Any,  # RuntimeServices from nodes.py
    ) -> Dict[str, Any]:
        """
        Execute the node asynchronously and return state updates.

        Args:
            node: The node instance to execute
            ctx: Execution context with state, config, and runtime info
            services: Runtime services (registries, type env, etc.)

        Returns:
            Dictionary of state updates to apply
        """

    @abstractmethod
    def register_type_sync(
        self,
        node: "Node",
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register node's output schema in the type environment (sync path).

        Called during workflow compilation to build type information.
        For MCP nodes, this registers a generic schema since async validation
        is not available in the sync path.
        """

    async def register_type_async(
        self,
        node: "Node",
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        Register node's output schema in the type environment (async path).

        Default implementation calls sync path. Override for nodes needing
        async validation (like MCPNode which validates against remote server).
        """
        self.register_type_sync(node, env, ctx)

    def get_analytics_properties(  # pylint: disable=unused-argument  # Reason: Optional override for richer per-type analytics
        self,
        node: "Node",
        ctx: "NodeExecutionContext",
    ) -> Dict[str, Any]:
        """
        Return node-type-specific properties for PostHog analytics.

        Override in concrete node types to add semantically meaningful
        properties (e.g., tool_name for ToolNode, http_url for HttpNode).
        The default returns an empty dict so all node types get baseline tracking.
        """
        return {}

    def get_routing(  # pylint: disable=unused-argument  # Reason: Optional override for control flow nodes
        self,
        node: "Node",
        outgoing_edges: List["Edge"],
    ) -> Optional[RoutingResult]:
        """
        Return routing configuration for control flow nodes.

        Override for IfNode (conditional routing) and ForEachNode (loop routing).
        Returns None for nodes that use simple sequential edges.
        """
        return None


# ============================================================================
# Shared Utilities for Node Implementations
# ============================================================================

def build_trace_entry(
    node_id: str,
    node_type: str,
    inputs: Dict[str, Any],
    output: Any,
    status: str = "succeeded",
    **extra: Any,
) -> Dict[str, Any]:
    """
    Build a trace entry dictionary for node execution.

    All nodes should use this to ensure consistent trace format.
    """
    trace = {
        "node_id": node_id,
        "node_type": node_type,
        "inputs": inputs,
        "output": output,
        "output_key": node_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": status,
    }
    trace.update(extra)
    return trace


def build_error_trace(
    node_id: str,
    node_type: str,
    inputs: Dict[str, Any],
    exc: Exception,
) -> Dict[str, Any]:
    """Build a trace entry for a failed node execution."""
    return {
        "node_id": node_id,
        "node_type": node_type,
        "inputs": inputs,
        "error": {
            "type": exc.__class__.__name__,
            "message": str(exc),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
    }


# ============================================================================
# Shared Utilities for Node Implementations (moved from runtime/nodes.py)
# ============================================================================

def build_eval_context(
    state: "WorkflowState",
    config: Mapping[str, Any],
    locals_ctx: Mapping[str, Any] | None,
    trigger: Mapping[str, Any] | None = None,
) -> "EvaluationContext":
    """
    Build evaluation context for expression evaluation.

    Args:
        state: Current workflow state
        config: Workflow configuration
        locals_ctx: Local variables (e.g., loop variables)
        trigger: Trigger event data

    Returns:
        EvaluationContext for expression evaluation
    """
    # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
    from seer.core.expr.evaluator import EvaluationContext
    from seer.core.runtime.state import INTERNAL_STATE_PREFIX

    visible_state = {k: v for k, v in state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
    return EvaluationContext(
        state=visible_state,
        locals=locals_ctx or {},
        config=config,
        trigger=trigger,
    )


def get_trace_key(
    node_id: str,
    state: "WorkflowState",
    loop_body_map: Dict[str, str],
    nested_loop_parents: Dict[str, str],
) -> str:
    """
    Generate loop-aware trace key for a node.

    For nodes inside loops, the trace key includes iteration indices to prevent
    collisions between different loop iterations.

    BUG FIX (2024-02): Original trace key generation only added ONE iteration suffix
    from the immediate parent loop. For nested loops, this caused collisions.
    Solution: Build full iteration path from outermost to innermost loop.

    Args:
        node_id: The node's unique ID
        state: Current workflow state (contains loop iteration state)
        loop_body_map: Mapping of node_id to parent loop_id
        nested_loop_parents: Mapping of inner_loop_id to outer_loop_id

    Returns:
        Trace key like "_trace_node_id" or "_trace_node_id_iter_0_iter_1" for nested loops
    """
    parent_loop_id = loop_body_map.get(node_id)
    if not parent_loop_id:
        return f"_trace_{node_id}"

    # Build full iteration path from innermost to outermost loop
    iteration_suffixes: List[str] = []
    current_loop_id = parent_loop_id

    while current_loop_id:
        loop_state_key = f"_loop_{current_loop_id}"
        loop_state = state.get(loop_state_key)
        if loop_state and isinstance(loop_state, dict):
            current_index = loop_state.get("current_index", 0)
            iteration_suffixes.append(f"_iter_{current_index}")

        # Walk up to parent loop (if this loop is nested)
        current_loop_id = nested_loop_parents.get(current_loop_id)

    # Reverse to get outermost first (e.g., _iter_0_iter_1 for outer=0, inner=1)
    iteration_suffixes.reverse()

    return f"_trace_{node_id}{''.join(iteration_suffixes)}"


def evaluate_inputs(
    inputs: Dict[str, Any],
    eval_ctx: "EvaluationContext",
) -> Dict[str, Any]:
    """
    Evaluate input expressions against evaluation context.

    Captures evaluation errors without raising, storing error info in the result.
    This allows partial execution and debugging when some inputs fail.

    Args:
        inputs: Dict of input expressions (may contain ${...} references)
        eval_ctx: Evaluation context with state, config, etc.

    Returns:
        Dict of evaluated values (or error info for failed evaluations)
    """
    from seer.core.expr.evaluator import evaluate_value  # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports

    evaluated: Dict[str, Any] = {}
    for key, expr in inputs.items():
        try:
            evaluated[key] = evaluate_value(eval_ctx, expr)
        except Exception as e:  # pylint: disable=broad-exception-caught  # Reason: Capture eval error in result instead of failing
            evaluated[key] = {"__error__": str(e), "__expression__": expr}
    return evaluated


def write_error_trace(  # pylint: disable=too-many-positional-arguments  # Reason: Error trace requires multiple context fields
    node_id: str,
    node_type: str,
    inputs: Dict[str, Any],
    exc: Exception,
    state: "WorkflowState",
    loop_body_map: Dict[str, str],
    nested_loop_parents: Dict[str, str],
) -> Dict[str, Any]:
    """
    Write a partial trace with error info when node execution fails.

    Args:
        node_id: The node's unique ID
        node_type: The type of node (tool, llm, etc.)
        inputs: The evaluated inputs that were passed to execution
        exc: The exception that occurred
        state: Current workflow state
        loop_body_map: Mapping of node_id to parent loop_id
        nested_loop_parents: Mapping of inner_loop_id to outer_loop_id

    Returns:
        Dict with trace key mapping to error trace data
    """
    trace_key = get_trace_key(node_id, state, loop_body_map, nested_loop_parents)

    return {
        trace_key: {
            "node_id": node_id,
            "node_type": node_type,
            "inputs": inputs,
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
        }
    }
