"""
ForEachNode - Loop iteration over lists.

Iterates over items in a list, executing the loop body for each item.
Handles nested loops with proper state isolation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from seer.core.errors import ExecutionError
from seer.core.expr.typecheck import schema_from_output_contract
from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, RoutingResult, TypeRegistrationContext
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import EdgeType, ForEachNode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import Edge, NodeBase


# =============================================================================
# Routing Functions
# =============================================================================

def _build_loop_router(node_id: str, body_target: Optional[str], exit_target: Optional[str]):
    """
    Build a routing function for ForEachNode conditional edges.

    The ForEachNode runner stores iteration state in state[f"_loop_{node_id}"].
    This router checks has_more_iterations and returns body or exit target.
    """
    def route_loop(state: dict) -> str:
        from langgraph.graph import END  # pylint: disable=import-outside-toplevel  # Reason: Late import for router closure
        loop_state = state.get(f"_loop_{node_id}", {})
        has_more = loop_state.get("has_more_iterations", False)
        if has_more:
            return body_target if body_target else END
        return exit_target if exit_target else END

    return route_loop


# =============================================================================
# Node Type Implementation
# =============================================================================

class ForEachNodeType(BaseNodeType):
    """Implementation of the for_each (loop) node type."""

    @property
    def type_literal(self) -> str:
        return "for_each"

    @property
    def model_class(self) -> type["NodeBase"]:
        return ForEachNode

    async def execute_async(  # pylint: disable=too-many-locals  # Reason: Loop iteration requires many state variables
        self,
        node: ForEachNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Initialize or advance loop iteration state.

        On first call: Evaluate items and initialize loop state.
        On subsequent calls: Advance the index.
        On nested loop re-entry after parent iteration: Reset the loop state.

        Loop body execution is handled by LangGraph graph traversal.
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext, evaluate_value
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        loop_key = f"_loop_{node.id}"
        existing_loop_state = ctx.state.get(loop_key)

        # Build eval context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
            vars=ctx.vars,
        )

        # Check for nested loop reset (parent iteration changed)
        should_reset = False
        nested_loop_parents = ctx.nested_loop_parents or {}
        parent_loop_id = nested_loop_parents.get(node.id)
        parent_state = ctx.state.get(f"_loop_{parent_loop_id}") if parent_loop_id else None

        if existing_loop_state and parent_state:
            stored_parent_idx = existing_loop_state.get("_parent_iteration_idx")
            current_parent_idx = parent_state.get("current_index", 0)
            if stored_parent_idx != current_parent_idx:
                should_reset = True

        if existing_loop_state is None or should_reset:
            # First invocation OR nested loop reset
            items_value = evaluate_value(eval_ctx, node.items)
            if not isinstance(items_value, list):
                raise ExecutionError(f"for_each node '{node.id}' items expression must produce a list")

            new_run_id = (existing_loop_state.get("_run_id", -1) + 1) if existing_loop_state else 0

            loop_state = {
                "items": items_value,
                "current_index": 0,
                "has_more_iterations": len(items_value) > 0,
                "results": [],
                "_run_id": new_run_id,
                "_parent_iteration_idx": parent_state.get("current_index", 0) if parent_state else None,
            }
        else:
            # Subsequent invocation - advance to next iteration
            loop_state = dict(existing_loop_state)
            loop_state["current_index"] += 1
            loop_state["has_more_iterations"] = loop_state["current_index"] < len(loop_state["items"])

        # Build updates
        updates: Dict[str, Any] = {loop_key: loop_state}

        # Set current item and index for body nodes
        if loop_state["has_more_iterations"]:
            idx = loop_state["current_index"]
            updates[node.item_var] = loop_state["items"][idx]
            updates[node.index_var] = idx

        return updates

    def register_type_sync(
        self,
        node: ForEachNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """Register ForEachNode's output schema (aggregated results)."""
        if node.outputs:
            loop_schema = schema_from_output_contract(node.outputs, ctx.schema_registry)
        else:
            loop_schema = {"type": "array"}

        if node.id:
            env.register(node.id, loop_schema)

    def get_routing(
        self,
        node: ForEachNode,  # type: ignore[override]
        outgoing_edges: List["Edge"],
    ) -> Optional[RoutingResult]:
        """Build routing configuration for loop edges."""
        from langgraph.graph import END  # pylint: disable=import-outside-toplevel  # Reason: Late import to avoid startup overhead

        # Check if this node has loop edges
        has_loop_edges = any(
            e.type in (EdgeType.loop_body, EdgeType.loop_exit)
            for e in outgoing_edges
        )

        if not has_loop_edges:
            return None

        # Extract targets
        body_target: Optional[str] = None
        exit_target: Optional[str] = None

        for edge in outgoing_edges:
            if edge.type == EdgeType.loop_body:
                body_target = edge.target
            elif edge.type == EdgeType.loop_exit:
                exit_target = edge.target

        # Build router
        router = _build_loop_router(node.id, body_target, exit_target)

        # Build path map
        path_map: Dict[str, str] = {}
        if body_target:
            path_map[body_target] = body_target
        if exit_target:
            path_map[exit_target] = exit_target
        if END not in path_map.values():
            path_map[END] = END

        return RoutingResult(
            is_conditional=True,
            router_func=router,
            path_map=path_map,
        )


# Auto-register on module import
register_node_type(ForEachNodeType())
