"""
IfNode - Conditional branching in workflows.

Evaluates a condition expression and routes execution to either the true or
false branch based on edges with type=conditional_true/conditional_false.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

from seer.core.nodes.base import BaseNodeType, NodeExecutionContext, RoutingResult, TypeRegistrationContext
from seer.core.nodes.registry import register_node_type
# Import model from schema/models.py (canonical location)
from seer.core.schema.models import EdgeType, IfNode

if TYPE_CHECKING:
    from seer.core.expr.typecheck import TypeEnvironment
    from seer.core.runtime.nodes import RuntimeServices
    from seer.core.schema.models import Edge, NodeBase


# =============================================================================
# Routing Functions
# =============================================================================

def _build_if_router(node_id: str, true_target: Optional[str], false_target: Optional[str]):
    """
    Build a routing function for IfNode conditional edges.

    The IfNode runner stores the condition result in state[f"_if_result_{node_id}"].
    This router reads that value and returns the appropriate target.
    """
    def route_if(state: dict) -> str:
        from langgraph.graph import END  # pylint: disable=import-outside-toplevel  # Reason: Late import for router closure
        condition_result = state.get(f"_if_result_{node_id}", False)
        if condition_result:
            return true_target if true_target else END
        return false_target if false_target else END

    return route_if


# =============================================================================
# Node Type Implementation
# =============================================================================

class IfNodeType(BaseNodeType):
    """Implementation of the if (conditional) node type."""

    @property
    def type_literal(self) -> str:
        return "if"

    @property
    def model_class(self) -> type["NodeBase"]:
        return IfNode

    async def execute_async(
        self,
        node: IfNode,  # type: ignore[override]
        ctx: NodeExecutionContext,
        services: "RuntimeServices",
    ) -> Dict[str, Any]:
        """
        Evaluate the condition and store the result in state.

        Branch selection is handled by LangGraph conditional edges via get_routing().
        The IfNode itself just evaluates the condition and stores it for the router.
        """
        # pylint: disable=import-outside-toplevel  # Reason: Avoid circular imports at module load time
        from seer.core.expr.evaluator import EvaluationContext, evaluate_condition
        from seer.core.runtime.state import INTERNAL_STATE_PREFIX

        # Build evaluation context
        visible_state = {k: v for k, v in ctx.state.items() if not k.startswith(INTERNAL_STATE_PREFIX)}
        eval_ctx = EvaluationContext(
            state=visible_state,
            locals=ctx.locals_ctx or {},
            config=ctx.config,
            trigger=ctx.trigger,
            vars=ctx.vars,
        )

        # Evaluate condition
        condition_result = evaluate_condition(eval_ctx, node.condition)

        # Store result for the router
        return {f"_if_result_{node.id}": condition_result}

    def register_type_sync(
        self,
        node: IfNode,  # type: ignore[override]
        env: "TypeEnvironment",
        ctx: TypeRegistrationContext,
    ) -> None:
        """
        IfNode doesn't produce output directly - branches do.

        The condition result is stored in internal state (prefixed with _)
        so it doesn't need type registration.
        """
        # IfNode doesn't register any output schema - intentionally empty

    def get_routing(
        self,
        node: IfNode,  # type: ignore[override]
        outgoing_edges: List["Edge"],
    ) -> Optional[RoutingResult]:
        """
        Build routing configuration for conditional edges.

        Returns routing function and path map for LangGraph conditional edges.
        """
        from langgraph.graph import END  # pylint: disable=import-outside-toplevel  # Reason: Late import to avoid startup overhead

        # Check if this node has conditional edges
        has_conditional = any(
            e.type in (EdgeType.conditional_true, EdgeType.conditional_false)
            for e in outgoing_edges
        )

        if not has_conditional:
            return None

        # Extract targets
        true_target: Optional[str] = None
        false_target: Optional[str] = None

        for edge in outgoing_edges:
            if edge.type == EdgeType.conditional_true:
                true_target = edge.target
            elif edge.type == EdgeType.conditional_false:
                false_target = edge.target

        # Build router
        router = _build_if_router(node.id, true_target, false_target)

        # Build path map
        path_map: Dict[str, str] = {}
        if true_target:
            path_map[true_target] = true_target
        if false_target:
            path_map[false_target] = false_target
        if END not in path_map.values():
            path_map[END] = END

        return RoutingResult(
            is_conditional=True,
            router_func=router,
            path_map=path_map,
        )


# Auto-register on module import
register_node_type(IfNodeType())
