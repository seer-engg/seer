"""
Stage 5 — Emit a LangGraph StateGraph from the lowered execution plan.

V2 uses explicit edges with conditional routing for if/else and loop control flow.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from seer.core.compiler.lower_control_flow import ExecutionPlan
from seer.core.nodes.registry import node_type_registry
from seer.core.runtime.nodes import NodeRuntime
from seer.core.schema.models import Edge, EdgeType, ForEachNode, Node


def merge_state(left: dict, right: dict) -> dict:
    """Merge two state dictionaries, preserving all keys from both.

    This ensures trace data from all nodes is preserved by merging
    state updates instead of replacing them.
    """
    return {**left, **right}


# State schema with reducer to merge all state updates (including trace keys)
# NOTE: Use builtin `dict` instead of `typing.Dict` because LangGraph + Sentry SDK
# needs to instantiate the state type for graph visualization, and typing.Dict
# raises "Type Dict cannot be instantiated; use dict() instead"
WorkflowState = Annotated[dict[str, Any], merge_state]


def _build_trigger_router(trigger_targets: Dict[str, str]):
    """
    Build a routing function for trigger-based entry point routing.

    The __trigger_bootstrap node stores the trigger_id in state["_trigger_id"].
    This router reads that value and returns the appropriate target node.
    """
    def route_by_trigger(state: dict) -> str:
        trigger_id = state.get("_trigger_id")
        if trigger_id and trigger_id in trigger_targets:
            return trigger_targets[trigger_id]
        # Fallback: use first trigger's target if available
        if trigger_targets:
            return next(iter(trigger_targets.values()))
        return END

    return route_by_trigger


def _add_regular_edges(
    graph: StateGraph,
    node: Node,
    outgoing_edges: List[Edge],
) -> None:
    """
    Add regular (non-conditional) edges for a node.

    If no outgoing edges, connects to END.
    """
    if not outgoing_edges:
        graph.add_edge(node.id, END)
        return

    for edge in outgoing_edges:
        graph.add_edge(node.id, edge.target)


def _is_terminal_of_nested_loop(
    terminal_node_id: str,
    loop_id: str,
    plan: ExecutionPlan,
) -> bool:
    """
    Check if a terminal node belongs to a nested inner loop of the given loop.

    This is used to avoid adding duplicate back-edges when a node is terminal
    for both an inner and outer loop.
    """
    for inner_loop_id, outer_loop_id in plan.nested_loop_parents.items():
        if outer_loop_id == loop_id:
            inner_terminal_nodes = plan.loop_terminal_nodes.get(inner_loop_id, set())
            if terminal_node_id in inner_terminal_nodes:
                return True
    return False


# pylint: disable=unused-argument,protected-access
# Reason: 'state' required by LangGraph interface; protected access for internal trigger state
def _connect_entry_points(
    graph: StateGraph,
    plan: ExecutionPlan,
    runtime: NodeRuntime,
) -> None:
    """
    Connect START to workflow entry point(s).

    Handles three cases:
    - Trigger-based routing: bootstrap node + conditional edges
    - Single entry point: direct edge from START
    - Fallback: use first node if no explicit entry
    """
    if plan.trigger_targets:
        # Trigger-based routing: add bootstrap node and conditional edges
        # Note: Using a closure that captures runtime to access trigger context
        def make_trigger_bootstrap(rt: NodeRuntime):
            def trigger_bootstrap(state: dict) -> dict:
                """Extract trigger_id from runtime trigger envelope into state."""
                trigger = rt._current_trigger
                if trigger:
                    return {"_trigger_id": trigger.get("trigger_id")}
                return {}
            return trigger_bootstrap

        graph.add_node("__trigger_bootstrap", make_trigger_bootstrap(runtime))
        graph.add_edge(START, "__trigger_bootstrap")

        # Build router and path map for conditional edges
        router = _build_trigger_router(plan.trigger_targets)
        path_map: Dict[str, str] = {target: target for target in set(plan.trigger_targets.values())}
        path_map[END] = END

        graph.add_conditional_edges("__trigger_bootstrap", router, path_map)
    elif plan.entry_node_id:
        # Single entry point: direct edge from START
        graph.add_edge(START, plan.entry_node_id)
    elif plan.nodes:
        # Fallback: use first node if no explicit entry
        graph.add_edge(START, plan.nodes[0].id)


def _add_implicit_loop_back_edges(
    graph: StateGraph,
    plan: ExecutionPlan,
) -> None:
    """
    Add implicit back-edges from loop terminal nodes to their ForEachNode.

    BUG FIX: Nested Loop Terminal Node Handling (2024-02 RCA)
    ---------------------------------------------------------
    PROBLEM: When detecting terminal nodes for outer loops, nodes inside
      nested inner loops were incorrectly marked as terminals of BOTH loops.
      This caused duplicate back-edges to be added:
        - process -> inner_loop (correct)
        - process -> outer_loop (WRONG - causes routing chaos)

    EXAMPLE: For outer_loop -> inner_loop -> process:
      - process is terminal of inner_loop (correct)
      - process is ALSO detected as terminal of outer_loop (incorrect)
      - Without this fix, LangGraph would add edges to both loops

    SOLUTION: Before adding an implicit back-edge from a terminal node to
      an outer loop, check if the terminal is ALSO a terminal of any nested
      inner loop. If so, skip adding the edge to the outer loop - the inner
      loop's back-edge will handle returning control correctly.
    """
    for node in plan.nodes:
        if not isinstance(node, ForEachNode):
            continue

        terminal_nodes = plan.loop_terminal_nodes.get(node.id, set())
        for terminal_node_id in terminal_nodes:
            # Skip if this terminal node belongs to a nested inner loop
            if _is_terminal_of_nested_loop(terminal_node_id, node.id, plan):
                continue

            # Check if explicit edge already exists
            existing_edges = plan.outgoing_edges.get(terminal_node_id, [])
            has_explicit_loop_back = any(
                e.target == node.id and e.type == EdgeType.default
                for e in existing_edges
            )

            if not has_explicit_loop_back:
                # Add implicit edge from terminal node back to loop
                graph.add_edge(terminal_node_id, node.id)


# pylint: disable=unused-argument,protected-access
# Reason: Unused params required by LangGraph interface; protected access for internal state
async def emit_langgraph(
    plan: ExecutionPlan,
    runtime: NodeRuntime,
    *,
    checkpointer: Optional[AsyncPostgresSaver] = None,
):
    """
    Emit a LangGraph StateGraph from the execution plan.

    Handles:
    - IfNode: Conditional edges based on condition result
    - ForEachNode: Conditional edges for loop body vs exit
    - Other nodes: Direct edges from the edge list
    """
    graph = StateGraph(WorkflowState)

    if not plan.nodes:
        graph.add_node("__noop", lambda state, config: {})
        graph.add_edge(START, "__noop")
        graph.add_edge("__noop", END)
        return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()

    # Build loop body map: node_id -> parent_loop_id
    loop_body_map: Dict[str, str] = {}
    for loop_id, body_nodes in plan.loop_body_nodes.items():
        for node_id in body_nodes:
            loop_body_map[node_id] = loop_id

    # Set loop body map in runtime for trace key generation
    runtime.set_loop_body_map(loop_body_map)

    # Set nested loop parents in runtime for state isolation
    runtime.set_nested_loop_parents(plan.nested_loop_parents)

    # Add all nodes to the graph
    node_map: Dict[str, Node] = {}
    for node in plan.nodes:
        graph.add_node(node.id, runtime.build_runner(node))
        node_map[node.id] = node

    # Connect START to entry point(s)
    _connect_entry_points(graph, plan, runtime)

    # Process edges for each node using registry-based routing
    for node in plan.nodes:
        outgoing = plan.outgoing_edges.get(node.id, [])

        # Get routing from node type registry
        node_impl = node_type_registry.get(node.type)
        if node_impl:
            routing = node_impl.get_routing(node, outgoing)
            if routing and routing.is_conditional:
                graph.add_conditional_edges(node.id, routing.router_func, routing.path_map)
                continue

        # Default: regular edges for non-routing nodes (or nodes not needing conditional edges)
        _add_regular_edges(graph, node, outgoing)

    # Add implicit back-edges for loop terminal nodes
    _add_implicit_loop_back_edges(graph, plan)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
