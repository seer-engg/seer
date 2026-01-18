"""
Stage 5 — Emit a LangGraph StateGraph from the lowered execution plan.

V2 uses explicit edges with conditional routing for if/else and loop control flow.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Optional

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from seer.core.compiler.lower_control_flow import ExecutionPlan
from seer.core.runtime.nodes import NodeRuntime
from seer.core.schema.models import Edge, EdgeType, ForEachNode, IfNode, Node


def merge_state(left: dict, right: dict) -> dict:
    """Merge two state dictionaries, preserving all keys from both.

    This ensures trace data from all nodes is preserved by merging
    state updates instead of replacing them.
    """
    return {**left, **right}


# State schema with reducer to merge all state updates (including trace keys)
WorkflowState = Annotated[Dict[str, Any], merge_state]


def _build_if_router(node_id: str, true_target: Optional[str], false_target: Optional[str]):
    """
    Build a routing function for IfNode conditional edges.

    The IfNode runner stores the condition result in state[f"_if_result_{node_id}"].
    This router reads that value and returns the appropriate target.
    """
    def route_if(state: dict) -> str:
        condition_result = state.get(f"_if_result_{node_id}", False)
        if condition_result:
            return true_target if true_target else END
        return false_target if false_target else END

    return route_if


def _build_loop_router(node_id: str, body_target: Optional[str], exit_target: Optional[str]):
    """
    Build a routing function for ForEachNode conditional edges.

    The ForEachNode runner stores iteration state in state[f"_loop_{node_id}"].
    This router checks has_more_iterations and returns body or exit target.
    """
    def route_loop(state: dict) -> str:
        loop_state = state.get(f"_loop_{node_id}", {})
        has_more = loop_state.get("has_more_iterations", False)
        if has_more:
            return body_target if body_target else END
        return exit_target if exit_target else END

    return route_loop


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


def _add_conditional_edges_for_if(
    graph: StateGraph,
    node: IfNode,
    outgoing_edges: List[Edge],
) -> None:
    """
    Add conditional edges for an IfNode.

    Routes to true or false branch based on condition result stored in state.
    """
    true_target: Optional[str] = None
    false_target: Optional[str] = None

    for edge in outgoing_edges:
        if edge.type == EdgeType.conditional_true:
            true_target = edge.target
        elif edge.type == EdgeType.conditional_false:
            false_target = edge.target

    # Build the routing function
    router = _build_if_router(node.id, true_target, false_target)

    # Build path map for all possible destinations
    path_map: Dict[str, str] = {}
    if true_target:
        path_map[true_target] = true_target
    if false_target:
        path_map[false_target] = false_target
    if END not in path_map.values():
        path_map[END] = END

    graph.add_conditional_edges(node.id, router, path_map)


def _add_conditional_edges_for_loop(
    graph: StateGraph,
    node: ForEachNode,
    outgoing_edges: List[Edge],
) -> None:
    """
    Add conditional edges for a ForEachNode.

    Routes to body (more iterations) or exit (done) based on iteration state.
    """
    body_target: Optional[str] = None
    exit_target: Optional[str] = None

    for edge in outgoing_edges:
        if edge.type == EdgeType.loop_body:
            body_target = edge.target
        elif edge.type == EdgeType.loop_exit:
            exit_target = edge.target

    # Build the routing function
    router = _build_loop_router(node.id, body_target, exit_target)

    # Build path map for all possible destinations
    path_map: Dict[str, str] = {}
    if body_target:
        path_map[body_target] = body_target
    if exit_target:
        path_map[exit_target] = exit_target
    if END not in path_map.values():
        path_map[END] = END

    graph.add_conditional_edges(node.id, router, path_map)


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

    # Add all nodes to the graph
    node_map: Dict[str, Node] = {}
    for node in plan.nodes:
        graph.add_node(node.id, runtime.build_runner(node))
        node_map[node.id] = node

    # Connect START to entry point(s)
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

    # Process edges for each node
    for node in plan.nodes:
        outgoing = plan.outgoing_edges.get(node.id, [])

        if isinstance(node, IfNode):
            # Check if this node has conditional edges
            has_conditional = any(
                e.type in (EdgeType.conditional_true, EdgeType.conditional_false)
                for e in outgoing
            )
            if has_conditional:
                _add_conditional_edges_for_if(graph, node, outgoing)
            else:
                _add_regular_edges(graph, node, outgoing)

        elif isinstance(node, ForEachNode):
            # Check if this node has loop edges
            has_loop_edges = any(
                e.type in (EdgeType.loop_body, EdgeType.loop_exit)
                for e in outgoing
            )
            if has_loop_edges:
                _add_conditional_edges_for_loop(graph, node, outgoing)
            else:
                _add_regular_edges(graph, node, outgoing)

        else:
            _add_regular_edges(graph, node, outgoing)

    if checkpointer:
        return graph.compile(checkpointer=checkpointer)
    return graph.compile()
