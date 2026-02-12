"""
Stage 5 — Emit a LangGraph StateGraph from the lowered execution plan.

Uses explicit edges with conditional routing for if/else and loop control flow.
Includes join nodes for parallel branch convergence (see _detect_convergence_points).
"""

from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, List, Optional, Set

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


def _build_trigger_router(trigger_targets: Dict[str, List[str]]):
    """Build a routing function that reads state["_trigger_id"] and returns target node(s)."""
    def route_by_trigger(state: dict) -> str | List[str]:
        trigger_id = state.get("_trigger_id")
        if trigger_id and trigger_id in trigger_targets:
            targets = trigger_targets[trigger_id]
            # Return single target as string, multiple as list for parallel execution
            if len(targets) == 1:
                return targets[0]
            return targets
        # Fallback: use first trigger's targets if available
        if trigger_targets:
            targets = next(iter(trigger_targets.values()))
            if len(targets) == 1:
                return targets[0]
            return targets
        return END

    return route_by_trigger


def _find_trigger_sources(
    node_id: str,
    plan: ExecutionPlan,
    visited: Set[str] | None = None,
) -> Set[str]:
    """Trace back from a node to find all trigger sources that can reach it."""
    if visited is None:
        visited = set()

    if node_id in visited:
        return set()
    visited.add(node_id)

    trigger_sources: Set[str] = set()

    # Check incoming edges to this node
    incoming = plan.incoming_edges.get(node_id, [])
    for edge in incoming:
        if edge.type == EdgeType.trigger:
            # This node is directly connected to a trigger
            trigger_sources.add(edge.source)
        else:
            # Recurse to find triggers upstream
            upstream_triggers = _find_trigger_sources(edge.source, plan, visited)
            trigger_sources.update(upstream_triggers)

    return trigger_sources


def _paths_share_conditional_ancestor(predecessors: List[str], plan: ExecutionPlan) -> bool:
    """Check if predecessors diverged from the same if-node (mutually exclusive branches)."""
    if len(predecessors) != 2:
        # For now, only handle the common case of 2 predecessors
        # More complex cases would need more sophisticated analysis
        return False

    pred_a, pred_b = predecessors

    # Check incoming edges to each predecessor
    incoming_a = plan.incoming_edges.get(pred_a, [])
    incoming_b = plan.incoming_edges.get(pred_b, [])

    # Look for conditional edges
    for edge_a in incoming_a:
        if edge_a.type not in (EdgeType.conditional_true, EdgeType.conditional_false):
            continue

        for edge_b in incoming_b:
            if edge_b.type not in (EdgeType.conditional_true, EdgeType.conditional_false):
                continue

            # Check if they come from the same if-node on opposite branches
            if edge_a.source == edge_b.source and edge_a.type != edge_b.type:
                return True

    return False


def _predecessors_in_same_loop_body(
    predecessors: List[str], loop_body_nodes: Dict[str, Set[str]]
) -> bool:
    """Check if all predecessors belong to the same loop body (sequential, not parallel)."""
    for body_nodes in loop_body_nodes.values():
        if all(pred in body_nodes for pred in predecessors):
            return True
    return False


def _find_common_triggers(predecessors: List[str], plan: ExecutionPlan) -> Set[str]:
    """Find trigger sources common to all predecessors (intersection of reachable triggers)."""
    predecessor_triggers: List[Set[str]] = []
    for pred in predecessors:
        triggers = _find_trigger_sources(pred, plan)
        predecessor_triggers.append(triggers)

    if not predecessor_triggers:
        return set()

    common_triggers = predecessor_triggers[0]
    for triggers in predecessor_triggers[1:]:
        common_triggers = common_triggers.intersection(triggers)

    return common_triggers


def _detect_convergence_points(
    plan: ExecutionPlan, loop_body_nodes: Dict[str, Set[str]]
) -> Dict[str, List[str]]:
    """
    Detect nodes with multiple incoming parallel edges requiring synchronization.

    Returns convergence points only when predecessors share a common trigger origin
    and are not from mutually exclusive conditional branches.
    """
    convergence_points: Dict[str, List[str]] = {}
    parallel_edge_types = {EdgeType.default}

    for node_id, incoming_edges in plan.incoming_edges.items():
        parallel_edges = [e for e in incoming_edges if e.type in parallel_edge_types]

        if len(parallel_edges) < 2:
            continue

        predecessors = list(set(e.source for e in parallel_edges))
        if len(predecessors) < 2:
            continue

        # Skip sequential loop iterations
        if _predecessors_in_same_loop_body(predecessors, loop_body_nodes):
            continue

        # Skip mutually exclusive if branches
        if _paths_share_conditional_ancestor(predecessors, plan):
            continue

        # Only sync if predecessors share a common trigger (parallel branches)
        common_triggers = _find_common_triggers(predecessors, plan)
        if common_triggers:
            convergence_points[node_id] = predecessors

    return convergence_points


def _create_join_node_func() -> Callable[[dict], dict]:
    """
    Create the function for a join node.

    The join node is a no-op that simply passes through.
    The actual synchronization logic is in the conditional router.
    """
    def join_func(state: dict) -> dict:  # pylint: disable=unused-argument
        # No-op: the join node doesn't modify state.
        # Synchronization happens via conditional routing.
        return {}

    return join_func


def _build_join_router(
    predecessors: List[str],
    convergent_node_id: str,
) -> Callable[[dict], str]:
    """
    Build a router that checks if all predecessors have completed.

    The router examines the state to see if all required predecessor outputs
    are present. If all are ready, it routes to the convergent node.
    If some are missing, it routes to END (this path is done, but another
    path will complete the join when it finishes).

    Args:
        predecessors: List of node IDs that must all be in state
        convergent_node_id: The target node to route to when all are ready

    Returns:
        Router function for conditional edges
    """
    def router(state: dict) -> str:
        # Check if all predecessors have their outputs in state
        for pred in predecessors:
            if pred not in state:
                # Not all predecessors ready yet.
                # Route to END for this execution path.
                # The state will be merged, and when the last predecessor
                # completes, all outputs will be available.
                return END

        # All predecessors are ready, proceed to the convergent node
        return convergent_node_id

    return router


def _add_join_nodes(
    graph: StateGraph,
    convergence_points: Dict[str, List[str]],
) -> Dict[str, str]:
    """
    Add join nodes to the graph for each convergence point.

    Creates a __join_{node_id} node for each convergent node that needs
    synchronization of parallel branches.

    Args:
        graph: The StateGraph being built
        convergence_points: Map of convergent node_id to predecessor list

    Returns:
        Dict mapping convergent node_id to its join node_id
    """
    join_node_map: Dict[str, str] = {}

    for convergent_node_id, predecessors in convergence_points.items():
        join_node_id = f"__join_{convergent_node_id}"

        # Add the join node (no-op function)
        graph.add_node(join_node_id, _create_join_node_func())

        # Add conditional edges from join node
        router = _build_join_router(predecessors, convergent_node_id)
        path_map = {
            convergent_node_id: convergent_node_id,
            END: END,
        }
        graph.add_conditional_edges(join_node_id, router, path_map)

        join_node_map[convergent_node_id] = join_node_id

    return join_node_map


def _add_regular_edges(
    graph: StateGraph,
    node: Node,
    outgoing_edges: List[Edge],
    join_node_map: Dict[str, str],
) -> None:
    """
    Add regular (non-conditional) edges for a node.

    If the target is a convergence point, routes through the join node instead.
    If no outgoing edges, connects to END.

    Args:
        graph: The StateGraph being built
        node: The source node
        outgoing_edges: List of edges from this node
        join_node_map: Map of convergent node_id to join node_id
    """
    if not outgoing_edges:
        graph.add_edge(node.id, END)
        return

    for edge in outgoing_edges:
        target = edge.target

        # If target is a convergence point, route through its join node
        if target in join_node_map:
            target = join_node_map[target]

        graph.add_edge(node.id, target)


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
        # Flatten all targets from all triggers (supports multiple parallel targets)
        router = _build_trigger_router(plan.trigger_targets)
        all_targets: set[str] = set()
        for targets in plan.trigger_targets.values():
            all_targets.update(targets)
        path_map: Dict[str, str] = {target: target for target in all_targets}
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


def _build_loop_body_map(plan: ExecutionPlan) -> Dict[str, str]:
    """
    Build a map from node_id to parent loop_id.

    This mapping is used by the runtime for trace key generation
    to track which loop iteration a node belongs to.

    Args:
        plan: The execution plan with loop body information

    Returns:
        Dict mapping node_id to the loop_id it belongs to
    """
    loop_body_map: Dict[str, str] = {}
    for loop_id, body_nodes in plan.loop_body_nodes.items():
        for node_id in body_nodes:
            loop_body_map[node_id] = loop_id
    return loop_body_map


def _process_node_edges(
    graph: StateGraph,
    node: Node,
    outgoing: List[Edge],
    join_node_map: Dict[str, str],
) -> None:
    """
    Process edges for a single node, handling routing and regular edges.

    For nodes with conditional routing (IfNode, ForEachNode), adds conditional
    edges with the appropriate router function. For other nodes, adds regular edges.

    Args:
        graph: The StateGraph being built
        node: The node to process edges for
        outgoing: List of outgoing edges from this node
        join_node_map: Map of convergent node_id to join node_id
    """
    # Get routing from node type registry
    node_impl = node_type_registry.get(node.type)
    if node_impl:
        routing = node_impl.get_routing(node, outgoing)
        if routing and routing.is_conditional:
            # Redirect convergence point targets through join nodes
            modified_path_map = {
                key: join_node_map.get(target, target)
                for key, target in routing.path_map.items()
            }
            graph.add_conditional_edges(node.id, routing.router_func, modified_path_map)
            return

    # Default: regular edges for non-routing nodes
    _add_regular_edges(graph, node, outgoing, join_node_map)


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
    - Parallel convergence: Join nodes for synchronization
    - Other nodes: Direct edges from the edge list
    """
    graph = StateGraph(WorkflowState)

    if not plan.nodes:
        graph.add_node("__noop", lambda state, config: {})
        graph.add_edge(START, "__noop")
        graph.add_edge("__noop", END)
        return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()

    # Build and set loop body map for trace key generation
    loop_body_map = _build_loop_body_map(plan)
    runtime.set_loop_body_map(loop_body_map)
    runtime.set_nested_loop_parents(plan.nested_loop_parents)

    # Detect convergence points and add join nodes for synchronization
    convergence_points = _detect_convergence_points(plan, plan.loop_body_nodes)
    join_node_map = _add_join_nodes(graph, convergence_points)

    # Add all workflow nodes to the graph
    for node in plan.nodes:
        graph.add_node(node.id, runtime.build_runner(node))

    # Connect START to entry point(s)
    _connect_entry_points(graph, plan, runtime)

    # Process edges for each node
    for node in plan.nodes:
        outgoing = plan.outgoing_edges.get(node.id, [])
        _process_node_edges(graph, node, outgoing, join_node_map)

    # Add implicit back-edges for loop terminal nodes
    _add_implicit_loop_back_edges(graph, plan)

    return graph.compile(checkpointer=checkpointer) if checkpointer else graph.compile()
