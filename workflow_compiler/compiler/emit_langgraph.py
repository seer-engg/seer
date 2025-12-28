"""
Stage 5 — Emit a LangGraph StateGraph from the lowered execution plan.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from workflow_compiler.compiler.lower_control_flow import ExecutionPlan
from workflow_compiler.runtime.nodes import NodeRuntime


def emit_langgraph(plan: ExecutionPlan, runtime: NodeRuntime):
    graph = StateGraph(dict)

    if not plan.nodes:
        graph.add_node("__noop", lambda state, config: {})
        graph.add_edge(START, "__noop")
        graph.add_edge("__noop", END)
        return graph.compile()

    for node in plan.nodes:
        graph.add_node(node.id, runtime.build_runner(node))

    graph.add_edge(START, plan.nodes[0].id)
    for prev, curr in zip(plan.nodes, plan.nodes[1:]):
        graph.add_edge(prev.id, curr.id)
    graph.add_edge(plan.nodes[-1].id, END)

    return graph.compile()


