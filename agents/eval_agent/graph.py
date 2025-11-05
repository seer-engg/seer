"""overall graph for the eval agent"""
from typing import Literal
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.constants import N_ROUNDS
from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph


def should_continue(state: EvalAgentState) -> Literal["plan", "finalize"]:
    """Determine if the eval loop should continue plan or finalize."""
    return "plan" if state.attempts < N_ROUNDS else "finalize"

def build_graph():
    """Build the evaluation agent graph."""
    workflow = StateGraph(EvalAgentState)
    plan_subgraph = build_plan_subgraph()
    run_subgraph = build_run_subgraph()
    finalize_subgraph = build_finalize_subgraph()

    workflow.add_node("plan", plan_subgraph)
    workflow.add_node("run", run_subgraph)
    workflow.add_node("reflect", reflect_node)
    workflow.add_node("finalize", finalize_subgraph)

    workflow.add_edge(START, "plan")
    workflow.add_edge("plan", "run")
    workflow.add_edge("run", "reflect")
    workflow.add_conditional_edges("reflect", should_continue, {
        "plan": "plan",
        "finalize": "finalize"
    })
    workflow.add_edge("finalize", END)

    return workflow.compile(debug=True)


graph = build_graph()
