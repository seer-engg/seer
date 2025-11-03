from typing import Literal
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.constants import ATTEMPTS, PASS_THRESHOLD
from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState, RunContext
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph


def finalize_router(state: EvalAgentState) -> Literal["plan", END]:
    """Determine if the eval loop should continue planning or finalize."""
    if state.pending_followup:
        return "plan"
    else:
        return END


def should_continue(state: EvalAgentState) -> Literal["reflect", "finalize"]:
    """Determine if the eval loop should continue reflecting or finalize."""

    run_ctx = state.run or RunContext()

    if run_ctx.attempts < ATTEMPTS:
        return "reflect"
    if run_ctx.score >= PASS_THRESHOLD or run_ctx.attempts >= ATTEMPTS:
        return "finalize"
    return "reflect"


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
    workflow.add_conditional_edges("run", should_continue)
    workflow.add_edge("reflect", "plan")
    workflow.add_conditional_edges("finalize", finalize_router)

    return workflow.compile(debug=True)


graph = build_graph()
