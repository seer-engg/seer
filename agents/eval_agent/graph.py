from typing import Literal
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.deps import logger
from agents.eval_agent.nodes.finalize import build_finalize_subgraph, should_continue
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph


def finalize_router(state: EvalAgentState) -> Literal["plan", END]:
    if state.pending_followup:
        return "plan"
    else:
        return END

def build_graph():
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
    workflow.add_conditional_edges("run", should_continue, {"reflect": "reflect", "finalize": "finalize"})
    workflow.add_edge("reflect", "plan")
    workflow.add_conditional_edges("finalize", finalize_router)

    graph = workflow.compile(debug=True)
    logger.info("Eval Agent graph compiled successfully")
    return graph


graph = build_graph()
