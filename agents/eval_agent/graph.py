"""overall graph for the eval agent"""
from typing import Literal
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.constants import N_ROUNDS, N_VERSIONS
from agents.eval_agent.nodes.finalize import build_finalize_subgraph
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.plan import build_plan_subgraph
from agents.eval_agent.nodes.reflect.graph import reflect_node
from agents.eval_agent.nodes.run import build_run_subgraph
from shared.logger import get_logger

logger = get_logger("eval_agent.graph")


def should_continue(state: EvalAgentState) -> Literal["plan", "finalize"]:
    """Determine if the eval loop should continue plan or finalize."""
    return "plan" if state.attempts < N_ROUNDS else "finalize"


def after_finalize(state: EvalAgentState) -> Literal["plan", "__end__"]:
    """If agent was updated and we haven't hit the version limit, start new round."""
    if state.agent_updated and state.target_agent_version < N_VERSIONS:
        logger.info(f"Agent updated to v{state.target_agent_version}. Starting new evaluation round.")
        return "plan"
    else:
        if not state.agent_updated:
            logger.info("Agent was not updated by Codex. Ending workflow.")
        else:
            logger.info(f"Reached max versions ({N_VERSIONS}). Ending workflow.")
        return "__end__"


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
    workflow.add_conditional_edges("finalize", after_finalize, {
        "plan": "plan",
        "__end__": END
    })

    return workflow.compile(debug=True)


graph = build_graph()
