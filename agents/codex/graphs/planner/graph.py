from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import PlannerState,PlannerIOState

from shared.logger import get_logger

from agents.codex.graphs.planner.nodes import initialize_project
from agents.codex.graphs.planner.nodes import context_and_plan_agent
from agents.codex.graphs.planner.nodes.raise_pr import raise_pr

from agents.codex.graphs.programmer.graph import graph as programmer_graph

logger = get_logger("codex.planner")


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state



def _interrupt_proposed_plan(state: PlannerState) -> PlannerState:
    # In MVP, auto-accept the plan
    return state


async def _run_programmer(state: PlannerState) -> PlannerState:
    programmer_input = {
        "request": state.request,
        "repo_url": state.repo_url,
        "repo_path": state.repo_path,
        "taskPlan": state.taskPlan,
        "sandbox_session_id": state.sandbox_session_id,
    }
    programmer_output = await programmer_graph.ainvoke(programmer_input)
    return state

async def _run_reviewer(state: PlannerState) -> PlannerState:
    # TODO: add reviewer graph
    # reviewer_input = {
    #     "request": state.request,
    #     "repo_path": state.repo_path,
    #     "messages": state.messages,
    #     "taskPlan": state.taskPlan,
    # }
    # reviewer_output = reviewer_graph.invoke(reviewer_input)
    return state

#TODO: add reflexion mechanism here

def compile_planner_graph():
    workflow = StateGraph(state_schema=PlannerState, input=PlannerIOState, output=PlannerIOState)
    workflow.add_node("prepare-graph-state", _prepare_graph_state)
    workflow.add_node("initialize-project", initialize_project)
    # Combined context + planning agent
    workflow.add_node("context-plan-agent", context_and_plan_agent)
    # Removed separate context agent; using combined agent instead
    workflow.add_node("interrupt-proposed-plan", _interrupt_proposed_plan)
    workflow.add_node("programmer", _run_programmer)
    workflow.add_node("reviewer", _run_reviewer)
    workflow.add_node("raise-pr", raise_pr)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-project")
    workflow.add_edge("initialize-project", "context-plan-agent")
    workflow.add_edge("context-plan-agent", "interrupt-proposed-plan")
    workflow.add_edge("interrupt-proposed-plan", "programmer")
    workflow.add_edge("programmer", "reviewer")
    workflow.add_edge("reviewer", "raise-pr")
    workflow.add_edge("raise-pr", END)

    return workflow.compile()

graph = compile_planner_graph()