from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import PlannerState

from shared.logger import get_logger

from agents.codex.graphs.planner.nodes import initialize_project
from agents.codex.graphs.planner.nodes import context_and_plan_agent

logger = get_logger("codex.planner")


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state



def _notetaker(state: PlannerState) -> PlannerState:
    # Minimal note taking: append a message summarizing the plan was generated
    messages = list(state.messages)
    messages.append({
        "role": "system",
        "content": "Generated initial task plan.",
    })
    state = dict(state)
    state["messages"] = messages
    return state


def _interrupt_proposed_plan(state: PlannerState) -> PlannerState:
    # In MVP, auto-accept the plan
    return state


def compile_planner_graph():
    workflow = StateGraph(PlannerState)
    workflow.add_node("prepare-graph-state", _prepare_graph_state)
    workflow.add_node("initialize-project", initialize_project)
    # Combined context + planning agent
    workflow.add_node("context-plan-agent", context_and_plan_agent)
    # Removed separate context agent; using combined agent instead
    workflow.add_node("notetaker", _notetaker)
    workflow.add_node("interrupt-proposed-plan", _interrupt_proposed_plan)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-project")
    workflow.add_edge("initialize-project", "context-plan-agent")
    workflow.add_edge("context-plan-agent", "notetaker")
    workflow.add_edge("notetaker", "interrupt-proposed-plan")
    workflow.add_edge("interrupt-proposed-plan", END)

    return workflow.compile()

graph = compile_planner_graph()