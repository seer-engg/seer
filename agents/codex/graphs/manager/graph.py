from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ManagerState, PlannerState
from agents.codex.graphs.planner.graph import graph as planner_graph
from agents.codex.graphs.programmer.graph import graph as programmer_graph


def _classify_message(state: ManagerState) -> ManagerState:
    # Classification happens in the router; this node passes state through.
    return state


def _route_after_classify(state: ManagerState) -> Literal["start-planner", "create-new-session"]:
    # Minimal routing: if no plan, create session; else start planner
    if not state.get("taskPlan"):
        return "create-new-session"
    return "start-planner"


def _create_new_session(state: ManagerState) -> ManagerState:
    # Initialize an empty plan session; actual plan will be created by Planner
    state = dict(state)
    state.setdefault("messages", [])
    return state


def _start_planner(state: ManagerState) -> ManagerState:
    # Call embedded Planner graph and merge results
    planner_input: PlannerState = {
        "request": state.get("request", ""),
        "repo_path": state.get("repo_path", ""),
        "messages": state.get("messages", []),
        "taskPlan": state.get("taskPlan"),
        "repo_url": state.get("repo_url", ""),
        "branch_name": state.get("branch_name", ""),
        "sandbox_session_id": state.get("sandbox_session_id", ""),
        "autoAcceptPlan": bool(state.get("autoAcceptPlan", True)),
    }
    planner_output: PlannerState = planner_graph.invoke(planner_input)
    state = dict(state)
    state.update({
        "messages": planner_output.get("messages", state.get("messages", [])),
        "taskPlan": planner_output.get("taskPlan", state.get("taskPlan")),
    })
    return state


def _start_programmer(state: ManagerState) -> ManagerState:
    # Invoke programmer graph to iterate through plan items (MVP marks todos done)
    programmer_input = {
        "request": state.get("request", ""),
        "repo_path": state.get("repo_path", ""),
        "messages": state.get("messages", []),
        "taskPlan": state.get("taskPlan"),
    }
    programmer_output = programmer_graph.invoke(programmer_input)
    new_state = dict(state)
    new_state.update({
        "messages": programmer_output.get("messages", state.get("messages", [])),
        "taskPlan": programmer_output.get("taskPlan", state.get("taskPlan")),
    })
    return new_state


def compile_manager_graph():
    workflow = StateGraph(ManagerState)
    workflow.add_node("classify-message", _classify_message)
    workflow.add_node("create-new-session", _create_new_session)
    workflow.add_node("start-planner", _start_planner)
    workflow.add_node("start-programmer", _start_programmer)

    workflow.add_edge(START, "classify-message")

    workflow.add_conditional_edges(
        "classify-message",
        _route_after_classify,
        path_map=["start-planner", "create-new-session"],
    )

    # After creating a new session, proceed to planning
    workflow.add_edge("create-new-session", "start-planner")

    # After planning, run programmer loop, then end (MVP)
    workflow.add_edge("start-planner", "start-programmer")
    workflow.add_edge("start-programmer", END)

    return workflow.compile()

graph = compile_manager_graph()