from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import PlannerState, TaskItem, TaskPlan
from agents.codex.llm.model import get_chat_model, generate_plan_steps
from agents.codex.graphs.shared.initialize_sandbox import ensure_sandbox_ready


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state


def _initialize_sandbox(state: PlannerState) -> PlannerState:
    ensure_sandbox_ready(state.get("repo_path", ""))
    return state


def _generate_plan(state: PlannerState) -> PlannerState:
    model = get_chat_model()
    request = state.get("request", "")
    steps = generate_plan_steps(model, request, None)
    plan_items: list[TaskItem] = [
        {"description": s, "status": "todo"} for s in steps
    ]
    state = dict(state)
    state["taskPlan"] = TaskPlan(title=f"Plan: {request[:50]}", items=plan_items)
    return state


def _notetaker(state: PlannerState) -> PlannerState:
    # Minimal note taking: append a message summarizing the plan was generated
    messages = list(state.get("messages", []))
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
    workflow.add_node("initialize-sandbox", _initialize_sandbox)
    workflow.add_node("generate-plan", _generate_plan)
    workflow.add_node("notetaker", _notetaker)
    workflow.add_node("interrupt-proposed-plan", _interrupt_proposed_plan)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-sandbox")
    workflow.add_edge("initialize-sandbox", "generate-plan")
    workflow.add_edge("generate-plan", "notetaker")
    workflow.add_edge("notetaker", "interrupt-proposed-plan")
    workflow.add_edge("interrupt-proposed-plan", END)

    return workflow.compile()

graph = compile_planner_graph()