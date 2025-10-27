from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ProgrammerState


def _initialize(state: ProgrammerState) -> ProgrammerState:
    return state


def _generate_action(state: ProgrammerState) -> ProgrammerState:
    # Pass-through; the router will decide based on remaining tasks
    return state


def _route_after_generate_action(state: ProgrammerState):
    plan = state.get("taskPlan") or {}
    items = list(plan.get("items", []))
    has_todo = any(i.get("status") != "done" for i in items)
    return "take-action" if has_todo else "generate-conclusion"


def _take_action(state: ProgrammerState) -> ProgrammerState:
    # Minimal MVP: mark first TODO item as done and append a message
    plan = dict(state.get("taskPlan") or {"title": "", "items": []})
    items = list(plan.get("items", []))
    for it in items:
        if it.get("status") != "done":
            it["status"] = "done"
            msg = f"Marked done: {it.get('description', '')}"
            messages = list(state.get("messages", []))
            messages.append({"role": "system", "content": msg})
            new_state = dict(state)
            plan["items"] = items
            new_state["taskPlan"] = plan
            new_state["messages"] = messages
            return new_state
    return state


def _generate_conclusion(state: ProgrammerState) -> ProgrammerState:
    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Programming phase complete."})
    state = dict(state)
    state["messages"] = messages
    return state


def compile_programmer_graph():
    workflow = StateGraph(ProgrammerState)
    workflow.add_node("initialize", _initialize)
    workflow.add_node("generate-action", _generate_action)
    workflow.add_node("take-action", _take_action)
    workflow.add_node("generate-conclusion", _generate_conclusion)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "generate-action")
    workflow.add_conditional_edges(
        "generate-action",
        _route_after_generate_action,
        path_map=["take-action", "generate-conclusion"],
    )
    workflow.add_edge("take-action", "generate-action")
    workflow.add_edge("generate-conclusion", END)

    return workflow.compile()

graph = compile_programmer_graph()