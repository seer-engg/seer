from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ReviewerState


def _initialize_state(state: ReviewerState) -> ReviewerState:
    return state


def _final_review(state: ReviewerState) -> ReviewerState:
    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Review complete (stub)."})
    state = dict(state)
    state["messages"] = messages
    return state


def compile_reviewer_graph():
    workflow = StateGraph(ReviewerState)
    workflow.add_node("initialize-state", _initialize_state)
    workflow.add_node("final-review", _final_review)

    workflow.add_edge(START, "initialize-state")
    workflow.add_edge("initialize-state", "final-review")
    workflow.add_edge("final-review", END)

    return workflow.compile()

graph = compile_reviewer_graph()