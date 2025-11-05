from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ProgrammerState
from shared.logger import get_logger
from agents.codex.graphs.programmer.nodes import (
    implement_task_plan,
    test_implementation,
    reflect,
    initialize,
    finalize,
)
from agents.codex.graphs.planner.nodes.test_server_ready import test_server_ready

logger = get_logger("codex.programmer")


def reflection_router(state: ProgrammerState) -> ProgrammerState:
    if state.success or state.attempt_number >= state.max_attempts:
        return "finalize"
    return "reflect"

def is_server_ready(state: ProgrammerState) -> ProgrammerState:
    if state.server_running:
        return "test-implementation"
    else:
        return "implement-task-plan"

## to do add liniting check and resolve

def compile_programmer_graph():
    workflow = StateGraph(ProgrammerState)
    workflow.add_node("initialize", initialize)
    workflow.add_node("implement-task-plan", implement_task_plan)
    workflow.add_node("test-implementation", test_implementation)
    workflow.add_node("reflect", reflect)
    workflow.add_node("finalize", finalize)
    workflow.add_node("test-server-ready", test_server_ready)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "implement-task-plan")
    workflow.add_edge("implement-task-plan", "test-server-ready")

    workflow.add_conditional_edges("test-server-ready", is_server_ready, {
        "test-implementation": "test-implementation",
        "implement-task-plan": "implement-task-plan"
    })

    workflow.add_conditional_edges("test-implementation", reflection_router
    , {
        "finalize": "finalize",
        "reflect": "reflect"
    })
    workflow.add_edge("reflect", "implement-task-plan")

    workflow.add_edge("finalize", END)

    return workflow.compile()

graph = compile_programmer_graph()