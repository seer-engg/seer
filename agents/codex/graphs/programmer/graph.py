from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ProgrammerState, TaskPlan, TaskItem
from agents.codex.graphs.reviewer.graph import graph as reviewer_graph
import time
from e2b_code_interpreter import AsyncSandbox
from shared.logger import get_logger
from agents.codex.graphs.programmer.nodes import implement_task_plan

logger = get_logger("codex.programmer")

async def _initialize(state: ProgrammerState) -> ProgrammerState:
    sandbox_id = state.sandbox_session_id
    if not sandbox_id:
        return state
    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_id)
    logger.info("Sandbox ready")
    return state

## to do add liniting check and resolve

def compile_programmer_graph():
    workflow = StateGraph(ProgrammerState)
    workflow.add_node("initialize", _initialize)
    workflow.add_node("implement-task-plan", implement_task_plan)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "implement-task-plan")
    workflow.add_edge("implement-task-plan", END)

    return workflow.compile()

graph = compile_programmer_graph()