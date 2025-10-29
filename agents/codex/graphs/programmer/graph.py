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



def _route_after_generate_action(state: ProgrammerState):
    plan = state.taskPlan or {}
    items = list(plan.items)
    has_todo = any(i.status != "done" for i in items)
    return "execute-task-item" if has_todo else "reviewer-subgraph"




def _generate_conclusion(state: ProgrammerState) -> ProgrammerState:
    messages = list(state.messages)
    messages.append({"role": "system", "content": "Programming phase complete."})
    return {
        "messages": messages,
    }


def _run_reviewer(state: ProgrammerState) -> ProgrammerState:
    reviewer_input = {
        "request": state.request,
        "repo_path": state.repo_path,
        "messages": state.messages,
        "taskPlan": state.taskPlan,
    }
    reviewer_output = reviewer_graph.invoke(reviewer_input)
    new_state = dict(state)
    new_state.update({
        "messages": reviewer_output.get("messages", state.get("messages", [])),
        "taskPlan": reviewer_output.get("taskPlan", state.get("taskPlan")),
    })
    return new_state


def compile_programmer_graph():
    workflow = StateGraph(ProgrammerState)
    workflow.add_node("initialize", _initialize)
    workflow.add_node("implement-task-plan", implement_task_plan)
    workflow.add_node("generate-conclusion", _generate_conclusion)
    workflow.add_node("reviewer-subgraph", _run_reviewer)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "implement-task-plan")
    workflow.add_edge("implement-task-plan", "reviewer-subgraph")
    workflow.add_edge("reviewer-subgraph", "generate-conclusion")
    workflow.add_edge("generate-conclusion", END)

    return workflow.compile()

graph = compile_programmer_graph()