from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from agents.codex.common.state import ProgrammerState, TaskPlan, TaskItem
from langchain.agents import create_agent
from sandbox.tools import run_command_in_sandbox
from agents.codex.llm.model import get_chat_model
from agents.codex.graphs.reviewer.graph import graph as reviewer_graph
import time
from e2b_code_interpreter import AsyncSandbox
from shared.logger import get_logger

logger = get_logger("codex.programmer")

async def _initialize(state: ProgrammerState) -> ProgrammerState:
    sandbox_id = state.get("sandbox_session_id")
    if not sandbox_id:
        return state
    sbx: AsyncSandbox = await AsyncSandbox.connect(sandbox_id)
    logger.info("Sandbox ready")
    return state


async def _generate_action(state: ProgrammerState) -> ProgrammerState:
    # pass through for now
   return state


def _route_after_generate_action(state: ProgrammerState):
    plan = state.get("taskPlan") or {}
    items = list(plan.get("items", []))
    has_todo = any(i.get("status") != "done" for i in items)
    return "take-action" if has_todo else "reviewer-subgraph"


async def _take_action(state: ProgrammerState) -> ProgrammerState:
    # Action ReAct agent: implement the chosen task using sandbox tools
    plan: TaskPlan | None = state.get("taskPlan")
    if not plan:
        raise ValueError("No plan found")
    
    chosen = None

    for idx, item in enumerate(plan.get("items", [])):
        if item.get("status") != "done":
            chosen = item.get("description")
            break
    
    if chosen is None:
        logger.info("All tasks are done")
        return state


    SYSTEM_PROMPT = (
        "You are an action agent. Implement the assigned task using the sandbox.\n"
        "Use run_command_in_sandbox to edit files (e.g., with sed/ed/node), run tests, and verify changes.\n"
        "When done, return a brief status summary."
    )

    agent = create_agent(
        model=get_chat_model(),
        tools=[run_command_in_sandbox],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ProgrammerState,
    )

    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": f"Implement task: {chosen}"})
    result = await agent.ainvoke({
        "messages": msgs,
        # Needed by tool runtime
        "sandbox_session_id": state.get("sandbox_session_id"),
        "repo_path": state.get("repo_path"),
    })

    summary = ""
    for m in result.get("messages", []):
        if getattr(m, "type", getattr(m, "role", "")) in ("ai", "assistant"):
            summary = (getattr(m, "content", "") or "").strip()


    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": f"ActionSummary: {summary or 'Updated task.'}"})
    plan.get("items", [])[idx]["status"] = "done"

    new_state = dict(state)
    new_state["taskPlan"] = plan
    new_state["messages"] = messages
    return new_state


def _generate_conclusion(state: ProgrammerState) -> ProgrammerState:
    messages = list(state.get("messages", []))
    messages.append({"role": "system", "content": "Programming phase complete."})
    state = dict(state)
    state["messages"] = messages
    return state


def _run_reviewer(state: ProgrammerState) -> ProgrammerState:
    reviewer_input = {
        "request": state.get("request", ""),
        "repo_path": state.get("repo_path", ""),
        "messages": state.get("messages", []),
        "taskPlan": state.get("taskPlan"),
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
    workflow.add_node("generate-action", _generate_action)
    workflow.add_node("take-action", _take_action)
    workflow.add_node("generate-conclusion", _generate_conclusion)
    workflow.add_node("reviewer-subgraph", _run_reviewer)

    workflow.add_edge(START, "initialize")
    workflow.add_edge("initialize", "generate-action")
    workflow.add_conditional_edges(
        "generate-action",
        _route_after_generate_action,
        path_map=["take-action", "reviewer-subgraph"],
    )
    workflow.add_edge("take-action", "generate-action")
    workflow.add_edge("reviewer-subgraph", "generate-conclusion")
    workflow.add_edge("generate-conclusion", END)

    return workflow.compile()

graph = compile_programmer_graph()