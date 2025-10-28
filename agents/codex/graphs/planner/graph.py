from __future__ import annotations

from langgraph.graph import END, START, StateGraph

import os
import json
from agents.codex.common.state import PlannerState, TaskItem, TaskPlan
from agents.codex.llm.model import get_chat_model
from sandbox.base import (
    ensure_sandbox_ready,
    initialize_e2b_sandbox,
)
from shared.logger import get_logger
from langchain.agents import create_agent

from sandbox.tools import run_command_in_sandbox

logger = get_logger("codex.planner")


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state


async def _initialize_sandbox(state: PlannerState) -> PlannerState:
    # If a remote repo URL is provided, initialize an E2B sandbox and clone/pull there.
    repo_url = state.get("repo_url")
    logger.info(f"State: {state}")
    logger.info(f"Initializing sandbox for repo_url: {repo_url}")
    if repo_url:
        branch_name = state.get("branch_name") or "main"
        github_token = os.getenv("GITHUB_TOKEN")
        existing_id = state.get("sandbox_session_id")

        sandbox_id, repo_dir, branch_in_sandbox = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
            existing_sandbox_id=existing_id,
        )

        messages = list(state.get("messages", []))
        messages.append({
            "role": "system",
            "content": f"E2B sandbox ready (id={sandbox_id}); repo cloned at {repo_dir} on branch {branch_in_sandbox}.",
        })

        new_state = dict(state)
        new_state["sandbox_session_id"] = sandbox_id
        # Store repo_dir in repo_path so downstream context actions have a path
        new_state["repo_path"] = repo_dir
        new_state["messages"] = messages
        return new_state

    # Local mode: validate provided repo_path exists
    await ensure_sandbox_ready(state.get("repo_path", ""))
    return state


async def _context_and_plan_agent(state: PlannerState) -> PlannerState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""
    if not (state.get("sandbox_session_id") and state.get("repo_path")):
        return state

    sandbox_id = state["sandbox_session_id"]
    repo_dir = state["repo_path"]
    request = state.get("request", "")

    SYSTEM_PROMPT = """
        You are an agent specializing in planning by gathering context about a codebase . Gather only what's needed for high-level planning: 
        Create a plan with 3-7 concrete steps to fulfill the request.

        Available tools:
        - run_command_in_sandbox: Run a command in the sandbox in working directory of the repo.
            - Parameters:
                - command: The command to run.
        
        ## Notes:
        - You should always use the run_command_in_sandbox tool to run commands in the sandbox.
        - you can execute any commands to inspect the codebase and gather context.
    """

    agent = create_agent(
        model=get_chat_model(),
        tools=[
            run_command_in_sandbox
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=PlannerState,
        response_format=TaskPlan,
    )

    msgs = list(state.get("messages", []))
    msgs.append({
        "role": "user",
        "content": (
            "Task: " + request + "\n"
            "Gather minimal repo context and return JSON with repo_context and plan_steps.\n"
            "Always use the tool to inspect the repo."
        ),
    })

    result = await agent.ainvoke({
        "messages": msgs,
        "sandbox_session_id": sandbox_id,
        "repo_path": repo_dir,
    })
    logger.info(f"Result: {result.keys()}")
    logger.info(f"Result: {result.get('structured_response')}")
    taskPlan: TaskPlan = result.get("structured_response")

    return {
        "messages": result.get("messages", []),
        "taskPlan": taskPlan,
    }


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
    # Combined context + planning agent
    workflow.add_node("context-plan-agent", _context_and_plan_agent)
    # Removed separate context agent; using combined agent instead
    workflow.add_node("notetaker", _notetaker)
    workflow.add_node("interrupt-proposed-plan", _interrupt_proposed_plan)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-sandbox")
    workflow.add_edge("initialize-sandbox", "context-plan-agent")
    workflow.add_edge("context-plan-agent", "notetaker")
    workflow.add_edge("notetaker", "interrupt-proposed-plan")
    workflow.add_edge("interrupt-proposed-plan", END)

    return workflow.compile()

graph = compile_planner_graph()