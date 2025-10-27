from __future__ import annotations

from langgraph.graph import END, START, StateGraph

import os
from agents.codex.common.state import PlannerState, TaskItem, TaskPlan
from agents.codex.llm.model import get_chat_model, generate_plan_steps
from agents.codex.graphs.shared.initialize_sandbox import (
    ensure_sandbox_ready,
    initialize_e2b_sandbox,
    cd_and_run_in_sandbox,
)
from shared.logger import get_logger
from langchain.agents import create_agent
from langchain.tools import tool

logger = get_logger("codex.planner")


def _prepare_graph_state(state: PlannerState) -> PlannerState:
    state = dict(state)
    state.setdefault("messages", [])
    state.setdefault("autoAcceptPlan", True)
    return state


def _initialize_sandbox(state: PlannerState) -> PlannerState:
    # If a remote repo URL is provided, initialize an E2B sandbox and clone/pull there.
    repo_url = state.get("repo_url")
    logger.info(f"State: {state}")
    logger.info(f"Initializing sandbox for repo_url: {repo_url}")
    if repo_url:
        branch_name = state.get("branch_name") or "main"
        github_token = os.getenv("GITHUB_TOKEN")
        existing_id = state.get("sandbox_session_id")

        sandbox_id, repo_dir, branch_in_sandbox = initialize_e2b_sandbox(
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
    ensure_sandbox_ready(state.get("repo_path", ""))
    return state


def _generate_plan(state: PlannerState) -> PlannerState:
    model = get_chat_model()
    request = state.get("request", "")
    steps = generate_plan_steps(model, request, state.get("repo_context"))
    plan_items: list[TaskItem] = [
        {"description": s, "status": "todo"} for s in steps
    ]
    state = dict(state)
    state["taskPlan"] = TaskPlan(title=f"Plan: {request[:50]}", items=plan_items)
    return state


## Manual bounded context actions removed in favor of ReAct agent


def _run_context_agent(state: PlannerState) -> PlannerState:
    """ReAct agent that extracts concise repo context by calling in-sandbox tools."""
    if not (state.get("sandbox_session_id") and state.get("repo_path")):
        return state

    sandbox_id = state["sandbox_session_id"]
    repo_dir = state["repo_path"]

    @tool
    def list_files(_: str = "") -> str:
        """List the files in the repository."""
        code, out, err = cd_and_run_in_sandbox(sandbox_id, repo_dir, "ls -la | head -n 200")
        return out or err or ""

    @tool
    def read_readme(_: str = "") -> str:
        """Read the README.md file in the repository."""
        code, out, err = cd_and_run_in_sandbox(sandbox_id, repo_dir, "[ -f README.md ] && sed -n '1,200p' README.md || true")
        return out or err or ""

    @tool
    def show_dependencies(_: str = "") -> str:
        """Show the dependencies in the repository."""
        code, out, err = cd_and_run_in_sandbox(
            sandbox_id,
            repo_dir,
            "([ -f package.json ] && cat package.json) || ([ -f pyproject.toml ] && sed -n '1,200p' pyproject.toml) || true",
        )
        return out or err or ""

    SYSTEM_PROMPT = (
        "You are a repo context agent. Gather only what's needed for high-level planning: "
        "top-level structure, readme gist, and main dependency manifest. "
        "Return a concise bullet summary; don't include raw listings beyond 20 lines each."
    )

    agent = create_agent(
        model=get_chat_model(),
        tools=[list_files, read_readme, show_dependencies],
        system_prompt=SYSTEM_PROMPT,
    )

    msgs = list(state.get("messages", []))
    msgs.append({"role": "user", "content": "Summarize repository context for planning."})
    result = agent.invoke({"messages": msgs})
    summary = ""
    try:
        # langchain agent returns dict with messages
        for m in result.get("messages", []):
            if getattr(m, "type", getattr(m, "role", "")) in ("ai", "assistant"):
                summary = getattr(m, "content", "") or summary
    except Exception:
        pass

    messages = list(state.get("messages", []))
    if summary:
        messages.append({"role": "system", "content": f"RepoContext:\n{summary}"})
    state = dict(state)
    state["messages"] = messages
    state["repo_context"] = summary
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
    workflow.add_node("react-context-agent", _run_context_agent)
    workflow.add_node("notetaker", _notetaker)
    workflow.add_node("interrupt-proposed-plan", _interrupt_proposed_plan)

    workflow.add_edge(START, "prepare-graph-state")
    workflow.add_edge("prepare-graph-state", "initialize-sandbox")
    workflow.add_edge("initialize-sandbox", "react-context-agent")
    workflow.add_edge("react-context-agent", "generate-plan")
    workflow.add_edge("generate-plan", "notetaker")
    workflow.add_edge("notetaker", "interrupt-proposed-plan")
    workflow.add_edge("interrupt-proposed-plan", END)

    return workflow.compile()

graph = compile_planner_graph()