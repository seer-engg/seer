from __future__ import annotations

from shared.logger import get_logger
logger = get_logger("codex.planner.nodes.context_and_plan_agent")

from sandbox.tools import run_command_in_sandbox
from sandbox import Sandbox

from langchain.agents import create_agent
from agents.codex.llm.model import get_chat_model
from agents.codex.common.state import PlannerState, TaskPlan

async def context_and_plan_agent(state: PlannerState) -> PlannerState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""
    if not (state.sandbox_session_id and state.repo_path):
        raise ValueError("Sandbox session ID and repository path are required")

    sandbox_id = state.sandbox_session_id
    repo_dir = state.repo_path
    request = state.request

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

    msgs = list(state.messages)
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
