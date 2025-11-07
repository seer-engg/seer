"""Implement the task plan"""
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from shared.tools import web_search, LANGCHAIN_MCP_TOOLS
from shared.logger import get_logger
from shared.llm import get_llm
from agents.codex.state import CodexState, TaskPlan
from sandbox.tools import (
    run_command,
    read_file,
    grep,
    inspect_directory,
    _inspect_directory_impl,
    create_file,
    create_directory,
    write_file,
    patch_file,
    SandboxToolContext,
)

logger = get_logger("codex.implement_task_plan")

USER_PROMPT = """
    based on the request 
    <request>
    {request}
    </request>

    Implement the following task plan:
    <task_plan>
    {task_plan}
    </task_plan>

    After implementing the task plan, return a brief status summary.
"""

SYSTEM_PROMPT = """
    You are a software engineer. You have been given a task to implement. Implement the assigned task to the codebase in the sandbox.
    When done, return a brief status summary. You just need to implement the task, you don't need to generate or run any test the implementation.
    You have been provided with following tools to do necessary operation in root directory of the codebase repository.

    # Important Notes:
    - use desired tools to implement the task.
    - for searching of packages, use the web_search tool, do not use pip search.
"""


async def implement_task_plan(state: CodexState) -> CodexState:
    """Action ReAct agent: implement the chosen task using sandbox tools"""
    plan: TaskPlan | None = state.taskPlan
    if not plan:
        raise ValueError("No plan found")
    updated_sandbox_context = state.updated_sandbox_context
    if not updated_sandbox_context:
        raise ValueError("No sandbox context found in state")

    # Ground the agent with the current directory structure
    initial_context = await _inspect_directory_impl(
        directory_path=".",
        depth=2,
        sandbox_context=updated_sandbox_context,
    )

    agent = create_agent(
        model=get_llm(model="codex"),
        tools=[
            run_command,
            read_file,
            grep,
            inspect_directory,
            create_file,
            create_directory,
            write_file,
            patch_file,
            web_search,
            *LANGCHAIN_MCP_TOOLS,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=CodexState,
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    )

    # Prepare messages for the agent
    messages = list(state.messages or [])
    
    # Check if this is the first attempt (no reflections yet)
    if state.attempt_number == 0:
         # --- Start of proposed change ---
         # Prepend the file system context to the user prompt
        user_prompt_content = USER_PROMPT.format(
            request=state.user_context.user_expectation,
            task_plan=state.taskPlan,
        )
        initial_prompt = (
            "Here is the current structure of the codebase you are working in:\n"
            f"<directory_listing>\n{initial_context}\n</directory_listing>\n\n"
            f"{user_prompt_content}"
        )
        messages.append(HumanMessage(content=initial_prompt))
        # --- End of proposed change ---

    # If this is a reflection loop, the 'reflect' node already added the new HumanMessage
    
    # Pass context along with state
    result = await agent.ainvoke(
        input={"messages": messages}, 
        config=RunnableConfig(recursion_limit=100),
        context=SandboxToolContext(sandbox_context=updated_sandbox_context)  # Pass sandbox context
    )

    pr_summary = ""
    last_message = result.get("messages", [])[-1].content
    if isinstance(last_message, list):
        for message in last_message:
            if message.get("type") == "text":
                pr_summary += message.get("text")

    return {
        "taskPlan": plan,
        "messages": result.get("messages", []), # Pass along the full history
        "pr_summary": pr_summary,
    }
