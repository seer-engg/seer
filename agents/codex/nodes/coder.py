"""Implement the task plan"""
import asyncio # ADDED
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from shared.tools import web_search
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
    apply_patch,    
    search_code,
    search_symbols,
    semantic_search,
    get_symbol_definition,
    find_usages,
    get_code_region,
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

SYSTEM_PROMPT = """### PROMPT: SYSTEM_PROMPT (CODEX/CODER) ###
    You are a software engineer. You have been given a task to implement. Implement the assigned task to the codebase in the sandbox.
    When done, return a brief status summary. You just need to implement the task, you don't need to generate or run any test the implementation.
    You have been provided with following tools to do necessary operation in root directory of the codebase repository.
    
    # ADDED:
    You also have access to external service tools (like 'asana.create_task', 'github.update_pr').
    Use these tools to write the logic for the agent you are building.

    # Important Notes:
    - use desired tools to implement the task.
    - for searching of packages, use the web_search tool, do not use pip search.
"""


async def coder(state: CodexState) -> CodexState:
    """Action ReAct agent: implement the chosen task using sandbox tools"""
    plan: TaskPlan | None = state.taskPlan
    if not plan:
        raise ValueError("No plan found")
    sandbox_context = state.context.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")

    
    all_tools = [
        run_command,
        read_file,
        grep,
        inspect_directory,
        create_file,
        create_directory,
        write_file,
        patch_file,
        web_search,
        # TODO: ADD langchain and other mcp tools required for target agent documentations
        # *mcp_tools, # Add the dynamic tools
    ]

    agent = create_agent(
        model=get_llm(model="codex"),
        tools=all_tools, # MODIFIED
        system_prompt=SYSTEM_PROMPT,
        state_schema=CodexState,
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    )

    # Prepare messages for the agent
    messages = list(state.coder_thread or [])
    
    user_prompt_content = USER_PROMPT.format(
        request=state.context.user_context.raw_request,
        task_plan=state.taskPlan,
    )
    # TODO: when coming from test_server_ready, do not add the user_prompt_content

    messages.append(HumanMessage(content=user_prompt_content))

    result = await agent.ainvoke(
        input={"messages": messages}, 
        config=RunnableConfig(recursion_limit=100),
        context=SandboxToolContext(sandbox_context=sandbox_context)  # Pass sandbox context
    )

    pr_summary = ""
    last_message = result.get("messages", [])[-1].content
    if isinstance(last_message, list):
        for message in last_message:
            if message.get("type") == "text":
                pr_summary += message.get("text")

    return {
        "taskPlan": plan,
        "coder_thread": result.get("messages"),
        "planner_thread": [AIMessage(content=pr_summary)],
        "pr_summary": pr_summary,
    }
