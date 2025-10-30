from langchain_core.messages.base import BaseMessage


from shared.logger import get_logger
from agents.codex.common.state import ProgrammerState, TaskPlan
logger = get_logger("programmer.execute_task_item")
from langchain.agents import create_agent
from agents.codex.llm.model import get_chat_model
from sandbox.tools import (
    run_command,
    read_file,
    grep,
    inspect_directory,
    create_file,
    create_directory,
    apply_patch,
    write_file,
    patch_file,
)
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from shared.tools import web_search, think
from langgraph.types import Command
from langchain.tools import tool,ToolRuntime

@tool
def mark_task_item_as_done(task_item_id: int, runtime: ToolRuntime) -> str:
    """
    Mark a task item as done.
    Args:
        task_item_id: The id of the task item to mark as done
    Returns:
        A command to update the task plan
    """
    taskplan: TaskPlan = runtime.state.get('taskPlan')  
    if not taskplan:
        raise ValueError("No task plan found")
    for item in taskplan.items:
        if item.id == task_item_id:
            item.status = "done"
            logger.info(f"Marked task item as done: {item.description}")
            break
    else:
        raise ValueError(f"Task item with id {task_item_id} not found")
    return Command(
        update={
            "taskPlan": taskplan,
        }
    )


SYSTEM_PROMPT = f"""
    You are a junior software engineer. You have been given a task to implement. Implement the assigned task to the codebase in the sandbox.
    When done, return a brief status summary. You just need to implement the task, you don't need to generate or run any test the implementation.
    You have been provided with following tools to do necessary operation in root directory of the codebase repository.

    Available tools:
    {read_file.__doc__}
    {grep.__doc__}
    {inspect_directory.__doc__}
    {create_file.__doc__}
    {create_directory.__doc__}
    {apply_patch.__doc__}
    {write_file.__doc__}
    {patch_file.__doc__}
    {web_search.__doc__}
    {think.__doc__}
    {mark_task_item_as_done.__doc__}

    # Important Notes:
    - Always use the think tool to think about the task before implementing it.
    - use desired tools to implement the task.
    - for searching of packages, use the web_search tool, do not use pip search.
"""


async def implement_task_plan(state: ProgrammerState) -> ProgrammerState:
    # Action ReAct agent: implement the chosen task using sandbox tools
    plan: TaskPlan | None = state.taskPlan
    if not plan:
        raise ValueError("No plan found")


    agent = create_agent(
        model=get_chat_model(),
        tools=[
            run_command,
            read_file,
            grep,
            inspect_directory,
            create_file,
            create_directory,
            think,
            write_file,
            patch_file,
            web_search,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ProgrammerState,
    )

    result = await agent.ainvoke({
        "messages": state.messages,
        # Needed by tool runtime
        "sandbox_session_id": state.sandbox_session_id,
        "repo_path": state.repo_path,
    }, config = RunnableConfig(recursion_limit=100))
    return {
        "taskPlan": plan,
        "messages": result.get("messages", []),
    }
