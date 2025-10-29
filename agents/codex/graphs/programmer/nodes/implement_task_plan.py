from shared.logger import get_logger
from agents.codex.common.state import ProgrammerState, TaskPlan
logger = get_logger("programmer.execute_task_item")
from langchain.agents import create_agent
from agents.codex.llm.model import get_chat_model
from sandbox.tools import (
    run_command_in_sandbox,
    read_file_in_sandbox,
    grep_in_sandbox,
    list_files_in_sandbox,
    create_file_in_sandbox,
    create_directory_in_sandbox,
)
from agents.codex.common.tools import think
from langchain_core.messages import AIMessage

SYSTEM_PROMPT = """
    You are a junior software engineer.You have been given a task to implement. Implement the assigned task to the codebase in the sandbox.
    Use run_command_in_sandbox to access the terminal of the sandbox and execute commands to implement the task.
    When done, return a brief status summary.

    Available tools:
    - run_command_in_sandbox: Run a command in the sandbox in working directory of the repo.
        - Parameters:
            - command: The command to run.
    - think: Think about something.
        - Parameters:
            - thought: The thought to think about.
    - read_file_in_sandbox: Read a file in the sandbox.
        - Parameters:
            - file_path: The path to the file to read.
    - grep_in_sandbox: Grep in the sandbox.
        - Parameters:
            - pattern: The pattern to grep.
    - list_files_in_sandbox: List files in the sandbox.
        - Parameters:
            - directory_path: The path to the directory to list files from.
    - create_file_in_sandbox: Create a file in the sandbox.
        - Parameters:
            - file_path: The path to the file to create.
            - content: The content of the file to create.
    - create_directory_in_sandbox: Create a directory in the sandbox.
        - Parameters:
            - directory_path: The path to the directory to create.

    # Important Notes:
    - Always use the think tool to think about the task before implementing it.
    - use desired tools to implement the task.
"""

USER_PROMPT = """
    based on the request 
    <request>
    {request}
    </request>

    Implement the following task plan:
    <task_plan>
    {task_plan}
    </task_plan>
"""

async def implement_task_plan(state: ProgrammerState) -> ProgrammerState:
    # Action ReAct agent: implement the chosen task using sandbox tools
    plan: TaskPlan | None = state.taskPlan
    if not plan:
        raise ValueError("No plan found")

    agent = create_agent(
        model=get_chat_model(),
        tools=[
            run_command_in_sandbox,
            read_file_in_sandbox,
            grep_in_sandbox,
            list_files_in_sandbox,
            create_file_in_sandbox,
            create_directory_in_sandbox,
            think,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ProgrammerState,
    )

    user_prompt = USER_PROMPT.format(request=state.request, task_plan=plan)
    msgs = []
    msgs.append({"role": "user", "content": user_prompt})
    result = await agent.ainvoke({
        "messages": msgs,
        # Needed by tool runtime
        "sandbox_session_id": state.sandbox_session_id,
        "repo_path": state.repo_path,
    })
    return {
        "taskPlan": plan,
        "messages": result.get("messages", []),
    }
