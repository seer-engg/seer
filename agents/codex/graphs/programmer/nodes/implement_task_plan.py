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
)
from agents.codex.common.tools import think
from langchain_core.messages import AIMessage

SYSTEM_PROMPT = """
    You are a junior software engineer.You have been given a task to implement. Implement the assigned task to the codebase in the sandbox.
    Use run_command to access the terminal of the sandbox and execute commands to implement the task.
    When done, return a brief status summary.

    Available tools:
    - run_command: Run a command in the working directory of the repository.
        - Parameters:
            - command: The command to run.
    - think: Think about something.
        - Parameters:
            - thought: The thought to think about.
    - read_file: Read a file in the repository.
        - Parameters:
            - file_path: The path to the file to read.
    - grep: Grep in the repository.
        - Parameters:
            - pattern: The pattern to grep.
    - inspect_directory: To understand the directory structure and files in the repository.
        - Parameters:
            - directory_path: The path to the directory to list files from.
            - depth: The depth of the directory tree to inspect.
    - create_file: Create a file in the repository.
        - Parameters:
            - file_path: The path to the file to create.
            - content: The content of the file to create.
    - create_directory: Create a directory in the repository.
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
            run_command,
            read_file,
            grep,
            inspect_directory,
            create_file,
            create_directory,
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
