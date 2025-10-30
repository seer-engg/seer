from agents.codex.common.state import ProgrammerState, TaskPlan, TestResults
from shared.logger import get_logger
logger = get_logger("programmer.test_implementation")
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
from shared.tools import web_search, think
from langchain_core.runnables import RunnableConfig


SYSTEM_PROMPT = f"""
    You are a software engineer in testing. another software engineer has implemented a task and you are to test the implementation, by creating and running unit tests for the implementation.
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
"""

USER_PROMPT = """
    based on the request 
    <request>
    {request}
    </request>

    You have to test the following implementation for the requested task:
    <implementation>
    {task_plan}
    </implementation>

    create unit tests for the implementation and run them to test the implementation.

    after testing the implementation, return a TestResults object with the following fields:
    - success: whether the tests passed for the requested implementations
    - failures: the failures of the tests
        - test_intention: the intention of the test
        - failure_reason: the reason for the failure
        - optinal_refrence: an optional refrence to the code that failed the test
"""



async def test_implementation(state: ProgrammerState) -> ProgrammerState:

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
            think,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=ProgrammerState,
        response_format=TestResults,
    )

    user_prompt = USER_PROMPT.format(request=state.request, task_plan=plan)
    msgs = []
    msgs.append({"role": "user", "content": user_prompt})
    result = await agent.ainvoke({
        "messages": msgs,
        # Needed by tool runtime
        "sandbox_session_id": state.sandbox_session_id,
        "repo_path": state.repo_path,
    }, config = RunnableConfig(recursion_limit=100))

    test_results: TestResults = result.get("structured_response", TestResults(success=True, failures=[]))
    return {
        "testResults": test_results,
    }
