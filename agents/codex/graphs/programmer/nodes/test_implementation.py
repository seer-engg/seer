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
    SandboxToolContext,
)
from shared.tools import web_search, think
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = f"""
    You are a software engineer in testing. another software engineer has implemented a task and you are to test the implementation, by creating and running unit tests for the implementation.
    Available tools:

    {read_file.description}
    {grep.description}
    {inspect_directory.description}
    {create_file.description}
    {create_directory.description}
    {apply_patch.description}
    {write_file.description}
    {patch_file.description}
    {web_search.description}
    {think.description}
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
    sandbox_context = state.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")

    # Extract sandbox context for tools
    sandbox_context = state.sandbox_context
    if not sandbox_context:
        raise ValueError("No sandbox context found in state")

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
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    )

    user_prompt = USER_PROMPT.format(request=state.user_context.user_expectation, task_plan=plan)
    state.messages.append(HumanMessage(content=user_prompt))    

    # Pass context along with state
    result = await agent.ainvoke(
        state, 
        config=RunnableConfig(recursion_limit=100),
        context=SandboxToolContext(sandbox_context=sandbox_context)  # Pass sandbox context
    )

    test_results: TestResults = result.get("structured_response", TestResults(success=True, failures=[]))
    return {
        "testResults": test_results,
    }
