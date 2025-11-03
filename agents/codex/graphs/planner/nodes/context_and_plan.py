from __future__ import annotations

from shared.logger import get_logger

logger = get_logger("codex.planner.nodes.context_and_plan_agent")

from sandbox.tools import (
    run_command,
    inspect_directory,
    read_file,
    grep,
    SandboxToolContext,
)
from shared.tools import web_search, think

from langchain.agents import create_agent
from agents.codex.llm.model import get_chat_model
from agents.codex.common.state import PlannerState, TaskPlan
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage


SYSTEM_PROMPT = f"""
    You are an Technical manager specializing in LLM based Agent development.
    Your role is to Create a plan for the Agent to be developed.
    Your task is to analyze the failed evals of the agent ,understand the current state of the agent through its code and plan the next steps to be taken to improve the agent.  
    Create a plan with 3-7 concrete steps to fulfill the request.

    Available tools:
    {run_command.description}
    {inspect_directory.description}
    {read_file.description}
    {grep.description}
    {web_search.description}
    {think.description}
    
    ## Notes:
    - use respective tools to gather context and plan the task.
"""

USER_PROMPT = """
    The Agent has failed the following evals:
    {eval_results}
    Create a plan with 3-7 concrete steps to improve the agent.
"""


async def context_and_plan_agent(state: PlannerState) -> PlannerState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""

    # Extract sandbox context for tools
    updated_sandbox_context = state.updated_sandbox_context
    if not updated_sandbox_context:
        raise ValueError("No sandbox context found in state")
    
    experiment = state.experiment_context
    if not experiment:
        raise ValueError("Experiment context is required for planning")

    failing_results = [res for res in experiment.results if not res.passed]
    if failing_results:
        eval_results = "\n\n".join(
            (
                f"Thread / Example ID: {res.dataset_example.example_id}\n"
                f"Input: {res.dataset_example.input_message}\n"
                f"Expected: {res.dataset_example.expected_output}\n"
                f"Actual: {res.actual_output}\n"
                f"Score: {res.score:.3f}\n"
                f"Judge feedback: {res.judge_reasoning}\n"
            )
            for res in failing_results
        )
    else:
        eval_results = "All recent evaluations passed."

    agent = create_agent(
        model=get_chat_model(),
        tools=[
            run_command,
            inspect_directory,
            read_file,
            grep,
            web_search,
            think,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=PlannerState,
        response_format=TaskPlan,
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    )

    msgs = list(state.messages or [])
    msgs.append(HumanMessage(content=USER_PROMPT.format(eval_results=eval_results)))

    # Pass context along with state
    result = await agent.ainvoke(
        {"messages": msgs},
        config=RunnableConfig(recursion_limit=100),
        context=SandboxToolContext(sandbox_context=updated_sandbox_context)  # Pass sandbox context
    )
    logger.info(f"Result: {result.keys()}")
    logger.info(f"Result: {result.get('structured_response')}")
    taskPlan: TaskPlan = result.get("structured_response")

    return {
        "messages": result.get("messages", []),
        "taskPlan": taskPlan,
    }