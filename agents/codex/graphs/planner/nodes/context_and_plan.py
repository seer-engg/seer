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

from agents.codex.graphs.planner.format_thread import fetch_thread_timeline_as_string
from sandbox.constants import TARGET_AGENT_LANGSMITH_PROJECT


SYSTEM_PROMPT = f"""
    You are an Technical manager specializing in LLM based Agent development.
    Your role is to Create a plan for the Agent to be developed.
    Your task is to plan the next steps to be taken to improve the agent by analyzing the failed eval thread  of the agent  and understanding the current state of the agent through its code .
    Create a plan with 3-7 concrete steps to fulfill the request.
    
    ## Notes:
    - use respective tools to gather context and plan the task.
"""

USER_PROMPT = """
    Analyse the following eval test cases and corresponding  thread trace of the agent .
    <EVALS AND THREAD TRACES>
    {evals_and_thread_traces}
    </EVALS AND THREAD TRACES>

    Create a plan with 3-7 concrete steps to improve the agent.
"""

EVALS_AND_THREAD_TRACE_TEMPLATE = """
    
    <EVAL> 
    {eval}
    </EVAL>
    <THREAD TRACE>
    {thread_trace}
    </THREAD TRACE>
"""


async def context_and_plan_agent(state: PlannerState) -> PlannerState:
    """Single ReAct agent that gathers repo context and returns a concrete plan."""

    # Extract sandbox context for tools
    updated_sandbox_context = state.updated_sandbox_context
    if not updated_sandbox_context:
        raise ValueError("No sandbox context found in state")

    agent = create_agent(
        model=get_chat_model(reasoning_effort="high"),
        tools=[
            run_command,
            inspect_directory,
            read_file,
            grep,
            web_search,
        ],
        system_prompt=SYSTEM_PROMPT,
        state_schema=PlannerState,
        response_format=TaskPlan,
        context_schema=SandboxToolContext,  # Add context schema for sandbox tools
    )

    evals_and_thread_traces=[] 
    for eval in state.experiment_context.results:
        if eval.passed:
            continue
        x={
            "INPUT:": eval.dataset_example.input_message,
            "EXPECTED OUTPUT:": eval.dataset_example.expected_output,
            "ACTUAL OUTPUT:": eval.actual_output,
            "SCORE:": eval.score,
            "JUDGE FEEDBACK:": eval.judge_reasoning
        }
        thread_trace = await fetch_thread_timeline_as_string(eval.thread_id, TARGET_AGENT_LANGSMITH_PROJECT)
        evals_and_thread_traces.append(
            EVALS_AND_THREAD_TRACE_TEMPLATE.format(
                eval=x,
                thread_trace=thread_trace
            )
        )

    msgs = list(state.messages or [])
    msgs.append(HumanMessage(content=USER_PROMPT.format(evals_and_thread_traces=evals_and_thread_traces)))

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