"""
Node for reflecting on the latest test results.
This node is an "Analyst Agent" that investigates failures and flakiness.
"""
from typing import List
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from agents.eval_agent.constants import LLM
from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.nodes.reflect.tools import (
    get_latest_run_results,
    get_historical_test_results,
    save_reflection,
    ReflectionToolContext,
)
from shared.tools import think
from shared.logger import get_logger

logger = get_logger("eval_agent.reflect")


# 1. Define the System Prompt for the "Analyst Agent"
ANALYST_AGENT_SYSTEM_PROMPT = """You are a senior QA Analyst. Your goal is to investigate the latest test run to identify new failures, regressions, and, most importantly, *test flakiness*.

Your investigation process:
1.  Start by calling `get_latest_run_results()` to see what just happened.
2.  **CRITICAL CHECK:** Compare the failures from step 1 to the *old* failures documented in `reflections_used_for_planning` (available in your context, use `think()` to analyze). For each new failure, determine: Is this a *truly novel bug* (a new failure mode), or just a repeat/regression of a known issue?
3.  Review the results. For any test that **passed**, especially if it looks like a test that *should* fail (e.g., divide_by_zero), you MUST call `get_historical_test_results()` to check if it has *ever* failed in the past. This is how you detect flakiness.
4.  For any test that **failed**, this is a new failure mode.
5.  Use `think()` to form your hypotheses about new failures, flakiness, or other patterns.
6.  Once your investigation is complete, call `save_reflection()` with your final analysis. This is your final step.

**Key Insight to find:**
* **New Failures:** What new bugs did you just find?
* **Flakiness:** Which tests are *unreliable* (pass sometimes, fail other times)?
* **Next Steps:** What *new* tests should be generated to explore these failures or confirm flakiness (e.g., "re-run divide_by_zero 3 times")?
"""

# 2. Create the Agent Runnable
tools: List = [
    think,
    get_latest_run_results,
    get_historical_test_results,
    save_reflection,
]

# This creates the ReAct agent
analyst_agent_runnable = create_agent(
    model=LLM,
    tools=tools,
    system_prompt=ANALYST_AGENT_SYSTEM_PROMPT,
    state_schema=EvalAgentState,
)


# 3. Define the Graph Node
async def reflect_node(state: EvalAgentState) -> dict:
    """
    Runs the Analyst Agent to investigate the latest test run
    and produce a high-quality reflection.
    """
    logger.info("reflect_node: Starting Analyst Agent investigation...")

    if not state.user_context or not state.user_context.user_id:
        raise ValueError("UserContext with user_id is required to reflect")

    if not state.latest_results:
        logger.warning("reflect_node: No results to analyze. Skipping.")
        return {"attempts": state.attempts + 1}

    # Define the context we will pass to the agent's tools
    tool_context = ReflectionToolContext(
        user_id=state.user_context.user_id,
        agent_name=state.github_context.agent_name,
        attempts=state.attempts,
        latest_results=state.latest_results,
        user_expectation=state.user_context.user_expectation,
        reflections_used_for_planning=state.reflections_used_for_planning
    )
    
    # Define the initial prompt to kick off the agent
    initial_prompt = "Start your investigation of the latest test run."

    # Invoke the agent
    _ = await analyst_agent_runnable.ainvoke(
        {"messages": [HumanMessage(content=initial_prompt)]},
        config=RunnableConfig(recursion_limit=100),
        context=tool_context  # Pass the context for the tools
    )

    return {"attempts": state.attempts + 1}
