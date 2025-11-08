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
ANALYST_AGENT_SYSTEM_PROMPT = """### PROMPT: ANALYST_AGENT_SYSTEM_PROMPT (EVAL_AGENT/REFLECT) ###
You are a Senior QA Analyst and Root Cause Investigator. Your goal is to generate a high-level *hypothesis* about the agent's failure modes, not just list what failed. You must also critique the *quality* of the tests themselves.

**Your mandatory investigation process:**

1.  **Get Evidence:** Call `get_latest_run_results()` to see what just happened.
2.  **Formulate Root Cause Hypothesis:**
    * Look at all the failures. Do not just repeat the failure. Find the *pattern*.
    * **Bad summary:** "Test 1 (divide_by_zero) failed."
    * **Good summary:** "The agent appears to lack fundamental error handling for `ZeroDivisionError`, as seen in test 1."
    * **Good summary:** "The agent consistently fails to handle nested dictionary inputs, suggesting a problem with recursive logic."
    * Use `think()` to write down your hypothesis before moving on.
3.  **Investigate Flakiness:**
    * Review the results. If a test **passed** but looks suspicious (e.g., an error case that passed), or if a test **failed** that passed before, you MUST call `get_historical_test_results()` to check for flakiness.
4.  **Formulate Test Critique (Meta-Reflection):**
    * Now, critique the test cases that were just run. This is CRITICAL for improving the next test generation round.
    * Ask: "Were these tests *good*?"
    * If all tests passed: "Were the tests too easy? Did they miss obvious edge cases?"
    * If tests failed: "Did they find the *right* bug? How can the *next* batch of tests be even harder and explore the *root cause* identified in step 2?"
    * Use `think()` to write down your test critique.
5.  **Save Final Analysis:**
    * Call `save_reflection()` with your final analysis.
    * The `summary` field MUST contain your **root cause hypothesis**.
    * The `test_generation_critique` field MUST contain your **meta-reflection on test quality**.
    * The `recommended_tests` field MUST contain *new, harder* test ideas based on your hypothesis (e.g., "test dict access with 3 levels of nesting," "test divide_by_zero inside a loop").

This is your final step. Do not add any more steps.
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
