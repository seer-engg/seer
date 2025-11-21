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


ANALYST_AGENT_SYSTEM_PROMPT = """### PROMPT: ANALYST_AGENT_SYSTEM_PROMPT (EVAL_AGENT/REFLECT) ###
You are a Senior QA Analyst and Root Cause Investigator. You are the "Fitness Function" in an evolutionary system.

Your goal is to:
1.  Identify the "fittest" test cases (the ones that **failed**).
2.  Hypothesize the root cause (the "gene") of that failure.
3.  Critique the *entire test generation* (the "environmental pressure").
4.  Recommend specific "mutations" for the next generation.
5.  **NEW: Critique the "Judge" (the scoring rubric) itself.**

**Your mandatory investigation process:**

1.  **Get Evidence:** Call `get_latest_run_results()` to see what just happened.
2.  **Formulate Root Cause Hypothesis:**
    * Look at all failures. Find the *pattern* (the "gene").
    * **Bad summary:** "Test 1 (divide_by_zero) failed."
    * **Good summary:** "The agent appears to lack fundamental error handling for `ZeroDivisionError`, as seen in test 1."
    * Use `think()` to write down your hypothesis.
3.  **Investigate Flakiness:**
    * If a test **passed** but looks suspicious (e.g., an error case that passed), or if a test **failed** that passed before, you MUST call `get_historical_test_results()` to check for flakiness.
4.  **Formulate Test Critique (Meta-Reflection):**
    * Now, critique the test cases that were just run.
    * If all tests passed: "The tests were too easy. The 'mutations' were not aggressive enough. We need to increase the complexity."
    * If tests failed: "These tests successfully found a bug. The *next* batch of tests should mutate this specific failure mode."
5.  **Save Final Analysis (Genetic Blueprint):**
    * Call `save_reflection()` with your final analysis.
    * `summary`: Your **root cause hypothesis**.
    * `test_generation_critique`: Your **meta-reflection on test quality**.

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


async def reflect_node(state: EvalAgentState) -> dict:
    """
    Runs the Analyst Agent to investigate the latest test run
    and produce a high-quality reflection.
    """
    logger.info("reflect_node: Starting Analyst Agent investigation...")

    if not state.context.user_context or not state.context.user_context.user_id:
        raise ValueError("UserContext with user_id is required to reflect")

    if not state.latest_results:
        logger.warning("reflect_node: No results to analyze. Skipping.")
        return {"attempts": state.attempts + 1}
    
    failed_tests = [r for r in state.latest_results if not r.passed]
    
    initial_prompt = ""
    if not failed_tests:
        # All tests passed - this is a failure of *test generation*
        logger.info("reflect_node: All tests passed. Priming Analyst to critique test quality.")
        initial_prompt = (
            "All tests passed. This is a failure of test generation. "
            "Your primary goal is to critique these tests (step 4) and "
            "the judge's leniency (step 5). "
            "Start your investigation."
        )
    else:
        # Some tests failed - this is a failure of the *agent*
        logger.info(f"reflect_node: {len(failed_tests)} tests failed. Priming Analyst to find root cause.")
        failed_ids = [r.dataset_example.example_id for r in failed_tests]
        initial_prompt = (
            f"Investigation required. Tests {failed_ids} failed. "
            "Your primary goal is to find the *root cause pattern* (the 'gene') "
            "connecting these failures (step 2). "
            "Start your investigation by calling `get_latest_run_results()`."
        )


    # Define the context we will pass to the agent's tools
    tool_context = ReflectionToolContext(
        user_id=state.context.user_context.user_id,
        agent_name=state.context.github_context.agent_name,
        attempts=state.attempts,
        latest_results=state.latest_results,
        raw_request=state.context.user_context.raw_request,
    )
    
    # Invoke the agent
    _ = await analyst_agent_runnable.ainvoke(
        {"messages": [HumanMessage(content=initial_prompt)]},
        config=RunnableConfig(recursion_limit=100),
        context=tool_context  # Pass the context for the tools
    )

    return {"attempts": state.attempts + 1}
