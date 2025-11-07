"""shared module for running evaluations"""
import asyncio
import os
from typing import List
from datetime import datetime, timezone

from langchain_core.prompts import PromptTemplate
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client

from shared.schema import DatasetExample, ExperimentResultContext, FailureAnalysis
from shared.logger import get_logger
from shared.llm import get_llm


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
logger = get_logger("eval_runner")


CORRECTNESS_PROMPT = """You are a hyper-strict, pathological data labeler. Your sole purpose is to evaluate an agent's output based on a *specific intent*.

**CRITICAL: This is the specific intent of this test case:**
<test_case_intent>
{test_case_intent}
</test_case_intent>

Use this intent as your *primary* source of truth. If the intent is "fix syntax only," you MUST fail the agent for changing logic. If the intent is "fix logic," you MUST fail the agent if it *only* fixes syntax.

Here is the original code:
<input>
{inputs}
</input>

Here is the agent's attempt to fulfill the intent:
<output>
{outputs}
</output>

Here is the "golden" reference output that perfectly fulfills the intent:
<reference_outputs>
{reference_outputs}
</reference_outputs>

Your task is to evaluate if the agent's output *matches the reference* AND *fulfills the specific intent*.

**Scoring Rubric:**

**1.0 = Completely correct, accurate, and complete**
- Fixes all issues, contains no bugs, preserves original structure.

**0.8 = Mostly correct with minor issues** - Fixes main issues, good structure preservation, minor syntax/logic issues.

**0.6 = Partially correct but with notable problems**
- Fixes some issues but misses others, moderate structural changes, logical errors.

**0.4 = Largely incorrect with some correct elements**
- Major structural changes, significant logical errors, some correct elements.

**0.2 = Mostly or completely incorrect**
- Fails to fix original issues, major structure violations, serious errors.

**0.0 = Total failure**
- No output, irrelevant output, or made the code significantly worse.

**Critical Structure Preservation Requirements:**
- Original function names, class names, and schemas MUST remain unchanged.
- Original invocation patterns MUST be preserved.
- You may create new internal variables or helper functions, but cannot rename existing components.
- The overall framework and interface must stay intact.

**Evaluation Process:**
1. Read the <test_case_intent> and internalize it. This is your only guide.
2. Compare <output> to <reference_outputs>.
3. Check if <output> *perfectly* achieved the goal stated in <test_case_intent>.
4. Note any deviations, *even if* the agent's output seems "better" but violates the *specific intent*. (e.g., if intent is syntax-only, "fixing" logic is a failure).
5. Assign a final score, failure type, severity, and reasoning *relative to the intent*.

Provide your final assessment as a structured object.
"""

LLM = get_llm(temperature=0.0)
SMART_LLM = get_llm(model="gpt-4.1-mini", temperature=0.0)
STRUCTURED_JUDGE = SMART_LLM.with_structured_output(FailureAnalysis)
JUDGE_PROMPT_TEMPLATE = PromptTemplate.from_template(CORRECTNESS_PROMPT)
CORRECTNESS_EVALUATOR = JUDGE_PROMPT_TEMPLATE | STRUCTURED_JUDGE


async def run_evals(target_url: str, graph_name: str, dataset_examples: List[DatasetExample]) -> List[ExperimentResultContext]:
    """Run evaluations for a given target URL and graph name."""

    sync_client = get_sync_client(url=target_url)

    #TODO: confusing we are supplying the sync_client and client to the remote graph 
    remote_graph = RemoteGraph(
        graph_name,
        url=target_url,
        client=LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        question = tc.input_message
        expected = tc.expected_output

        # create a fresh thread for the example
        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

        run_start = datetime.now(timezone.utc)
        result = await asyncio.to_thread(
            remote_graph.invoke,
            {"messages": [{"role": "user", "content": question}]},
            thread_cfg,
        )
        answer = result.get("messages", [{}])[-1].get("content", "")
        run_end = datetime.now(timezone.utc)

        eval_result_obj: FailureAnalysis = await asyncio.to_thread(
            CORRECTNESS_EVALUATOR.invoke,
            {
                "inputs": question,
                "outputs": answer,
                "reference_outputs": expected,
                "test_case_intent": tc.reasoning, 
            },
        )
        
        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread["thread_id"],
                actual_output=answer,
                analysis=eval_result_obj,
                started_at=run_start,
                completed_at=run_end,
            )
        )

    logger.info(
        "run.execute: completed %d tests",
        len(dataset_examples),
    )

    return results
