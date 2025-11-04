"""shared module for running evaluations"""
import asyncio
import os
from typing import List
from datetime import datetime, timezone

from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client
from openevals.types import EvaluatorResult
from openevals.llm import create_llm_as_judge

from shared.schema import DatasetExample, ExperimentResultContext
from shared.logger import get_logger
from shared.llm import get_llm


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
logger = get_logger("eval_runner")


CORRECTNESS_PROMPT = """You are an expert data labeler evaluating the correctness of a coding agent's output. 
The coding agent's job is to fix broken Python code snippets while preserving the original code structure.

Here is the original broken Python code that was given to the agent:
<input>
{inputs}
</input>

Here is the agent's attempt to fix the code:
<output>
{outputs}
</output>

Use the reference outputs below to help you evaluate the correctness of the response:
<reference_outputs>
{reference_outputs}
</reference_outputs>

Your task is to evaluate the agent's output and assign a correctness score from 0 to 1.

**Scoring Rubric:**

**1 = Completely correct, accurate, and complete**
- Fixes all issues in the original code
- Contains no factual errors or bugs
- Addresses all parts of the problem
- Maintains exact original structure (function names, class names, schemas, invocation patterns)
- Uses precise and accurate Python syntax

**0.8 = Mostly correct with minor issues**  
- Fixes the main issues but may have small problems
- Structure preservation is good with minimal deviations
- Minor syntax or logic issues that don't break functionality

**0.6 = Partially correct but with notable problems**
- Fixes some issues but misses others
- May have moderate structural changes or naming violations
- Contains logical errors or incomplete solutions

**0.4 = Largely incorrect with some correct elements**
- Major structural changes or renaming of original components
- Significant logical errors or missing functionality
- Some correct elements present

**0.2 = Mostly or completely incorrect**
- Fails to fix the original issues
- Major violations of structure preservation requirements
- Contains serious errors or completely wrong approach

**Critical Structure Preservation Requirements:**
- Original function names, class names, and schemas MUST remain unchanged
- Original invocation patterns MUST be preserved
- You may create new internal variables or helper functions, but cannot rename existing components
- The overall framework and interface must stay intact

**Evaluation Process:**
1. Identify the specific issues/problems in the original broken code (list each one)
2. For each identified issue, check whether the agent's output successfully resolves it
3. Verify structure preservation by listing the original function names, class names, and key components, then checking if they remain unchanged in the agent's output
4. Compare the agent's approach and solution to the reference outputs
5. Note any additional errors or problems introduced by the agent's fix

It's OK for this section to be quite long.

Then provide your final assessment in the following format:

`<justification>`
[Write a detailed explanation of your evaluation, covering technical correctness, completeness, and structure preservation. Explain your reasoning for the score.]
`</justification>`

`<score>`
[Your score from 0 to 1]
`</score>`
"""

LLM = get_llm(temperature=0.0)
CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="score",
    continuous=True,
)

PASS_THRESHOLD = 0.8


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

        eval_result: EvaluatorResult = await asyncio.to_thread(
            CORRECTNESS_EVALUATOR,
            inputs={"question": question},
            outputs={"answer": answer},
            reference_outputs={"answer": expected},
        )
        score = float(eval_result.get("score", 0))
        evaluator_comment = eval_result.get("comment", "")
        passed = score >= PASS_THRESHOLD

        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread["thread_id"],
                actual_output=answer,
                score=score,
                passed=passed,
                judge_reasoning=evaluator_comment,
                started_at=run_start,
                completed_at=run_end,
            )
        )

    logger.info(
        "run.execute: completed %d tests",
        len(dataset_examples),
    )

    return results
