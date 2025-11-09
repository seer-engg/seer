"""shared module for running evaluations"""
import asyncio
import os
import re
from typing import List
from datetime import datetime, timezone

from e2b import AsyncSandbox
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client

from shared.schema import DatasetExample, ExperimentResultContext, FailureAnalysis
from shared.logger import get_logger


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
logger = get_logger("eval_runner")


async def run_evals(
        target_url: str, 
        graph_name: str, 
        dataset_examples: List[DatasetExample],
        user_id: str,
        sandbox_id: str,
    ) -> List[ExperimentResultContext]:
    """Run evaluations for a given target URL and graph name."""

    sync_client = get_sync_client(url=target_url)
    sbx = await AsyncSandbox.connect(sandbox_id)

    remote_graph = RemoteGraph(
        graph_name,
        url=target_url,
        sync_client=sync_client,
    )

    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        question = tc.input_message
        expected = tc.expected_output

        # create a fresh thread for the example
        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

        run_start = datetime.now(timezone.utc)
        try:
            result = await asyncio.to_thread(
                remote_graph.invoke,
                {"messages": [{"role": "user", "content": question}]},
                thread_cfg,
            )
            answer = result.get("messages", [{}])[-1].get("content", "")
            run_end = datetime.now(timezone.utc)

            eval_result_obj: FailureAnalysis

            # 1. Extract code from the agent's response
            code_match = re.search(r"```python\n(.*?)```", answer, re.DOTALL)
            if not code_match:
                eval_result_obj = FailureAnalysis(
                    score=0.0,
                    failure_type="structure_preservation",
                    judge_reasoning="Agent failed to provide a ```python code block in its output."
                )
            else:
                agent_code = code_match.group(1)
                hidden_tests = expected  # 'expected_output' now holds the hidden tests

                # Robustness: Parse hidden_tests in case the LLM (or prompt)
                # still included markdown
                hidden_test_match = re.search(r"```python\n(.*?)```", hidden_tests, re.DOTALL)
                if hidden_test_match:
                    hidden_tests = hidden_test_match.group(1)

                # 2. Write files to sandbox
                await sbx.files.write('solution.py', agent_code)
                await sbx.files.write('test_solution.py', hidden_tests)

                # 3. Run tests
                test_result = await sbx.commands.run('python -m unittest test_solution.py')

                # 4. Create objective analysis
                if test_result.exit_code == 0:
                    eval_result_obj = FailureAnalysis(
                        score=1.0,
                        judge_reasoning="All hidden unit tests passed."
                    )
                else:
                    eval_result_obj = FailureAnalysis(
                        score=0.0,
                        failure_type="logical_error",
                        judge_reasoning=f"Failed hidden unit tests. Traceback:\n{test_result.stderr}"
                    )

        except Exception as e:
            logger.error(f"Error running eval: {e}")
            answer = str(e)
            run_end = datetime.now(timezone.utc)
            eval_result_obj = FailureAnalysis(score=0.0, judge_reasoning="Runtime error in Target agent while running eval")
            
        
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
