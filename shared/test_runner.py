from typing import List
import asyncio
from datetime import datetime, timezone
import re
from e2b import AsyncSandbox
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from shared.schema import DatasetExample, ExperimentResultContext, SandboxContext, GithubContext
from shared.logger import get_logger
from shared.schema import FailureAnalysis


logger = get_logger("test_runner")


async def run_tests(dataset_examples: List[DatasetExample],sandbox_context: SandboxContext, github_context: GithubContext) -> List[ExperimentResultContext]:

    logger.info("Connecting to sandbox and remote graph for test execution...")
    sbx = await AsyncSandbox.connect(sandbox_context.sandbox_id, timeout=60 * 20)
    sync_client = get_sync_client(url=sandbox_context.deployment_url)

    remote_graph = RemoteGraph(
        github_context.agent_name,
        sync_client=sync_client,
    )


    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        question = tc.input_message
        expected = tc.expected_output

        thread = await asyncio.to_thread(sync_client.threads.create)
        thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

        run_start = datetime.now(timezone.utc)
        
        # --- NEW: Define variable outside the try block ---
        agent_actual_output = "" 

        try:
            # 1. Get agent response
            result = await asyncio.to_thread(
                remote_graph.invoke,
                {"messages": [{"role": "user", "content": question}]},
                thread_cfg,
            )
            answer = result.get("messages", [{}])[-1].get("content", "")
            run_end = datetime.now(timezone.utc)
            
            # --- NEW: Secure the agent's output immediately ---
            agent_actual_output = answer

            eval_result_obj: FailureAnalysis

            # 2. Extract agent code
            code_match = re.search(r"```python\n(.*?)```", answer, re.DOTALL)
            if not code_match:
                eval_result_obj = FailureAnalysis(
                    score=0.0,
                    failure_type="structure_preservation",
                    judge_reasoning="Agent failed to provide a ```python code block in its output.",
                )
            else:
                agent_code = code_match.group(1)
                hidden_tests = expected

                hidden_test_match = re.search(
                    r"```python\n(.*?)```", hidden_tests, re.DOTALL
                )
                if hidden_test_match:
                    hidden_tests = hidden_test_match.group(1)

                await sbx.files.write("solution.py", agent_code)
                await sbx.files.write("test_solution.py", hidden_tests)

                # --- NEW: Nested try/except for the test run ---
                try:
                    # 3. Run tests. This will raise CommandExitException on failure.
                    await sbx.commands.run("python -m unittest test_solution.py")
                    
                    # If it didn't raise, it passed.
                    eval_result_obj = FailureAnalysis(
                        score=1.0, 
                        judge_reasoning="All hidden unit tests passed."
                    )
                except Exception as test_failure:
                    # This is the *expected* failure path.
                    # The exception string *is* the traceback.
                    eval_result_obj = FailureAnalysis(
                        score=0.0,
                        failure_type="logical_error",
                        judge_reasoning=f"Failed hidden unit tests. Traceback:\n{str(test_failure)}",
                    )
                # --- End nested try/except ---
            
        except Exception as e:
            # This is the *outer* block. It catches agent invocation failures.
            logger.error(f"Error during agent invocation or setup: {e}")
            run_end = datetime.now(timezone.utc)
            
            # If agent_actual_output is still empty, the invocation itself failed.
            if not agent_actual_output:
                agent_actual_output = f"Agent invocation failed: {str(e)}"
            
            eval_result_obj = FailureAnalysis(
                score=0.0,
                judge_reasoning=f"Runtime error in Target agent or Eval Agent: {e}",
            )

        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread["thread_id"],
                actual_output=agent_actual_output, # <-- This is now safe
                analysis=eval_result_obj,         # <-- This now has the correct reasoning
                started_at=run_start,
                completed_at=run_end,
            )
        )
    
    return results
