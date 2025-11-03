"""shared module for running evaluations"""
import asyncio
import os
import uuid
from typing import List, Dict, Any
from datetime import datetime, timezone

from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client
from openevals.types import EvaluatorResult
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.llm import create_llm_as_judge

from shared.schema import GeneratedTestCase
from shared.logger import get_logger
from shared.llm import get_llm


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
logger = get_logger("eval_runner")

LLM = get_llm(temperature=0.2)
CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)

PASS_THRESHOLD = 0.99


async def run_evals(target_url: str, graph_name: str, test_cases: List[GeneratedTestCase]) -> dict:
    """Run evaluations for a given target URL and graph name."""
    sync_client = get_sync_client(url=target_url)

    # always create a new thread for the evaluations
    thread = await asyncio.to_thread(sync_client.threads.create)
    thread_id = thread["thread_id"]
    
    # configure the remote graph to use the new thread
    thread_cfg = {"configurable": {"thread_id": thread_id}}
    remote_graph = RemoteGraph(
        graph_name,
        url=target_url,
        client=LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    results_payload: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []
    scores: List[float] = []
    passed_count = 0
    total_tests = len(test_cases)

    experiment_start_time = datetime.now(timezone.utc)
    earliest_start = experiment_start_time
    latest_end = experiment_start_time


    for tc in test_cases:
        question = tc.input_message
        expected = tc.expected_output or tc.expected_behavior
        row_id = uuid.uuid4().hex

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
        score = float(eval_result.get("score", 0.0))
        evaluator_comment = eval_result.get("comment", "")

        scores.append(score)
        if score >= PASS_THRESHOLD:
            passed_count += 1
        else:
            failed_cases.append(
                {
                    "input": question,
                    "expected_output": expected,
                    "actual_output": answer,
                    "success_criteria": tc.success_criteria,
                    "score": score,
                    "judge_comment": evaluator_comment,
                }
            )

        results_payload.append(
            {
                "row_id": row_id,
                "inputs": {"question": question},
                "expected_outputs": {"answer": expected},
                "actual_outputs": {"answer": answer},
                "evaluation_scores": [
                    {
                        "key": "correctness",
                        "score": score,
                        "comment": evaluator_comment,
                    }
                ],
                "start_time": run_start,
                "end_time": run_end,
                "run_name": graph_name,
                "run_metadata": {
                    "success_criteria": tc.success_criteria,
                },
            }
        )

        if run_start < earliest_start:
            earliest_start = run_start
        if run_end > latest_end:
            latest_end = run_end

    logger.info(
        "run.execute: completed %d tests (failures=%d)",
        total_tests,
        len(failed_cases),
    )

    return results_payload, failed_cases, scores, passed_count, total_tests, earliest_start, latest_end
