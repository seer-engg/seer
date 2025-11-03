from langgraph_sdk import get_sync_client

from langgraph.pregel.remote import RemoteGraph
import os
from langsmith import Client
from shared.schema import GeneratedTestCase
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
from typing import List, Dict, Any
from datetime import datetime, timezone
import uuid
from shared.logger import get_logger
from openevals.types import EvaluatorResult
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.llm import create_llm_as_judge
import asyncio
from shared.llm import get_llm


logger = get_logger("eval_runner")


LLM = get_llm(temperature=0.2)
CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)

PASS_THRESHOLD = 0.99


async def run_evals(target_url: str, graph_name: str, test_cases: List[GeneratedTestCase]) -> dict:
    sync_client = get_sync_client(url=target_url)
    if not thread_id:
        thread = sync_client.threads.create()
        thread_id = thread["thread_id"]
    
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
        try:
            result = await asyncio.to_thread(
                remote_graph.invoke,
                {"messages": [{"role": "user", "content": question}]},
                thread_cfg,
            )
            answer = result.get("messages", [{}])[-1].get("content", "")
        except Exception as invoke_error:
            logger.error("run.execute: error invoking remote graph: %s", invoke_error)
            answer = ""
        run_end = datetime.now(timezone.utc)

        try:
            eval_result: EvaluatorResult = await asyncio.to_thread(
                CORRECTNESS_EVALUATOR,
                inputs={"question": question},
                outputs={"answer": answer},
                reference_outputs={"answer": expected},
            )
            score = float(eval_result.get("score", 0.0))
            evaluator_comment = eval_result.get("comment", "")
        except Exception as eval_error:
            logger.error("run.execute: error running correctness evaluator: %s", eval_error)
            score = 0.0
            evaluator_comment = f"Evaluation error: {eval_error}"

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
