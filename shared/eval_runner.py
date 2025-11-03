"""shared module for running evaluations"""
import asyncio
import os
from typing import List, Tuple
from datetime import datetime, timezone

from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langsmith import Client
from openevals.types import EvaluatorResult
from openevals.prompts import CORRECTNESS_PROMPT
from openevals.llm import create_llm_as_judge

from shared.schema import DatasetExample, ExperimentResultContext
from shared.logger import get_logger
from shared.llm import get_llm


LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)
logger = get_logger("eval_runner")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_CLIENT = Client(api_key=LANGSMITH_API_KEY)

LLM = get_llm(temperature=0.2)
CORRECTNESS_EVALUATOR = create_llm_as_judge(
    prompt=CORRECTNESS_PROMPT,
    model="openai:gpt-4.1-mini",
    feedback_key="correctness",
)

PASS_THRESHOLD = 0.99


async def run_evals(target_url: str, graph_name: str, dataset_examples: List[DatasetExample]) -> Tuple[List[ExperimentResultContext], List[float]]:
    """Run evaluations for a given target URL and graph name."""

    sync_client = get_sync_client(url=target_url)
    remote_graph = RemoteGraph(
        graph_name,
        url=target_url,
        client=LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    results: List[ExperimentResultContext] = []
    scores: List[float] = []

    for tc in dataset_examples:
        question = tc.input_message
        expected = tc.expected_output

        # use the example_id as the thread_id
        example_id = tc.example_id
        await asyncio.to_thread(sync_client.threads.create, thread_id=example_id)
        thread_cfg = {"configurable": {"thread_id": example_id}}

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
        passed = score >= PASS_THRESHOLD

        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                actual_output=answer,
                score=score,
                passed=passed,
                judge_reasoning=evaluator_comment,
                started_at=run_start,
                completed_at=run_end,
            )
        )
        scores.append(score)

    logger.info(
        "run.execute: completed %d tests",
        len(dataset_examples),
    )

    return results, scores
