import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List

import requests
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.pregel.remote import RemoteGraph
from langgraph_sdk import get_sync_client
from openevals.types import EvaluatorResult

from agents.eval_agent.deps import (
    CORRECTNESS_EVALUATOR,
    LANGSMITH_CLIENT,
    PASS_THRESHOLD,
    logger,
)
from agents.eval_agent.models import DeploymentContext, EvalAgentState, RunContext


async def _prepare_run_context(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_prepare_run_context requires target_agent_config to be set")

    target_graph_name = cfg.graph_name
    deployment = state.deployment or DeploymentContext()
    run_ctx = state.run or RunContext()
    target_url = cfg.url or deployment.url
    if not target_url:
        raise ValueError("_prepare_run_context requires a deployment URL")

    dataset_name = run_ctx.dataset_name
    if not dataset_name:
        agent_id = target_graph_name.replace("/", "_")
        date_tag = datetime.now().strftime("%Y%m%d")
        dataset_name = f"seer_eval_{agent_id}_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # always use a new experiment name, else langsmith will raise a 409 while uploading results
    experiment_name = f"seer-local-eval-{target_graph_name}-{timestamp}"

    sync_client = get_sync_client(url=target_url)
    thread = await asyncio.to_thread(sync_client.threads.create)
    thread_id = thread["thread_id"]

    metadata = dict(deployment.metadata)
    metadata.setdefault("deployment_url", target_url)
    metadata.setdefault("repo_url", cfg.repo_url)
    metadata.setdefault("branch_name", cfg.branch_name)

    deployment = deployment.model_copy(
        update={
            "url": target_url,
            "repo_url": cfg.repo_url or deployment.repo_url,
            "branch_name": cfg.branch_name or deployment.branch_name,
            "metadata": metadata,
        }
    )

    logger.info(
        "run.prepare: dataset=%s experiment=%s thread=%s",
        dataset_name,
        experiment_name,
        thread_id,
    )

    run_updates = {
        "dataset_name": dataset_name,
        "experiment_name": experiment_name,
        "current_thread_id": thread_id,
        "last_failed_cases": [],
        "last_results": [],
        "last_metadata": {
            "target_graph_name": target_graph_name,
            "target_url": target_url,
            "experiment_prepared_at": datetime.now(timezone.utc),
        },
    }
    updated_run = run_ctx.model_copy(update=run_updates)

    return {
        "run": updated_run,
        "deployment": deployment,
    }


async def _execute_test_cases(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_execute_test_cases requires target_agent_config to be set")

    deployment = state.deployment or DeploymentContext()
    run_ctx = state.run or RunContext()
    target_url = cfg.url or deployment.url
    if not target_url:
        raise ValueError("_execute_test_cases requires a deployment URL")

    thread_id = run_ctx.current_thread_id
    sync_client = get_sync_client(url=target_url)
    if not thread_id:
        thread = sync_client.threads.create()
        thread_id = thread["thread_id"]

    thread_cfg = {"configurable": {"thread_id": thread_id}}
    remote_graph = RemoteGraph(
        cfg.graph_name,
        url=target_url,
        client=LANGSMITH_CLIENT,
        sync_client=sync_client,
        distributed_tracing=True,
    )

    results_payload: List[Dict[str, Any]] = []
    failed_cases: List[Dict[str, Any]] = []
    scores: List[float] = []
    passed_count = 0
    total_tests = len(state.test_cases)

    experiment_start_time = datetime.now(timezone.utc)
    earliest_start = experiment_start_time
    latest_end = experiment_start_time

    for tc in state.test_cases:
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
                "run_name": cfg.graph_name,
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

    metadata = dict(run_ctx.last_metadata or {})
    metadata.update(
        {
            "scores": scores,
            "passed_count": passed_count,
            "total_tests": total_tests,
            "experiment_start_time": experiment_start_time,
            "earliest_start": earliest_start,
            "latest_end": latest_end,
        }
    )

    accumulated_cases = list(run_ctx.accumulated_failed_cases or [])
    accumulated_context = dict(run_ctx.accumulated_metadata or {})
    if failed_cases:
        accumulated_cases.extend(failed_cases)
        accumulated_context.update(
            {
                "dataset_name": run_ctx.dataset_name,
                "experiment_name": run_ctx.experiment_name,
                "target_url": metadata.get("target_url", deployment.url or cfg.url),
                "latest_score": metadata.get("scores", [0])[-1] if metadata.get("scores") else run_ctx.score,
                "aggregate_score": metadata.get("aggregate_score", run_ctx.score),
                "total_tests": metadata.get("total_tests", total_tests),
                "passed_tests": metadata.get("passed_count", passed_count),
            }
        )

    run_updates = {
        "current_thread_id": thread_id,
        "last_results": results_payload,
        "last_failed_cases": failed_cases,
        "last_metadata": metadata,
        "accumulated_failed_cases": accumulated_cases,
        "accumulated_metadata": accumulated_context,
    }
    updated_run = run_ctx.model_copy(update=run_updates)

    return {
        "run": updated_run,
    }


async def _upload_run_results(state: EvalAgentState) -> dict:
    cfg = state.target_agent_config
    if cfg is None:
        raise ValueError("_upload_run_results requires target_agent_config to be set")

    deployment = state.deployment or DeploymentContext()
    run_ctx = state.run or RunContext()
    results_payload = list(run_ctx.last_results or [])
    metadata = dict(run_ctx.last_metadata or {})
    failed_cases = list(run_ctx.last_failed_cases or [])
    scores: List[float] = list(metadata.get("scores", []))
    passed_count = metadata.get("passed_count", 0)
    total_tests = metadata.get("total_tests", len(state.test_cases))

    mean_score = round(sum(scores) / max(len(scores), 1), 4)
    earliest_start = metadata.get("earliest_start", datetime.now(timezone.utc))
    latest_end = metadata.get("latest_end", earliest_start)
    experiment_end_time = max(latest_end, datetime.now(timezone.utc))

    score_history = list(run_ctx.score_history or [])
    score_history.append(float(mean_score))
    aggregate_score = round(sum(score_history) / len(score_history), 4)

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

    endpoint_base = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    endpoint = f"{endpoint_base.rstrip('/')}/api/v1/datasets/upload-experiment"

    serialised_results = []
    for row in results_payload:
        serialised_results.append(
            {
                **row,
                "start_time": row["start_time"].isoformat(),
                "end_time": row["end_time"].isoformat(),
            }
        )

    upload_body = {
        "experiment_name": run_ctx.experiment_name,
        "experiment_description": "Evaluation uploaded by Seer eval_agent run_node.",
        "dataset_name": run_ctx.dataset_name,
        "experiment_start_time": earliest_start.isoformat(),
        "experiment_end_time": experiment_end_time.isoformat(),
        "summary_experiment_scores": [
            {
                "key": "mean_correctness",
                "score": mean_score,
                "comment": "Average correctness score across generated tests.",
            },
            {
                "key": "aggregate_correctness",
                "score": aggregate_score,
                "comment": "Rolling average correctness score across eval attempts.",
            },
        ],
        "results": serialised_results,
    }

    logger.info(f"Uploading experiment to LangSmith: {upload_body}")

    response = await asyncio.to_thread(
        requests.post,
        endpoint,
        json=upload_body,
        headers={"x-api-key": api_key},
        timeout=300,
    )
    if not response.ok:
        raise RuntimeError(
            f"LangSmith upload failed with status {response.status_code}: {response.text}"
        )
    else:
        response_data = response.json()
        uploaded_dataset_name = response_data.get("dataset", {}).get("name", run_ctx.dataset_name)
        uploaded_experiment_name = response_data.get("experiment", {}).get("name", run_ctx.experiment_name)

    metadata.update(
        {
            "dataset_name": uploaded_dataset_name,
            "experiment_name": uploaded_experiment_name,
            "latest_score": mean_score,
            "aggregate_score": aggregate_score,
            "failed_tests": len(failed_cases),
            "passed_tests": passed_count,
            "total_tests": total_tests,
            "handoff_ready_at": datetime.now(timezone.utc),
        }
    )

    target_url = metadata.get("target_url", deployment.url or cfg.url)
    metadata.setdefault("target_url", target_url)

    deployment_metadata = dict(deployment.metadata)
    deployment_metadata.update(
        {
            "deployment_url": target_url,
            "repo_url": cfg.repo_url,
            "branch_name": cfg.branch_name,
            "last_upload_at": datetime.now(timezone.utc).isoformat(),
            "codex_handoff_status": "pending",
        }
    )

    deployment = deployment.model_copy(
        update={
            "url": target_url,
            "repo_url": cfg.repo_url or deployment.repo_url,
            "branch_name": cfg.branch_name or deployment.branch_name,
            "metadata": deployment_metadata,
        }
    )

    accumulated_metadata = dict(run_ctx.accumulated_metadata or {})
    accumulated_metadata.update(
        {
            "dataset_name": uploaded_dataset_name,
            "experiment_name": uploaded_experiment_name,
            "latest_score": mean_score,
            "aggregate_score": aggregate_score,
            "total_tests": total_tests,
            "passed_tests": passed_count,
            "target_url": target_url,
        }
    )

    run_updates = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "score": float(mean_score),
        "score_history": score_history,
        "last_results": results_payload,
        "last_metadata": metadata,
        "last_failed_cases": failed_cases,
        "accumulated_metadata": accumulated_metadata,
    }
    updated_run = run_ctx.model_copy(update=run_updates)

    tool_payload = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "latest_score": mean_score,
        "aggregate_score": aggregate_score,
        "failed_tests": len(failed_cases),
        "passed_tests": passed_count,
        "deployment_url": target_url,
        "codex_handoff_status": "pending",
    }

    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    return {
        "run": updated_run,
        "messages": [tool_message],
        "deployment": deployment,
        "codex_request": None,
        "codex_response": None,
    }


def build_run_subgraph():
    builder = StateGraph(EvalAgentState)
    builder.add_node("prepare", _prepare_run_context)
    builder.add_node("execute", _execute_test_cases)
    builder.add_node("upload", _upload_run_results)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "execute")
    builder.add_edge("execute", "upload")
    builder.add_edge("upload", END)

    return builder.compile()


