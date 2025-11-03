"""nodes for running and uploading evaluation results"""
import asyncio
import json
import os
from datetime import datetime
import requests

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph_sdk import get_sync_client

from shared.eval_runner import run_evals
from shared.logger import get_logger
from agents.eval_agent.models import EvalAgentState, RunContext

logger = get_logger("eval_agent.run")


async def _prepare_run_context(state: EvalAgentState) -> dict:
    run_ctx = state.run or RunContext()

    dataset_name = run_ctx.dataset_name
    if not dataset_name:
        date_tag = datetime.now().strftime("%Y%m%d-%H%M")
        dataset_name = f"seer_eval_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    # always use a new experiment name, else langsmith will raise a 409 while uploading results
    experiment_name = f"seer-eval-local-{timestamp}"

    sync_client = get_sync_client(url=state.sandbox_context.deployment_url)
    thread = await asyncio.to_thread(sync_client.threads.create)
    thread_id = thread["thread_id"]

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
    }
    updated_run = run_ctx.model_copy(update=run_updates)

    return {
        "run": updated_run,
    }


async def _execute_test_cases(state: EvalAgentState) -> dict:
    """Execute the test cases and return the results."""
    results_payload, failed_cases, scores, passed_count, total_tests, earliest_start, latest_end = await run_evals(state.sandbox_context.deployment_url, state.github_context.agent_name, state.test_cases)

    logger.info(
        "run.execute: completed %d tests (failures=%d)",
        total_tests,
        len(failed_cases),
    )

    state.run.score = sum(scores) / len(scores)
    state.run.score_history.append(state.run.score)
    state.run.last_results = results_payload
    state.run.last_failed_cases = failed_cases
    state.run.experiment_end_time = latest_end
    state.run.experiment_start_time = earliest_start

    return {
        "run": state.run,
    }


async def _upload_run_results(state: EvalAgentState) -> dict:
    run_ctx = state.run or RunContext()
    results_payload = list(run_ctx.last_results or [])
    failed_cases = list(run_ctx.last_failed_cases or [])

    score_history = list(run_ctx.score_history or [])

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
        "experiment_start_time": run_ctx.experiment_start_time.isoformat(),
        "experiment_end_time": run_ctx.experiment_end_time.isoformat(),
        "summary_experiment_scores": [
            {
                "key": "mean_correctness",
                "score": round(run_ctx.score, 3),
                "comment": "Average correctness score across generated tests.",
            },
            {
                "key": "aggregate_correctness",
                "score": round(sum(run_ctx.score_history) / len(run_ctx.score_history), 3),
                "comment": "Rolling average correctness score across eval attempts.",
            },
        ],
        "results": serialised_results,
    }

    logger.info("run.upload: uploading experiment to LangSmith: %s", upload_body)

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

    run_updates = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "score": run_ctx.score,
        "score_history": score_history,
        "last_results": results_payload,
        "last_failed_cases": failed_cases,
    }
    updated_run = run_ctx.model_copy(update=run_updates)

    tool_payload = {
        "dataset_name": uploaded_dataset_name,
        "experiment_name": uploaded_experiment_name,
        "latest_score": run_ctx.score,
        "aggregate_score": sum(run_ctx.score_history) / len(run_ctx.score_history),
        "failed_tests": len(failed_cases),
        "codex_handoff_status": "pending",
    }

    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    return {
        "run": updated_run,
        "messages": [tool_message],
        "codex_request": None,
        "codex_response": None,
    }


def build_run_subgraph():
    """Build the run subgraph."""
    builder = StateGraph(EvalAgentState)
    builder.add_node("prepare", _prepare_run_context)
    builder.add_node("execute", _execute_test_cases)
    builder.add_node("upload", _upload_run_results)

    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "execute")
    builder.add_edge("execute", "upload")
    builder.add_edge("upload", END)

    return builder.compile()


