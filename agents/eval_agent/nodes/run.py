"""nodes for running and uploading evaluation results"""
import asyncio
import json
import os
import requests
from datetime import datetime

from langchain_core.messages import ToolMessage

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import ExperimentContext
from shared.config import config

logger = get_logger("eval_agent.run")

async def _prepare_run_context(state: EvalAgentState) -> dict:
    dataset = state.dataset_context

    if not dataset.dataset_name:
        date_tag = datetime.now().strftime("%Y%m%d-%H%M")
        dataset.dataset_id = dataset.dataset_id or f"seer-dataset-{date_tag}"
        dataset.dataset_name = f"seer_eval_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = ExperimentContext(
        experiment_name=f"seer-eval-local-{timestamp}",
        started_at=datetime.utcnow(),
    )
    dataset.experiments.append(experiment)

    logger.info(
        "run.prepare: dataset=%s experiment=%s",
        dataset.dataset_name,
        experiment.experiment_name,
    )

    return {
        "dataset_context": dataset,
        "active_experiment": experiment,
        "latest_results": [],
    }

async def _upload_run_results(state: EvalAgentState) -> dict:
    """used in eval_agent.run node to upload results to LangSmith"""
    experiment = state.active_experiment
    dataset = state.dataset_context

    if not experiment or not dataset:
        raise RuntimeError("Cannot upload without dataset and experiment context")

    # Prefer latest_results if present; otherwise fall back to experiment.results
    results_payload = list(state.latest_results) if state.latest_results else list(experiment.results)
    failed_cases = list(experiment.failed_results)

    api_key = config.langsmith_api_key
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

    endpoint_base = config.langsmith_api_url
    endpoint = f"{endpoint_base.rstrip('/')}/api/v1/datasets/upload-experiment"

    serialised_results = [
        {
            "row_id": res.dataset_example.example_id,
            "thread_id": res.dataset_example.example_id,
            "inputs": {"question": res.dataset_example.input_message},
            "expected_outputs": {"answer": json.dumps(res.dataset_example.expected_output.model_dump())},
            "actual_outputs": {"answer": res.actual_output},
            "evaluation_scores": [
                {
                    "key": "correctness",
                    "score": res.score,
                    "comment": res.judge_reasoning,
                }
            ],
            "start_time": res.started_at.isoformat(),
            "end_time": res.completed_at.isoformat(),
            "run_name": state.context.agent_name if state.context.agent_name else "",
            "run_metadata": {"passed": res.passed},
        }
        for res in results_payload
    ]

    # Compute experiment aggregates from results if missing or outdated
    if results_payload:
        experiment.results = results_payload
        experiment.started_at = min(res.started_at for res in results_payload)
        experiment.completed_at = max(res.completed_at for res in results_payload)
        experiment.mean_score = round(sum(res.score for res in results_payload) / len(results_payload), 5)
    mean_score = experiment.mean_score

    upload_body = {
        "experiment_name": experiment.experiment_name,
        "experiment_description": "Evaluation uploaded by Seer eval_agent run_node.",
        "dataset_name": dataset.dataset_name,
        "experiment_start_time": (experiment.started_at or datetime.utcnow()).isoformat(),
        "experiment_end_time": (experiment.completed_at or datetime.utcnow()).isoformat(),
        "summary_experiment_scores": [
            {
                "key": "correctness",
                "score": round(mean_score, 3),
                "comment": "Average correctness score across generated tests.",
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
        dataset.dataset_name = response_data.get("dataset", {}).get("name", dataset.dataset_name)
        experiment.experiment_name = response_data.get("experiment", {}).get("name", experiment.experiment_name)

    tool_payload = {
        "dataset_name": dataset.dataset_name,
        "experiment_name": experiment.experiment_name,
        "latest_score": mean_score,
        "failed_tests": len(failed_cases),
        "codex_handoff_status": "pending",
    }

    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    return {
        "dataset_context": dataset,
        "active_experiment": experiment,
        "latest_results": results_payload,
        "messages": [tool_message],
    }


