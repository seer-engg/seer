"""nodes for running and uploading evaluation results"""
import asyncio
import json
import os
from datetime import datetime
from typing import List

from langchain_core.messages import ToolMessage

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import ExperimentContext, ExperimentResultContext
from shared.config import config

logger = get_logger("eval_agent.run")

async def prepare_run_context(state: EvalAgentState) -> dict:
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

async def _upload_to_mlflow(
    dataset_name: str,
    experiment_name: str,
    results_payload: List[ExperimentResultContext],
    mean_score: float,
) -> None:
    """Upload experiment results to MLflow."""
    import mlflow
    
    logger.info("run.upload: uploading experiment to MLflow: dataset=%s experiment=%s", 
                dataset_name, experiment_name)
    
    # Set tracking URI from config
    if config.mlflow_tracking_uri:
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    
    # Set or create experiment
    mlflow_experiment_name = config.mlflow_experiment_name or f"seer-eval-{dataset_name}"
    mlflow.set_experiment(mlflow_experiment_name)
    
    # Start a run for this experiment
    with mlflow.start_run(run_name=experiment_name):
        # Log experiment parameters
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("experiment_name", experiment_name)
        mlflow.log_param("num_test_cases", len(results_payload))
        
        # Log aggregate metrics
        mlflow.log_metric("mean_score", mean_score)
        mlflow.log_metric("total_cases", len(results_payload))
        
        passed_cases = sum(1 for res in results_payload if res.score >= config.eval_pass_threshold)
        failed_cases = len(results_payload) - passed_cases
        mlflow.log_metric("passed_cases", passed_cases)
        mlflow.log_metric("failed_cases", failed_cases)
        mlflow.log_metric("pass_rate", passed_cases / len(results_payload) if results_payload else 0)
        
        # Log individual test case results as a JSON artifact
        test_results = []
        for i, res in enumerate(results_payload):
            test_result = {
                "example_id": res.dataset_example.example_id,
                "input": res.dataset_example.input_message,
                "expected_output": res.dataset_example.expected_output.model_dump(),
                "score": res.score,
                "passed": res.score >= config.eval_pass_threshold,
                "started_at": res.started_at.isoformat() if res.started_at else None,
                "completed_at": res.completed_at.isoformat() if res.completed_at else None,
            }
            test_results.append(test_result)
            
            # Log individual scores as metrics
            mlflow.log_metric(f"score_case_{i}", res.score)
        
        # Log the full results as a JSON artifact
        results_json = json.dumps(test_results, indent=2)
        mlflow.log_text(results_json, "eval_results.json")
        
    logger.info("run.upload: Successfully uploaded experiment results to MLflow")


async def upload_run_results(state: EvalAgentState) -> dict:
    """Upload evaluation results to MLflow."""
    experiment = state.active_experiment
    dataset = state.dataset_context

    if not experiment or not dataset:
        raise RuntimeError("Cannot upload without dataset and experiment context")

    # Prefer latest_results if present; otherwise fall back to experiment.results
    results_payload = list(state.latest_results) if state.latest_results else list(experiment.results)
    failed_cases = list(experiment.failed_results)

    # Compute experiment aggregates from results if missing or outdated
    if results_payload:
        experiment.results = results_payload
        experiment.started_at = min(res.started_at for res in results_payload)
        experiment.completed_at = max(res.completed_at for res in results_payload)
        experiment.mean_score = round(sum(res.score for res in results_payload) / len(results_payload), 5)
    mean_score = experiment.mean_score

    # Upload to MLflow if configured
    if config.is_mlflow_tracing_enabled:
        try:
            await _upload_to_mlflow(
                dataset.dataset_name,
                experiment.experiment_name,
                results_payload,
                mean_score,
            )
        except Exception as e:
            logger.error("run.upload: Failed to upload to MLflow: %s", str(e))
            raise RuntimeError(f"MLflow upload failed: {str(e)}")
    else:
        logger.warning("run.upload: MLflow not configured, skipping upload")
        return {
            "dataset_context": dataset,
            "active_experiment": experiment,
            "latest_results": results_payload,
        }

    tool_payload = {
        "dataset_name": dataset.dataset_name,
        "experiment_name": experiment.experiment_name,
        "latest_score": mean_score,
        "failed_tests": len(failed_cases),
        "codex_handoff_status": "pending",
        "tracing_provider": "mlflow",
    }

    tool_message = ToolMessage(content=json.dumps(tool_payload), tool_call_id="run_node")

    return {
        "dataset_context": dataset,
        "active_experiment": experiment,
        "latest_results": results_payload,
        "messages": [tool_message],
    }


