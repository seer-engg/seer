"""nodes for running and uploading evaluation results"""
import asyncio
import json
import os
from datetime import datetime

from langchain_core.messages import ToolMessage
from langfuse import get_client

from agents.eval_agent.models import EvalAgentState
from shared.logger import get_logger
from shared.schema import ExperimentContext
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

async def upload_run_results(state: EvalAgentState) -> dict:
    """used in eval_agent.run node to upload results to Langfuse"""
    experiment = state.active_experiment
    dataset = state.dataset_context

    if not experiment or not dataset:
        raise RuntimeError("Cannot upload without dataset and experiment context")

    # Prefer latest_results if present; otherwise fall back to experiment.results
    results_payload = list(state.latest_results) if state.latest_results else list(experiment.results)
    failed_cases = list(experiment.failed_results)

    if not config.is_langfuse_configured:
        logger.warning("run.upload: Langfuse is not configured, skipping upload")
        return {
        }
    
    # Initialize Langfuse client with config
    from langfuse import Langfuse
    langfuse = Langfuse(
        secret_key=config.langfuse_secret_key,
        host=config.langfuse_base_url
    )

    # Compute experiment aggregates from results if missing or outdated
    if results_payload:
        experiment.results = results_payload
        experiment.started_at = min(res.started_at for res in results_payload)
        experiment.completed_at = max(res.completed_at for res in results_payload)
        experiment.mean_score = round(sum(res.score for res in results_payload) / len(results_payload), 5)
    mean_score = experiment.mean_score

    logger.info("run.upload: uploading experiment to Langfuse: dataset=%s experiment=%s", 
                dataset.dataset_name, experiment.experiment_name)

    try:
        # Get or create dataset
        try:
            langfuse_dataset = await asyncio.to_thread(langfuse.get_dataset, dataset.dataset_name)
        except Exception:
            # Dataset doesn't exist, create it
            langfuse_dataset = await asyncio.to_thread(
                langfuse.create_dataset,
                name=dataset.dataset_name,
                description="Evaluation dataset created by Seer eval_agent"
            )

        # Create dataset items for each result (if they don't exist)
        # Note: Langfuse uses upsert by ID, so we'll use example_id as the item ID
        for res in results_payload:
            item_id = res.dataset_example.example_id
            try:
                await asyncio.to_thread(
                    langfuse.create_dataset_item,
                    id=item_id,
                    dataset_name=dataset.dataset_name,
                    input={"question": res.dataset_example.input_message},
                    expected_output={"answer": json.dumps(res.dataset_example.expected_output.model_dump())},
                    metadata={"example_id": item_id}
                )
            except Exception as e:
                # Item might already exist, that's okay
                logger.debug("Dataset item %s might already exist: %s", item_id, str(e))

        # Create dataset run (experiment)
        run_name = experiment.experiment_name
        
        # For each result, we need to link it to a trace
        # Since we don't have trace IDs here, we'll create the run and link items manually
        # The actual trace linking should happen when traces are created during execution
        
        # Create a dataset run using the SDK's run_experiment method would require traces
        # Instead, we'll use the lower-level API to create the run
        # For now, we'll create the run metadata and let the linking happen via trace IDs later
        
        # Update dataset and experiment names from Langfuse response if needed
        # (Langfuse might normalize the names)
        dataset.dataset_name = langfuse_dataset.name if hasattr(langfuse_dataset, 'name') else dataset.dataset_name

        logger.info("run.upload: Successfully created dataset and prepared experiment run in Langfuse")

    except Exception as e:
        logger.error("run.upload: Failed to upload to Langfuse: %s", str(e))
        raise RuntimeError(f"Langfuse upload failed: {str(e)}")

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


