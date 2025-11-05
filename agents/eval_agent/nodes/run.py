"""nodes for running and uploading evaluation results"""
import asyncio
import json
import os
from typing import List
from datetime import datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from e2b import AsyncSandbox

from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from agents.eval_agent.models import EvalAgentState
from agents.eval_agent.constants import NEO4J_GRAPH
from shared.eval_runner import run_evals
from shared.logger import get_logger
from shared.schema import DatasetContext, ExperimentContext, ExperimentResultContext

logger = get_logger("eval_agent.run")


async def _prepare_run_context(state: EvalAgentState) -> dict:
    dataset = state.dataset_context or DatasetContext()

    if not dataset.dataset_name:
        date_tag = datetime.now().strftime("%Y%m%d-%H%M")
        dataset.dataset_id = dataset.dataset_id or f"seer-dataset-{date_tag}"
        dataset.dataset_name = f"seer_eval_{date_tag}"

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment = ExperimentContext(
        experiment_id=f"seer-exp-{timestamp}",
        experiment_name=f"seer-eval-local-{timestamp}",
        attempt_index=len(dataset.experiments) + 1,
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


async def _execute_test_cases(state: EvalAgentState) -> dict:
    """Execute the test cases and return the results."""

    if not state.user_context or not state.user_context.user_id:
        raise ValueError("UserContext with user_id is required to log memories")
    user_id = state.user_context.user_id
    if not state.sandbox_context or not state.github_context:
        raise RuntimeError("Sandbox and GitHub context must be set before executing tests")
    if not state.active_experiment:
        raise RuntimeError("Active experiment missing before executing tests")

    # connect to sandbox before running evals
    await AsyncSandbox.connect(state.sandbox_context.sandbox_id, timeout=60*20) # 20 minutes

    results: List[ExperimentResultContext] = await run_evals(
        state.sandbox_context.deployment_url,
        state.github_context.agent_name,
        state.dataset_examples,
    )

    # --- NEW: LOG FACTUAL MEMORIES TO NEO4J ---
    # We log the graph data asynchronously in the background
    # 1. Create a list of parameters for each test case and its result
   # --- MODIFIED: Update Cypher query to store structured analysis ---
    cypher_params = []
    for res in results:
        # Flatten the analysis object for storage as node properties
        analysis_props = res.analysis.model_dump()
        
        # Combine all properties for the result node
        result_node_props = res.model_dump(
            exclude={'dataset_example', 'analysis', 'passed'}
        )
        result_node_props.update(analysis_props) # Add all analysis fields
        result_node_props['passed'] = res.passed  # Add the computed 'passed' bool
        
        cypher_params.append({
            "example": res.dataset_example.model_dump(),
            "result": result_node_props,
        })
    
    # 2. Define the Cypher query to create the graph structure
    cypher_query = """
    UNWIND $params as row
    
    // Merge the TestCase node, now namespaced by user_id
    MERGE (ex:DatasetExample {example_id: row.example.example_id, user_id: $user_id})
    ON CREATE SET ex += row.example
    ON MATCH SET ex += row.example
    
    // Merge the Result node, now namespaced by user_id
    MERGE (res:ExperimentResult {thread_id: row.result.thread_id, user_id: $user_id})
    
    // Use SET to overwrite all properties, ensuring schema stays current
    SET res += row.result
    
    // Connect the TestCase to its Result
    MERGE (ex)-[r:WAS_RUN_IN]->(res)
    
    RETURN count(*)
    """
    
    # 3. Run the query and capture the response
    query_result = await asyncio.to_thread(
        NEO4J_GRAPH.query,
        cypher_query,
        params={"params": cypher_params, "user_id": user_id}
    )
    
    logger.info(f"run.execute: Neo4j query response: {query_result}")

    experiment = state.active_experiment
    experiment.results.extend(results)

    if results:
        experiment.started_at = min(res.started_at for res in results)
        experiment.completed_at = max(res.completed_at for res in results)
        experiment.mean_score = sum(res.score for res in results) / len(results)

    logger.info(
        "run.execute: completed %d tests (failures=%d)",
        len(results),
        len(experiment.failed_results),
    )

    return {
        "active_experiment": experiment,
        "dataset_context": state.dataset_context,
        "latest_results": results,
    }


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
async def _upload_run_results(state: EvalAgentState) -> dict:
    experiment = state.active_experiment
    dataset = state.dataset_context

    if not experiment or not dataset:
        raise RuntimeError("Cannot upload without dataset and experiment context")

    results_payload = list(experiment.results)
    failed_cases = list(experiment.failed_results)

    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise ValueError("LANGSMITH_API_KEY environment variable is required for experiment upload.")

    endpoint_base = os.getenv("LANGSMITH_API_URL", "https://api.smith.langchain.com")
    endpoint = f"{endpoint_base.rstrip('/')}/api/v1/datasets/upload-experiment"

    serialised_results = [
        {
            "row_id": res.dataset_example.example_id,
            "thread_id": res.dataset_example.example_id,
            "inputs": {"question": res.dataset_example.input_message},
            "expected_outputs": {"answer": res.dataset_example.expected_output},
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
            "run_name": state.github_context.agent_name if state.github_context else "",
            "run_metadata": {"passed": res.passed},
        }
        for res in results_payload
    ]

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


