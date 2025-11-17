"""nodes for running and uploading evaluation results"""
import re
import asyncio
import json
import os
import requests
from typing import List
from datetime import datetime, timezone

from e2b import AsyncSandbox
from langchain_core.messages import ToolMessage
from langgraph.graph import END, START, StateGraph

from langgraph.pregel.remote import RemoteGraph
from agents.eval_agent.models import EvalAgentState, TestExecutionState
from langgraph_sdk import get_sync_client
from shared.logger import get_logger
from shared.schema import ExperimentContext, ExperimentResultContext, FailureAnalysis
from agents.eval_agent.nodes.execute import build_test_execution_subgraph
from graph_db import NEO4J_GRAPH

logger = get_logger("eval_agent.run")

async def _prepare_run_context(state: EvalAgentState) -> dict:
    dataset = state.dataset_context

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

    if not state.context.user_context or not state.context.user_context.user_id:
        raise ValueError("UserContext with user_id is required to log memories")
    user_id = state.context.user_context.user_id

    if not state.context.github_context or not state.context.github_context.agent_name:
        raise ValueError("GithubContext with agent_name is required to log memories")
    agent_name = state.context.github_context.agent_name

    if not state.context.sandbox_context:
        raise RuntimeError("Sandbox context must be set before executing tests")
    if not state.active_experiment:
        raise RuntimeError("Active experiment missing before executing tests")

    # Enrich mcp_resources with github_owner and github_repo for test execution
    # This ensures tests can reference [resource:github_owner] and [resource:github_repo]
    enriched_resources = dict(state.context.mcp_resources or {})
    
    if state.context.github_context and state.context.github_context.repo_url:
        from shared.parameter_population import extract_all_context_variables
        
        # Extract context variables including github_owner and github_repo
        context_vars = extract_all_context_variables(
            user_context=state.context.user_context,
            github_context=state.context.github_context,
            mcp_resources=enriched_resources,
        )
        
        # Add github_owner and github_repo as resources if they were extracted
        if 'github_owner' in context_vars:
            enriched_resources['github_owner'] = {'id': context_vars['github_owner']}
            logger.info(f"Added github_owner to mcp_resources: {context_vars['github_owner']}")
        
        if 'github_repo' in context_vars:
            enriched_resources['github_repo'] = {'id': context_vars['github_repo']}
            logger.info(f"Added github_repo to mcp_resources: {context_vars['github_repo']}")

    # Build test execution subgraph (provision → invoke → assert)
    test_graph = build_test_execution_subgraph()
    results: List[ExperimentResultContext] = []

    # Execute each dataset example through the execution subgraph
    for idx, example in enumerate(state.dataset_examples, start=1):
        logger.info(f"Executing test {idx}/{len(state.dataset_examples)}: {example.example_id}")
        initial = TestExecutionState(
            context=state.context,
            dataset_example=example,
            mcp_resources=dict(enriched_resources),
            tool_selection_log=state.tool_selection_log,
        )
        final_state = await test_graph.ainvoke(initial)
        # Support both dict and pydantic object returns
        result_ctx = (
            final_state.get("result") if isinstance(final_state, dict) else getattr(final_state, "result", None)
        )
        if not result_ctx:
            raise RuntimeError(f"Test execution subgraph did not produce a result for {example.example_id}")
        results.append(result_ctx)
    
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

        example_node_props = res.dataset_example.model_dump(exclude={'expected_output'})
        # Serialize the entire expected_output (3-phase format: provision, expected, assert actions)
        example_node_props['expected_output'] = json.dumps(res.dataset_example.expected_output.model_dump())
        
        cypher_params.append({
            "example": example_node_props,
            "result": result_node_props,
        })
    
    # 2. Define the Cypher query to create the graph structure
    cypher_query = """
    UNWIND $params as row

    // Merge the TestCase node, now namespaced by user_id
    MERGE (ex:DatasetExample {example_id: row.example.example_id, user_id: $user_id})
    ON CREATE SET 
        ex += row.example,
        ex.status = 'active',
        ex.agent_name = $agent_name 
    ON MATCH SET 
        ex += row.example,
        ex.status = COALESCE(ex.status, 'active'),
        ex.agent_name = $agent_name 

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
        params={"params": cypher_params, "user_id": user_id, "agent_name": agent_name}
    )
    
    logger.info(f"run.execute: Neo4j query response: {query_result}")

    experiment = state.active_experiment
    experiment.results.extend(results)

    if results:
        experiment.started_at = min(res.started_at for res in results)
        experiment.completed_at = max(res.completed_at for res in results)
        experiment.mean_score = round(sum(res.score for res in results) / len(results), 5)
        
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
            "run_name": state.context.github_context.agent_name if state.context.github_context else "",
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
