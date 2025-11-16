"""Test orchestration and high-level test runner functions."""
import uuid
from typing import List, Dict, Any
from datetime import datetime, timezone

from shared.schema import DatasetExample, ExperimentResultContext, SandboxContext, GithubContext
from shared.config import EVAL_PASS_THRESHOLD
from shared.logger import get_logger
from shared.test_runner.action_executor import load_mcp_tools, execute_action_sequence


logger = get_logger("test_runner.orchestrator")


async def run_tests(
    dataset_examples: List[DatasetExample],
    sandbox_context: SandboxContext, 
    github_context: GithubContext,
    mcp_services: List[str] = None,
    mcp_resources: Dict[str, Any] = None
) -> List[ExperimentResultContext]:
    """
    Run a batch of test cases using MCP tools and action sequences.
    
    Each test case contains an expected output with a list of actions to execute.
    The test passes if all actions execute successfully and assertions pass.
    """
    mcp_services = mcp_services or []
    mcp_resources = mcp_resources or {}
    
    logger.info(f"Starting action-based test runner for {len(dataset_examples)} tests...")
    tools_dict = await load_mcp_tools(mcp_services)

    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        run_start = datetime.now(timezone.utc)
        thread_id = f"mcp_run_{uuid.uuid4().hex[:8]}" 

        eval_result_obj, agent_actual_output, _, _ = await execute_action_sequence(
            tc.expected_output.actions,
            tools_dict,
            mcp_resources,
            run_label=f"Test {tc.example_id}",
        )
        
        run_end = datetime.now(timezone.utc)
        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread_id,
                actual_output=agent_actual_output,
                analysis=eval_result_obj,
                passed=eval_result_obj.score >= EVAL_PASS_THRESHOLD,
                started_at=run_start,
                completed_at=run_end,
            )
        )
    
    logger.info(f"Action-based test run complete. {len(results)} results.")
    return results


async def execute_action_plan(
    actions,
    mcp_services,
    mcp_resources,
    *,
    run_label: str,
    assign_callback=None,
    require_assertion: bool = True,
):
    """
    Execute a single action plan, typically used for provisioning or setup.
    
    This is a convenience wrapper around execute_action_sequence that handles
    MCP tool loading.
    """
    tools_dict = await load_mcp_tools(mcp_services)
    return await execute_action_sequence(
        actions,
        tools_dict,
        mcp_resources,
        run_label,
        assign_callback=assign_callback,
        require_assertion=require_assertion,
    )

