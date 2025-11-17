"""
Test orchestration and high-level test runner functions.

Implements 3-phase testing:
1. PROVISION: Create test data
2. INVOKE: Run agent with input_message
3. ASSERT: Verify final state
4. CLEANUP: Delete created resources
"""
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from shared.schema import (
    DatasetExample, 
    ExperimentResultContext, 
    SandboxContext, 
    GithubContext,
    FailureAnalysis
)
from shared.config import EVAL_PASS_THRESHOLD
from shared.logger import get_logger
from shared.test_runner.action_executor import load_mcp_tools, execute_action_sequence
from shared.test_runner.agent_invoker import invoke_target_agent
from shared.mcp_client import get_mcp_client_and_configs


logger = get_logger("test_runner.orchestrator")


async def run_tests(
    dataset_examples: List[DatasetExample],
    sandbox_context: SandboxContext, 
    github_context: GithubContext,
    mcp_services: List[str] = None,
    mcp_resources: Dict[str, Any] = None
) -> List[ExperimentResultContext]:
    """
    Run a batch of test cases using 3-phase testing architecture.
    
    **Phase 1: PROVISION** - Create test data from scratch
    **Phase 2: INVOKE** - Send input_message to agent, capture tool calls
    **Phase 3: ASSERT** - Verify final state with assertions
    **Phase 4: CLEANUP** - Delete created resources
    
    Args:
        dataset_examples: Test cases to run
        sandbox_context: Sandbox where agent is deployed
        github_context: Context about the agent being tested
        mcp_services: MCP services to load tools from
        mcp_resources: Pre-provisioned resources (from earlier setup)
    
    Returns:
        List of test results
        
    Raises:
        ValueError: If required context is missing
    """
    if not sandbox_context:
        raise ValueError("sandbox_context is required for test execution. Cannot run tests without sandbox.")
    
    if not github_context:
        raise ValueError("github_context is required for test execution. Cannot identify agent without context.")
    
    mcp_services = mcp_services or []
    mcp_resources = mcp_resources or {}
    
    logger.info(f"Starting 3-phase test runner for {len(dataset_examples)} tests...")
    logger.info(f"Testing agent: {github_context.agent_name}")
    
    # Load MCP tools for provision/assert phases
    tools_dict = await load_mcp_tools(mcp_services)
    logger.info(f"Loaded {len(tools_dict)} MCP tools")
    
    # Get MCP configs for agent invocation
    _, mcp_configs = await get_mcp_client_and_configs(mcp_services)

    results: List[ExperimentResultContext] = []

    for idx, tc in enumerate(dataset_examples, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {idx}/{len(dataset_examples)}: {tc.example_id}")
        logger.info(f"Input: {tc.input_message}")
        logger.info(f"{'='*80}\n")
        
        run_start = datetime.now(timezone.utc)
        
        # 3-phase testing
        thread_id = None
        eval_result_obj = None
        agent_actual_output = ""
        provision_variables = {}
        cleanup_stack = []
        
        try:
            # ===== PHASE 1: PROVISION =====
            if tc.expected_output.provision_actions:
                logger.info(f"┌─ PHASE 1: PROVISION ({len(tc.expected_output.provision_actions)} actions)")
                
                provision_result, _, provision_variables, provision_cleanup = await execute_action_sequence(
                    tc.expected_output.provision_actions,
                    tools_dict,
                    mcp_resources,
                    run_label=f"Provision {tc.example_id}",
                    require_assertion=False  # Provisioning doesn't need assertions
                )
                
                cleanup_stack.extend(provision_cleanup)
                logger.info(f"└─ Provisioned {len(provision_variables)} resources")
                
                # If provisioning failed, skip to cleanup
                if provision_result.score < EVAL_PASS_THRESHOLD:
                    raise RuntimeError(
                        f"Provisioning failed: {provision_result.judge_reasoning}. "
                        f"Cannot continue with agent invocation."
                    )
            else:
                logger.info("┌─ PHASE 1: PROVISION (skipped - no provision_actions)")
            
            # Merge provisioned resources with global mcp_resources
            test_resources = {**mcp_resources, **provision_variables}
            
            # ===== PHASE 2: INVOKE AGENT =====
            logger.info(f"┌─ PHASE 2: INVOKE AGENT")
            logger.info(f"│  Sending: '{tc.input_message}'")
            
            invocation_result = await invoke_target_agent(
                sandbox_context=sandbox_context,
                github_context=github_context,
                input_message=tc.input_message,
                mcp_resources=test_resources,
                mcp_configs=mcp_configs,
                timeout_seconds=300
            )
            
            thread_id = invocation_result.thread_id
            agent_actual_output = invocation_result.final_output or ""
            
            logger.info(f"│  Agent made {len(invocation_result.tool_calls)} tool calls")
            logger.info(f"└─ Execution time: {invocation_result.execution_time_seconds:.2f}s")
            
            if invocation_result.error:
                raise RuntimeError(f"Agent invocation error: {invocation_result.error}")
            
            # ===== PHASE 3: ASSERT =====
            if not tc.expected_output.assert_actions:
                raise ValueError(
                    f"Test {tc.example_id} has no assert_actions! "
                    "Tests must include assertions to verify final state."
                )
            
            logger.info(f"┌─ PHASE 3: ASSERT ({len(tc.expected_output.assert_actions)} assertions)")
            
            eval_result_obj, assert_output, _, assert_cleanup = await execute_action_sequence(
                tc.expected_output.assert_actions,
                tools_dict,
                test_resources,
                run_label=f"Assert {tc.example_id}",
                require_assertion=True
            )
            
            cleanup_stack.extend(assert_cleanup)
            
            if eval_result_obj.score >= EVAL_PASS_THRESHOLD:
                logger.info(f"└─ ✅ ASSERTIONS PASSED")
            else:
                logger.warning(f"└─ ❌ ASSERTIONS FAILED: {eval_result_obj.judge_reasoning}")
            
        except Exception as e:
            logger.error(f"Test execution failed: {e}", exc_info=True)
            eval_result_obj = FailureAnalysis(
                score=0.0,
                failure_type="runtime_error",
                judge_reasoning=f"Test execution failed: {str(e)}"
            )
            thread_id = thread_id or f"failed_run_{uuid.uuid4().hex[:8]}"
        
        finally:
            # ===== PHASE 4: CLEANUP =====
            if cleanup_stack:
                logger.info(f"┌─ PHASE 4: CLEANUP ({len(cleanup_stack)} actions)")
                try:
                    await execute_action_sequence(
                        cleanup_stack[::-1],  # Reverse order (LIFO)
                        tools_dict,
                        test_resources,
                        run_label=f"Cleanup {tc.example_id}",
                        require_assertion=False
                    )
                    logger.info(f"└─ Cleanup completed")
                except Exception as cleanup_error:
                    logger.error(f"Cleanup failed (non-fatal): {cleanup_error}")
        
        run_end = datetime.now(timezone.utc)
        
        # Create result
        result = ExperimentResultContext(
            dataset_example=tc,
            thread_id=thread_id,
            actual_output=agent_actual_output,
            analysis=eval_result_obj,
            passed=eval_result_obj.score >= EVAL_PASS_THRESHOLD,
            started_at=run_start,
            completed_at=run_end,
        )
        
        results.append(result)
        
        logger.info(f"\n{'='*80}")
        logger.info(f"TEST {idx} RESULT: {'✅ PASSED' if result.passed else '❌ FAILED'}")
        logger.info(f"Score: {eval_result_obj.score:.2f}")
        logger.info(f"{'='*80}\n")
    
    passed_count = sum(1 for r in results if r.passed)
    logger.info(f"\n{'='*80}")
    logger.info(f"TEST RUN COMPLETE: {passed_count}/{len(results)} passed")
    logger.info(f"{'='*80}\n")
    
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

