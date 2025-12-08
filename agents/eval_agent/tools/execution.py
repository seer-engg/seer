"""
Execution phase tools for eval agent Supervisor pattern.
"""
import json
from typing import Optional
from langchain_core.tools import tool
from langchain.tools import ToolRuntime

from shared.logger import get_logger
from shared.schema import AgentContext, DatasetExample, ExperimentResultContext
from agents.eval_agent.models import EvalAgentState, TestExecutionState
from agents.eval_agent.nodes.execute.graph import build_test_execution_subgraph
from agents.eval_agent.nodes.run import _prepare_run_context

logger = get_logger("eval_agent.tools.execution")


@tool
async def execute_test_batch(runtime: ToolRuntime = None) -> str:
    """
    Execute all test cases in batch.
    
    Reads dataset_examples and agent_context from state, executes tests,
    and stores results in latest_results.
    
    Returns:
        JSON string with execution summary
    """
    # Get state
    if not runtime or not hasattr(runtime, "state"):
        return json.dumps({"status": "error", "error": "No runtime state available"})
    
    state = runtime.state if isinstance(runtime.state, dict) else {}
    agent_context_dict = state.get("agent_context")
    dataset_examples_dicts = state.get("dataset_examples", [])
    
    if not agent_context_dict or not dataset_examples_dicts:
        return json.dumps({"status": "error", "error": "agent_context or dataset_examples not found"})
    
    try:
        # Reconstruct objects
        agent_context = AgentContext(**agent_context_dict) if isinstance(agent_context_dict, dict) else agent_context_dict
        
        dataset_examples = []
        for ex_dict in dataset_examples_dicts:
            if isinstance(ex_dict, dict):
                dataset_examples.append(DatasetExample(**ex_dict))
            else:
                dataset_examples.append(ex_dict)
        
        # Prepare run context
        temp_state = EvalAgentState(
            messages=[],
            context=agent_context,
            dataset_examples=dataset_examples,
        )
        
        run_context_result = await _prepare_run_context(temp_state)
        
        # Execute tests using the execution subgraph
        # For now, we'll use a simplified approach - execute each test
        # TODO: Use the full execution subgraph logic
        
        execution_subgraph = build_test_execution_subgraph()
        
        # Create test execution state
        test_exec_state = TestExecutionState(
            context=agent_context,
            dataset_examples=dataset_examples,
        )
        
        # Execute the subgraph
        config = {"recursion_limit": 100}
        result = await execution_subgraph.ainvoke(test_exec_state, config)
        
        latest_results = result.get("latest_results", [])
        
        # Count results
        passed = sum(1 for r in latest_results if r.passed)
        failed = len(latest_results) - passed
        
        results_dicts = [r.model_dump() if hasattr(r, "model_dump") else r for r in latest_results]
        
        logger.info(f"✅ Executed {len(latest_results)} tests: {passed} passed, {failed} failed")
        return json.dumps({
            "status": "success",
            "latest_results": results_dicts,
            "message": f"✅ Executed {len(latest_results)} tests: {passed} passed, {failed} failed"
        })
    except Exception as e:
        logger.error(f"Error executing test batch: {e}", exc_info=True)
        return json.dumps({"status": "error", "error": str(e)})

