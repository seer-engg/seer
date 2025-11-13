import uuid
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime, timezone
import re
from e2b import AsyncSandbox
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from shared.mcp_client import get_mcp_client_and_configs
from shared.schema import DatasetExample, ExperimentResultContext, SandboxContext, GithubContext, FailureAnalysis
from shared.logger import get_logger
from sandbox.constants import TARGET_AGENT_PORT
from langchain_core.tools import BaseTool

logger = get_logger("test_runner.action_executor")


def _get_field(obj: Any, field_path: str) -> Any:
    """Helper to get a nested field from an object using dot notation."""
    if not obj or not field_path:
        return obj
    try:
        keys = field_path.split('.')
        value = obj
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return None # Path is invalid
        return value
    except Exception:
        return None

def _inject_variables(
    params_data: Any, # Can be dict, list, or str
    variables: Dict[str, Any], 
    mcp_resources: Dict[str, Any]
) -> Any:
    """
    Recursively injects variables and resource IDs into tool parameters.
    - [var:var_name] is replaced by variables['var_name']
    - [resource:resource_key] is replaced by mcp_resources['resource_key']['id']
    - [resource:resource_key.field] is replaced by a nested field
    """
    if isinstance(params_data, dict):
        return {
            k: _inject_variables(v, variables, mcp_resources) 
            for k, v in params_data.items()
        }
    if isinstance(params_data, list):
        return [
            _inject_variables(item, variables, mcp_resources) 
            for item in params_data
        ]
    if isinstance(params_data, str):
        # 1. Check for full-string replacement
        
        # Inject [var:var_name]
        var_match = re.match(r"^\[var:(\w+)\]$", params_data)
        if var_match:
            var_name = var_match.group(1)
            return variables.get(var_name)
        
        # Inject [resource:resource_key] (assumes ID)
        res_match = re.match(r"^\[resource:(\w+)\]$", params_data)
        if res_match:
            res_key = res_match.group(1)
            return mcp_resources.get(res_key, {}).get('id')
            
        # Inject [resource:resource_key.field_name]
        res_field_match = re.match(r"^\[resource:(\w+)\.(.+)\]$", params_data)
        if res_field_match:
            res_key = res_field_match.group(1)
            field_path = res_field_match.group(2)
            return _get_field(mcp_resources.get(res_key), field_path)

        # 2. Check for inline (partial string) replacement
        
        # Handle inline [var:...]
        injected_str = re.sub(
            r'\[var:(\w+)\]', 
            lambda m: str(variables.get(m.group(1), '')), 
            params_data
        )
        # Handle inline [resource:...] (assumes ID)
        injected_str = re.sub(
            r'\[resource:(\w+)\]', 
            lambda m: str(mcp_resources.get(m.group(1), {}).get('id', '')), 
            injected_str
        )
        return injected_str
        
    return params_data

async def run_tests(
    dataset_examples: List[DatasetExample],
    sandbox_context: SandboxContext, 
    github_context: GithubContext,
    mcp_services: List[str],
    mcp_resources: Dict[str, Any]
) -> List[ExperimentResultContext]:

    logger.info(f"Starting action-based test runner for {len(dataset_examples)} tests...")
    
    # 1. Get the MCP client and tools
    mcp_client, mcp_configs = await get_mcp_client_and_configs(mcp_services)
    mcp_tools = await mcp_client.get_tools()
    tools_dict: Dict[str, BaseTool] = {t.name: t for t in mcp_tools}
    
    if not tools_dict and mcp_services:
        logger.error(f"MCP services {mcp_services} were requested, but no tools were loaded.")
        logger.error("This usually means the local MCP servers (ports 8004, 8005) are not running.")
    else:
        logger.info(f"Loaded {len(tools_dict.keys())} MCP tools: {list(tools_dict.keys())}")
    
    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        run_start = datetime.now(timezone.utc)
        variables: Dict[str, Any] = {}
        eval_result_obj: FailureAnalysis
        agent_actual_output = ""
        # We don't create a TA thread here, we are just running MCP tools
        thread_id = f"mcp_run_{uuid.uuid4().hex[:8]}" 

        try:

            # 4. Iterate through the action sequence
            for i, action in enumerate(tc.expected_output.actions):
                tool_name = action.tool
                
                # ... (Params parsing and variable injection logic is unchanged) ...
                params_str = action.params or "{}"
                try:
                    params_dict = json.loads(params_str)
                except json.JSONDecodeError:
                    logger.warning(f"  > Invalid JSON in params: {params_str}. Using empty dict.")
                    params_dict = {}
                params = _inject_variables(params_dict, variables, mcp_resources)
                # ... (end of unchanged logic) ...

                logger.info(f"Test {tc.example_id}: Step {i+1} - Executing {tool_name}")

                # 5. Execute action
                output: Any
                if tool_name == "system.wait":
                    await asyncio.sleep(params.get("seconds", 1))
                    output = {"status": "wait_completed"}
                
                elif tool_name == "target_agent.invoke":
                    # This is a special case for invoking the agent being tested
                    try:
                        # Get a client for the target agent
                        target_agent_client = get_sync_client(
                            url=f"http://127.0.0.1:{TARGET_AGENT_PORT}"
                        )
                        output = target_agent_client.runs.invoke(
                            thread_id=thread_id,
                            graph_id="agent",
                            input=params,
                        )
                    except Exception as e:
                        logger.error(f"Error invoking target agent: {e}", exc_info=True)
                        output = {"status": "error", "error": str(e)}

                elif tool_name in tools_dict:
                    # This is an MCP tool call (e.g., asana.create_task)
                    tool_to_run = tools_dict[tool_name]
                    output = await tool_to_run.ainvoke(params)
                
                else:
                    raise ValueError(f"Unknown service/tool: {tool_name}. Available MCP tools: {list(tools_dict.keys())}")
                
                # 6. Store variable if needed
                if action.assign_to_var:
                    var_value = output.get('id') if isinstance(output, dict) else output
                    variables[action.assign_to_var] = var_value
                    logger.info(f"  > Stored '{str(var_value)[:50]}' in var '{action.assign_to_var}'")
                
                agent_actual_output = json.dumps(output, default=str)

                # 7. Perform assertion if needed
                if action.assert_field:
                    # ... (This logic is unchanged, it's correct) ...
                    actual_value = _get_field(output, action.assert_field)
                    expected_value_str = action.assert_expected
                    
                    passed = False
                    comparison_reason = ""
                    try:
                        expected_as_json = json.loads(expected_value_str)
                        passed = (actual_value == expected_as_json)
                        comparison_reason = f"Actual object ({actual_value}) == Expected object ({expected_as_json})"
                    except json.JSONDecodeError:
                        passed = (str(actual_value) == expected_value_str)
                        comparison_reason = f"Actual value ({str(actual_value)}) == Expected value ({expected_value_str})"
                    
                    if passed:
                        logger.info(f"  > ASSERTION PASSED: {action.assert_field} == {expected_value_str}")
                        eval_result_obj = FailureAnalysis(score=1.0, judge_reasoning="Assertion passed.")
                    else:
                        logger.warning(f"  > ASSERTION FAILED: Expected {action.assert_field} to be '{expected_value_str}', but got '{actual_value}'")
                        eval_result_obj = FailureAnalysis(
                            score=0.0,
                            failure_type="assertion_error",
                            judge_reasoning=f"Assertion failed: Expected {action.assert_field} to be '{expected_value_str}', but got '{actual_value}'"
                        )
                    break 
            
            else:
                logger.warning(f"Test {tc.example_id} finished without an assertion step.")
                eval_result_obj = FailureAnalysis(
                    score=0.0,
                    failure_type="completeness",
                    judge_reasoning="Test case finished without a final assertion step."
                )

        except Exception as e:
            logger.error(f"Error during test execution for {tc.example_id}: {e}", exc_info=True)
            eval_result_obj = FailureAnalysis(
                score=0.0,
                failure_type="runtime_error",
                judge_reasoning=f"Test failed with runtime error: {e}",
            )
        
        run_end = datetime.now(timezone.utc)
        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread_id, # Use our generated run ID
                actual_output=agent_actual_output,
                analysis=eval_result_obj,
                started_at=run_start,
                completed_at=run_end,
            )
        )
    
    logger.info(f"Action-based test run complete. {len(results)} results.")
    return results
