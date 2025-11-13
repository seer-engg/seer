import asyncio
import json
import re
from typing import List, Dict, Any
from datetime import datetime, timezone
from e2b import AsyncSandbox
from langgraph_sdk import get_sync_client
from langgraph.pregel.remote import RemoteGraph
from langchain_core.tools import BaseTool

from shared.mcp_client import get_mcp_client_and_configs
from shared.schema import DatasetExample, ExperimentResultContext, SandboxContext, GithubContext, FailureAnalysis
from shared.logger import get_logger
from sandbox.constants import TARGET_AGENT_PORT

logger = get_logger("test_runner.action_executor")


def _get_field(obj: Any, field_path: str) -> Any:
    """Helper to get a nested field from an object using dot notation."""
    try:
        keys = field_path.split('.')
        value = obj
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                value = value[int(key)]
            else:
                return None
        return value
    except Exception:
        return None

def _inject_variables(
    params: Dict[str, Any], 
    variables: Dict[str, Any], 
    mcp_resources: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively injects variables and resource IDs into tool parameters.
    - [var:var_name] is replaced by variables['var_name']
    - [resource:resource_key] is replaced by mcp_resources['resource_key']['id']
    """
    if isinstance(params, dict):
        return {
            k: _inject_variables(v, variables, mcp_resources) 
            for k, v in params.items()
        }
    if isinstance(params, list):
        return [
            _inject_variables(item, variables, mcp_resources) 
            for item in params
        ]
    if isinstance(params, str):
        # Inject variables
        var_match = re.match(r"^\[var:(\w+)\]$", params)
        if var_match:
            var_name = var_match.group(1)
            return variables.get(var_name)
        
        # Inject resource IDs
        res_match = re.match(r"^\[resource:(\w+)\]$", params)
        if res_match:
            res_key = res_match.group(1)
            # Assumes we want the 'id' field from the resource object
            return mcp_resources.get(res_key, {}).get('id')
            
        # Inject resource fields (e.g., [resource:github_repo.full_name])
        res_field_match = re.match(r"^\[resource:(\w+)\.(.+)\]$", params)
        if res_field_match:
            res_key = res_field_match.group(1)
            field_path = res_field_match.group(2)
            return _get_field(mcp_resources.get(res_key), field_path)

        # Handle inline replacements
        params = re.sub(
            r'\[var:(\w+)\]', 
            lambda m: str(variables.get(m.group(1), '')), 
            params
        )
        params = re.sub(
            r'\[resource:(\w+)\]', 
            lambda m: str(mcp_resources.get(m.group(1), {}).get('id', '')), 
            params
        )
    return params


async def run_tests(
    dataset_examples: List[DatasetExample],
    sandbox_context: SandboxContext, 
    github_context: GithubContext,
    mcp_services: List[str],
    mcp_resources: Dict[str, Any],
) -> List[ExperimentResultContext]:

    logger.info(f"Starting action-based test runner for {len(dataset_examples)} tests...")
    
    # 1. Get the MCP client and tools
    mcp_client, _ = await get_mcp_client_and_configs(mcp_services)
    mcp_tools = await mcp_client.get_tools()
    tools_dict: Dict[str, BaseTool] = {t.name: t for t in mcp_tools}
    
    # 2. Get Target Agent client
    sbx = await AsyncSandbox.connect(sandbox_context.sandbox_id, timeout=60 * 20)
    sync_client = get_sync_client(url=sbx.get_host(TARGET_AGENT_PORT))
    remote_graph = RemoteGraph(
        github_context.agent_name,
        sync_client=sync_client,
    )

    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        run_start = datetime.now(timezone.utc)
        variables: Dict[str, Any] = {}
        eval_result_obj: FailureAnalysis
        agent_actual_output = "" # Will store the final output for logging

        try:
            # 3. Create a fresh thread for the Target Agent
            thread = await asyncio.to_thread(sync_client.threads.create)
            thread_cfg = {"configurable": {"thread_id": thread["thread_id"]}}

            # 4. Iterate through the action sequence
            for i, action in enumerate(tc.expected_output.actions):
                service = action.get("service")
                tool_name = action.get("tool")
                full_tool_name = f"{service}.{tool_name}"
                
                params = _inject_variables(
                    action.get("params", {}), 
                    variables, 
                    mcp_resources
                )

                logger.info(f"Test {tc.example_id}: Step {i+1} - Executing {full_tool_name}")

                # 5. Execute action
                output: Any
                if service == "system" and tool_name == "wait":
                    await asyncio.sleep(params.get("seconds", 1))
                    output = {"status": "wait_completed"}
                
                elif full_tool_name in tools_dict:
                    # This is an MCP tool call (e.g., asana.create_task)
                    tool_to_run = tools_dict[full_tool_name]
                    output = await tool_to_run.ainvoke(params)
                
                elif service == "target_agent":
                    # This is an invocation of the Target Agent
                    logger.info(f"Invoking target_agent with: {params.get('content')}")
                    agent_response = await asyncio.to_thread(
                        remote_graph.invoke,
                        {"messages": [{"role": "user", "content": params.get("content")}]},
                        thread_cfg,
                    )
                    # Extract text content
                    last_msg = agent_response.get("messages", [{}])[-1]
                    output = last_msg.get("content", "")
                    logger.info(f"Target agent responded with: {output[:100]}...")

                else:
                    raise ValueError(f"Unknown service/tool: {full_tool_name}")

                # 6. Store variable if needed
                if "assign_to_var" in action:
                    # Try to get 'id' first, fall back to full output
                    var_value = output.get('id') if isinstance(output, dict) else output
                    variables[action["assign_to_var"]] = var_value
                    logger.info(f"  > Stored '{var_value}' in var '{action['assign_to_var']}'")
                
                agent_actual_output = json.dumps(output, default=str)

                # 7. Perform assertion if needed (must be the last step)
                if "assert_field" in action:
                    actual_value = _get_field(output, action["assert_field"])
                    expected_value = action["assert_expected"]
                    
                    if actual_value == expected_value:
                        logger.info(f"  > ASSERTION PASSED: {action['assert_field']} == {expected_value}")
                        eval_result_obj = FailureAnalysis(
                            score=1.0,
                            judge_reasoning="Assertion passed."
                        )
                    else:
                        logger.warning(f"  > ASSERTION FAILED: Expected {action['assert_field']} to be '{expected_value}', but got '{actual_value}'")
                        eval_result_obj = FailureAnalysis(
                            score=0.0,
                            failure_type="assertion_error",
                            judge_reasoning=f"Assertion failed: Expected {action['assert_field']} to be '{expected_value}', but got '{actual_value}'"
                        )
                    break # Test is over
            
            else:
                # Loop finished without an assertion
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
                thread_id=thread["thread_id"],
                actual_output=agent_actual_output,
                analysis=eval_result_obj,
                started_at=run_start,
                completed_at=run_end,
            )
        )
    
    logger.info(f"Action-based test run complete. {len(results)} results.")
    return results
