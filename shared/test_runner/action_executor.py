"""Action execution core logic for test runner."""
import asyncio
import json
import inspect
from typing import List, Dict, Any, Optional, Callable

from shared.schema import FailureAnalysis, ActionStep
from shared.tools import canonicalize_tool_name
from shared.tool_service import get_tool_service
from shared.logger import get_logger
from langchain_core.tools import BaseTool

from shared.test_runner.variable_injection import get_field, inject_variables
from shared.test_runner.parameter_sanitization import sanitize_tool_params
from shared.cleanup_inverses import create_inverse_action


logger = get_logger("test_runner.action_executor")


async def load_mcp_tools(mcp_services: List[str]) -> Dict[str, BaseTool]:
    """Load MCP tools using ToolService singleton."""
    tool_service = get_tool_service()
    await tool_service.initialize(mcp_services)
    tools_dict = tool_service.get_tools()

    if not tools_dict and mcp_services:
        logger.error(
            f"MCP services {mcp_services} were requested, but no tools were loaded."
        )
        logger.error(
            "This usually means the local MCP servers (ports 8004, 8005) are not running."
        )
    else:
        logger.info(
            f"Loaded {len(tools_dict)} MCP tools from ToolService"
        )
    return tools_dict


async def maybe_call_assign_callback(callback: Optional[Callable], name: str, output: Any):
    """Call the assignment callback if provided, handling both sync and async."""
    if not callback:
        return
    result = callback(name, output)
    if inspect.isawaitable(result):
        await result


async def execute_action_sequence(
    actions: List[ActionStep],
    tools_dict: Dict[str, BaseTool],
    mcp_resources: Dict[str, Any],
    run_label: str,
    assign_callback: Optional[Callable] = None,
    require_assertion: bool = True,
) -> tuple[FailureAnalysis, str, Dict[str, Any], List[ActionStep]]:
    """
    Execute a sequence of actions, injecting variables and evaluating assertions.
    
    Returns:
        - FailureAnalysis: The evaluation result
        - str: The final agent output (JSON string)
        - Dict[str, Any]: Variables collected during execution
        - List[ActionStep]: Cleanup actions generated (inverse of provisioning actions)
    """
    variables: Dict[str, Any] = {}
    agent_actual_output = ""
    eval_result_obj: Optional[FailureAnalysis] = None
    cleanup_stack: List[ActionStep] = []  # NEW: Track cleanup actions

    try:
        for idx, action in enumerate(actions):
            tool_name = canonicalize_tool_name(action.tool)

            params_str = action.params or "{}"
            try:
                params_dict = json.loads(params_str)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{run_label}: Step {idx+1} has invalid JSON params: {params_str}. "
                    f"JSON decode error: {str(e)}. Fix the test generation to produce valid JSON."
                ) from e
            
            params = inject_variables(params_dict, variables, mcp_resources)
            params = sanitize_tool_params(tool_name, params, mcp_resources)

            logger.info(f"{run_label}: Step {idx+1} - Executing {action.tool}")

            if tool_name == "system.wait":
                await asyncio.sleep(params.get("seconds", 1))
                output: Any = {"status": "wait_completed"}

            elif tool_name in tools_dict:
                tool_to_run = tools_dict[tool_name]
                output = await tool_to_run.ainvoke(params)

            else:
                raise ValueError(
                    f"Unknown tool: {action.tool}. Available MCP tools: {list(tools_dict.keys())}"
                )

            if action.assign_to_var:
                var_value = output.get("id") if isinstance(output, dict) else output
                variables[action.assign_to_var] = var_value
                logger.info(
                    f"  > Stored '{str(var_value)[:50]}' in var '{action.assign_to_var}'"
                )
                await maybe_call_assign_callback(assign_callback, action.assign_to_var, output)

            # NEW: Generate inverse cleanup action with LLM
            inverse_action = await create_inverse_action(
                original=action,
                output=output,
                assign_var=action.assign_to_var if action.assign_to_var else None,
                available_tools=list(tools_dict.keys())
            )
            if inverse_action:
                cleanup_stack.append(inverse_action)
                logger.info(f"  > Recorded cleanup action: {inverse_action.tool}")

            agent_actual_output = json.dumps(output, default=str)

            if action.assert_field:
                actual_value = get_field(output, action.assert_field)
                expected_value_str = action.assert_expected

                # Normalize None handling: None is represented as None, not empty string
                # This ensures consistent comparison across different data sources
                passed = False
                try:
                    expected_as_json = json.loads(expected_value_str)
                    # Direct comparison - None matches None, "" matches ""
                    passed = actual_value == expected_as_json
                except json.JSONDecodeError:
                    # String comparison with special handling for None
                    if actual_value is None:
                        # None only matches "null" string or empty expected value  
                        passed = expected_value_str.lower() == "null" or expected_value_str == ""
                    else:
                        # Normal string comparison
                        passed = str(actual_value) == expected_value_str

                if passed:
                    logger.info(
                        f"  > ASSERTION PASSED: {action.assert_field} == {expected_value_str}"
                    )
                    eval_result_obj = FailureAnalysis(
                        score=1.0, judge_reasoning="Assertion passed."
                    )
                else:
                    # Assertion failed - this is EXPECTED behavior for tests, not an error
                    logger.info(
                        f"  > ASSERTION FAILED: Expected {action.assert_field} to be '{expected_value_str}', but got '{actual_value}'"
                    )
                    eval_result_obj = FailureAnalysis(
                        score=0.0,
                        failure_type="assertion_error",
                        judge_reasoning=(
                            f"Assertion failed: Expected {action.assert_field} to be '{expected_value_str}', but got '{actual_value}'"
                        ),
                    )
                break

        else:
            if require_assertion:
                raise ValueError(
                    f"{run_label}: Action sequence finished without any assertion step. "
                    f"Test must have at least one action with assert_field set to verify behavior."
                )
                eval_result_obj = FailureAnalysis(
                    score=0.0,
                    failure_type="completeness",
                    judge_reasoning="Action list finished without a final assertion step."
                )
            else:
                eval_result_obj = FailureAnalysis(
                    score=1.0,
                    failure_type=None,
                    judge_reasoning="Provisioning plan executed without assertions.",
                )

    except Exception as exc:
        logger.error(
            f"Error during action execution for {run_label}: {exc}", exc_info=True
        )
        eval_result_obj = FailureAnalysis(
            score=0.0,
            failure_type="runtime_error",
            judge_reasoning=f"Execution failed: {exc}",
        )

    return eval_result_obj, agent_actual_output, variables, cleanup_stack
