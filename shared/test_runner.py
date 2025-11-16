import os
import uuid
from typing import List, Dict, Any
import asyncio
import json
from datetime import datetime, timezone
import re
import inspect
from shared.mcp_client import get_mcp_client_and_configs
from shared.schema import DatasetExample, ExperimentResultContext, SandboxContext, GithubContext, FailureAnalysis, ActionStep
from shared.tool_catalog import canonicalize_tool_name
from shared.logger import get_logger
from shared.config import EVAL_PASS_THRESHOLD
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


_ASANA_WORKSPACE_ENV_KEYS = ("ASANA_WORKSPACE_ID", "ASANA_DEFAULT_WORKSPACE_GID")
_ASANA_PROJECT_ENV_KEYS = ("ASANA_PROJECT_ID", "ASANA_DEFAULT_PROJECT_GID")


def _get_env_value(*keys: str) -> str | None:
    for key in keys:
        val = os.getenv(key)
        if val:
            trimmed = val.strip()
            if trimmed:
                return trimmed
    return None


def _extract_resource_id(resource: Any) -> str | None:
    if not isinstance(resource, dict):
        return None
    data = resource.get("data")
    if isinstance(data, dict):
        resource = data
    for field in ("id", "gid"):
        value = resource.get(field)
        if value:
            return str(value)
    return None


def _resolve_workspace_gid(current: Any, mcp_resources: Dict[str, Any]) -> str | None:
    fallback = _extract_resource_id(mcp_resources.get("asana_workspace"))
    fallback = fallback or _get_env_value(*_ASANA_WORKSPACE_ENV_KEYS)
    current_str = str(current).strip() if current else ""

    if fallback:
        if current_str and current_str != fallback:
            logger.info(
                f"Overriding workspace_gid '{current_str}' with configured workspace '{fallback}'"
            )
        return fallback
    return current_str or None


def _resolve_project_gid(current: Any, mcp_resources: Dict[str, Any]) -> str | None:
    fallback = _extract_resource_id(mcp_resources.get("asana_project"))
    fallback = fallback or _get_env_value(*_ASANA_PROJECT_ENV_KEYS)
    current_str = str(current).strip() if current else ""

    if fallback:
        if current_str and current_str != fallback:
            logger.info(
                f"Overriding project_gid '{current_str}' with configured project '{fallback}'"
            )
        return fallback
    return current_str or None


def _sanitize_tool_params(
    tool_name: str,
    params: Dict[str, Any],
    mcp_resources: Dict[str, Any]
) -> Dict[str, Any]:
    sanitized = dict(params)

    if "opt_fields" in sanitized and isinstance(sanitized["opt_fields"], str):
        raw = sanitized["opt_fields"].strip()
        if raw:
            fields = [field.strip() for field in raw.split(",") if field.strip()]
            sanitized["opt_fields"] = fields or [raw]

    if tool_name.startswith("asana_"):
        if "workspace_gid" in sanitized:
            resolved_workspace = _resolve_workspace_gid(
                sanitized.get("workspace_gid"), mcp_resources
            )
            if resolved_workspace:
                sanitized["workspace_gid"] = resolved_workspace
        elif "workspace" in sanitized and isinstance(sanitized["workspace"], dict):
            resolved_workspace = _resolve_workspace_gid(
                sanitized["workspace"].get("gid"), mcp_resources
            )
            if resolved_workspace:
                sanitized.setdefault("workspace_gid", resolved_workspace)

        if "project_gid" in sanitized:
            resolved_project = _resolve_project_gid(
                sanitized.get("project_gid"), mcp_resources
            )
            if resolved_project:
                sanitized["project_gid"] = resolved_project

    return sanitized


async def _load_mcp_tools(mcp_services: List[str]) -> Dict[str, BaseTool]:
    mcp_client, _ = await get_mcp_client_and_configs(mcp_services)
    mcp_tools = await mcp_client.get_tools()
    tools_dict: Dict[str, BaseTool] = {
        canonicalize_tool_name(t.name): t for t in mcp_tools
    }

    if not tools_dict and mcp_services:
        logger.error(
            f"MCP services {mcp_services} were requested, but no tools were loaded."
        )
        logger.error(
            "This usually means the local MCP servers (ports 8004, 8005) are not running."
        )
    else:
        logger.info(
            f"Loaded {len(tools_dict.keys())} MCP tools: {[tool.name for tool in mcp_tools]}"
        )
    return tools_dict


async def _maybe_call_assign_callback(callback, name: str, output: Any):
    if not callback:
        return
    result = callback(name, output)
    if inspect.isawaitable(result):
        await result


async def _execute_action_sequence(
    actions: List[ActionStep],
    tools_dict: Dict[str, BaseTool],
    mcp_resources: Dict[str, Any],
    run_label: str,
    assign_callback=None,
    require_assertion: bool = True,
):
    variables: Dict[str, Any] = {}
    agent_actual_output = ""
    eval_result_obj: FailureAnalysis | None = None

    try:
        for idx, action in enumerate(actions):
            tool_name = canonicalize_tool_name(action.tool)

            params_str = action.params or "{}"
            try:
                params_dict = json.loads(params_str)
            except json.JSONDecodeError:
                logger.warning(
                    f"{run_label}: Step {idx+1} has invalid JSON params {params_str}. Using empty dict."
                )
                params_dict = {}
            params = _inject_variables(params_dict, variables, mcp_resources)
            params = _sanitize_tool_params(tool_name, params, mcp_resources)

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
                await _maybe_call_assign_callback(assign_callback, action.assign_to_var, output)

            agent_actual_output = json.dumps(output, default=str)

            if action.assert_field:
                actual_value = _get_field(output, action.assert_field)
                expected_value_str = action.assert_expected

                passed = False
                try:
                    expected_as_json = json.loads(expected_value_str)
                    passed = actual_value == expected_as_json
                except json.JSONDecodeError:
                    passed = str(actual_value) == expected_value_str

                if passed:
                    logger.info(
                        f"  > ASSERTION PASSED: {action.assert_field} == {expected_value_str}"
                    )
                    eval_result_obj = FailureAnalysis(
                        score=1.0, judge_reasoning="Assertion passed."
                    )
                else:
                    logger.warning(
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
                logger.warning(f"{run_label}: Finished without an assertion step.")
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

    if eval_result_obj is None:
        eval_result_obj = FailureAnalysis(
            score=1.0 if not require_assertion else 0.0,
            failure_type=None if not require_assertion else "completeness",
            judge_reasoning="Actions executed successfully." if not require_assertion else "No assertion was evaluated.",
        )

    return eval_result_obj, agent_actual_output, variables

async def run_tests(
    dataset_examples: List[DatasetExample],
    sandbox_context: SandboxContext, 
    github_context: GithubContext,
    mcp_services: List[str],
    mcp_resources: Dict[str, Any]
) -> List[ExperimentResultContext]:

    logger.info(f"Starting action-based test runner for {len(dataset_examples)} tests...")
    tools_dict = await _load_mcp_tools(mcp_services)

    results: List[ExperimentResultContext] = []

    for tc in dataset_examples:
        run_start = datetime.now(timezone.utc)
        thread_id = f"mcp_run_{uuid.uuid4().hex[:8]}" 

        eval_result_obj, agent_actual_output, _ = await _execute_action_sequence(
            tc.expected_output.actions,
            tools_dict,
            mcp_resources,
            run_label=f"Test {tc.example_id}",
        )
        
        run_end = datetime.now(timezone.utc)
        results.append(
            ExperimentResultContext(
                dataset_example=tc,
                thread_id=thread_id, # Use our generated run ID
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
    actions: List[ActionStep],
    mcp_services: List[str],
    mcp_resources: Dict[str, Any],
    *,
    run_label: str,
    assign_callback=None,
    require_assertion: bool = True,
):
    tools_dict = await _load_mcp_tools(mcp_services)
    return await _execute_action_sequence(
        actions,
        tools_dict,
        mcp_resources,
        run_label,
        assign_callback=assign_callback,
        require_assertion=require_assertion,
    )
