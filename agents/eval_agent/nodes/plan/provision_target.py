import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict

from agents.eval_agent.models import EvalAgentPlannerState
from sandbox import (
    initialize_e2b_sandbox,
    setup_project,
    TARGET_AGENT_COMMAND,
    TARGET_AGENT_PORT,
    deploy_server_and_confirm_ready,
)
from shared.agent_context import AgentContext
from shared.schema import SandboxContext, ActionStep
from shared.logger import get_logger
from shared.tool_service import get_tool_service
from shared.tools import ToolEntry
from shared.tools.schema_formatter import format_tool_schemas_for_llm
from shared.resource_utils import format_resource_hints
from shared.test_runner import execute_action_plan
from shared.parameter_population import (
    extract_all_context_variables,
    format_context_variables_for_llm,
    complete_action_list,
)

logger = get_logger("eval_agent.plan")


class _ProvisioningPlan(BaseModel):
    """Structured response describing the MCP tool calls to run."""

    model_config = ConfigDict(extra="forbid")
    actions: List[ActionStep]


def _seed_default_resources(mcp_resources: Dict[str, Any]) -> None:
    workspace_id = os.getenv("ASANA_WORKSPACE_ID") or os.getenv("ASANA_DEFAULT_WORKSPACE_GID")
    if workspace_id and "asana_workspace" not in mcp_resources:
        mcp_resources["asana_workspace"] = {"id": workspace_id, "gid": workspace_id}
        logger.info("Seeded Asana workspace from environment: %s", workspace_id)

    project_id = os.getenv("ASANA_PROJECT_ID") or os.getenv("ASANA_DEFAULT_PROJECT_GID")
    if project_id and "asana_project" not in mcp_resources:
        mcp_resources["asana_project"] = {"id": project_id, "gid": project_id}
        logger.info("Seeded Asana project from environment: %s", project_id)


def _resource_assignment_callback(mcp_resources: Dict[str, Any]):
    async def _callback(name: str, output: Any):
        if not name:
            return
        if isinstance(output, dict):
            payload = output
        else:
            payload = {"value": output}
        mcp_resources[name] = payload
        logger.info("Stored MCP resource '%s'", name)

    return _callback


_PROVISION_PROMPT = """### TOOL PROVISIONING PLANNER ###
You are responsible for preparing external services before the evaluation agent runs.

**User Request / Context:**
{raw_request}

**MCP Services In Scope:** {services}

**Existing Resources:**
{resource_hints}

{context_variables}

**Available Tools with Schemas:**
{formatted_tool_schemas}

Design a concise sequence of MCP tool calls (1-6 steps) that ensures:
1. All required workspaces/projects/repos exist for the services mentioned.
2. Any IDs needed later are captured with `assign_to_var` so they can be referenced via `[var:...]` and persisted.
3. You MUST provide ALL required parameters for each tool call using the context variables above.
4. Use tools from the list above and respect their parameter schemas exactly.

**IMPORTANT**: Use the context variables shown above to populate required parameters.
For example, if you need 'owner' and 'repo', use the github_owner and github_repo values provided.

Return the ProvisioningPlan with fully-populated ActionStep objects.
Each ActionStep.params must be valid JSON containing ALL required fields.
"""


async def _plan_provisioning_actions(
    state: EvalAgentPlannerState,
    available_tools: List[str],
    resource_hints: str,
    tool_entries: Dict[str, ToolEntry],
    context_vars: Dict[str, Any],
) -> List[ActionStep]:
    provision_llm = ChatOpenAI(
        model="gpt-5-codex",
        use_responses_api=True,
        output_version="responses/v1",
        reasoning={"effort": "low"},
    )
    structured = provision_llm.with_structured_output(_ProvisioningPlan)

    # Format tool schemas for the prompt using shared formatter
    formatted_schemas = format_tool_schemas_for_llm(tool_entries, available_tools)
    
    # Format context variables for LLM
    formatted_context_vars = format_context_variables_for_llm(context_vars)

    prompt = _PROVISION_PROMPT.format(
        raw_request=state.context.user_context.raw_request if state.context.user_context else "",
        services=", ".join(state.context.mcp_services),
        resource_hints=resource_hints,
        formatted_tool_schemas=formatted_schemas,
        context_variables=formatted_context_vars,
    )

    plan = await structured.ainvoke(prompt)
    
    completed_actions = await complete_action_list(
        actions=plan.actions,
        tool_entries=tool_entries,
        context_vars=context_vars,
        aggressive=True,  # FAIL if required params missing
    )
    
    return completed_actions


async def _run_mcp_provisioning(
    state: EvalAgentPlannerState,
    mcp_resources: Dict[str, Any],
) -> tuple[Dict[str, Any], List[ActionStep]]:
    """
    Run MCP provisioning actions and generate cleanup stack.
    
    Returns:
        - mcp_resources: Updated resources dict
        - cleanup_stack: List of inverse cleanup actions (LIFO)
    """
    if not state.context.mcp_services:
        return mcp_resources, []

    tool_service = get_tool_service()
    await tool_service.initialize(state.context.mcp_services)
    tool_entries = tool_service.get_tool_entries()
    
    if not tool_entries:
        logger.warning("No MCP tools available for provisioning plan generation.")
        return mcp_resources, []

    # EXTRACT CONTEXT VARIABLES (Layer 1)
    context_vars = extract_all_context_variables(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        mcp_resources=mcp_resources,
    )
    logger.info(f"Extracted {len(context_vars)} context variables for parameter population")

    user_ctx = state.context.user_context
    context_for_scoring = user_ctx.raw_request if user_ctx else ""
    prioritized = await tool_service.select_relevant_tools(
        context_for_scoring,
        max_total=20,
        max_per_service=10,
    )
    
    # Convert tools to names
    tool_names = [tool.name for tool in prioritized]
    logger.info(f"Prioritized tools: {tool_names}")
    
    if not tool_names:
        tool_names = sorted({entry.name for entry in tool_entries.values()})

    tool_names = list(dict.fromkeys(tool_names))
    lower = {name.lower() for name in tool_names}
    if "system.wait" not in lower:
        tool_names.append("system.wait")

    resource_hints = format_resource_hints(mcp_resources)
    
    # Pass context_vars to planning
    actions = await _plan_provisioning_actions(
        state, 
        tool_names, 
        resource_hints, 
        tool_entries,
        context_vars,
    )
    
    if not actions:
        logger.info("Provisioning planner returned no actions; skipping MCP provisioning.")
        return mcp_resources, []

    assign_cb = _resource_assignment_callback(mcp_resources)
    failure_analysis, _, _, cleanup_stack = await execute_action_plan(
        actions,
        state.context.mcp_services,
        mcp_resources,
        run_label="Provisioning",
        assign_callback=assign_cb,
        require_assertion=False,
    )

    if failure_analysis.score < 1.0:
        logger.warning(
            f"Provisioning plan had issues but continuing: {failure_analysis.judge_reasoning}"
        )

    return mcp_resources, cleanup_stack


async def provision_target_agent(state: EvalAgentPlannerState) -> dict:
    if not state.context.github_context:
        raise ValueError("GitHub context is required to provision the target agent.")

    repo_url = state.context.github_context.repo_url
    branch_name = state.context.github_context.branch_name
    mcp_resources = dict(state.context.mcp_resources or {})
    _seed_default_resources(mcp_resources)

    updates: Dict[str, Any] = {}

    if not state.context.sandbox_context:
        github_token = os.getenv("GITHUB_TOKEN")
        logger.info(
            "plan.provision: provisioning sandbox (repo=%s branch=%s)",
            repo_url,
            branch_name,
        )

        sbx, repo_dir, resolved_branch = await initialize_e2b_sandbox(
            repo_url=repo_url,
            branch_name=branch_name,
            github_token=github_token,
        )
        sandbox_branch = resolved_branch or branch_name
        sandbox_id = sbx.sandbox_id

        await setup_project(sandbox_id, repo_dir, "pip install -e .")

        sandbox, _ = await deploy_server_and_confirm_ready(
            cmd=TARGET_AGENT_COMMAND,
            sb=sbx,
            cwd=repo_dir,
        )

        deployment_url = sandbox.get_host(TARGET_AGENT_PORT)
        if not deployment_url.startswith("http"):
            deployment_url = f"https://{deployment_url}"

        logger.info("plan.provision: sandbox ready at %s", deployment_url)

        sandbox_ctx = SandboxContext(
            sandbox_id=sandbox_id,
            working_directory=repo_dir,
            working_branch=sandbox_branch,
        )
        updates["sandbox_context"] = sandbox_ctx
    else:
        logger.info(
            "plan.provision: reusing existing sandbox %s",
            state.context.sandbox_context.sandbox_id,
        )

    cleanup_stack = []
    if state.context.mcp_services:
        mcp_resources, cleanup_stack = await _run_mcp_provisioning(state, mcp_resources)

    # Update the AgentContext with new sandbox and resources
    updated_context = AgentContext(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        sandbox_context=updates.get("sandbox_context", state.context.sandbox_context),
        target_agent_version=state.context.target_agent_version,
        mcp_services=state.context.mcp_services,
        mcp_resources=mcp_resources,
    )
    
    return {"context": updated_context, "cleanup_stack": cleanup_stack}
