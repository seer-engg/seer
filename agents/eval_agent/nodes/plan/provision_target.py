import os
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI

from agents.eval_agent.models import EvalAgentPlannerState
from sandbox import (
    initialize_e2b_sandbox,
    setup_project,
    deploy_server_and_confirm_ready,
)
from shared.schema import AgentContext
from shared.schema import SandboxContext
from shared.logger import get_logger
from shared.config import config
from shared.integrations.main import get_provider

logger = get_logger("eval_agent.plan")



async def _seed_default_resources(mcp_resources: Dict[str, Any], mcp_services: List[str]) -> None:
    for service in mcp_services:
        provider = await get_provider(service)
        mcp_resources[service] = provider.persistent_resource


async def provision_target_agent(state: EvalAgentPlannerState) -> dict:
    # Skip sandbox provisioning in plan-only mode (not needed for plan generation)
    if config.eval_plan_only_mode:
        logger.info("plan.provision: Plan-only mode enabled - skipping sandbox provisioning")
        mcp_resources = dict(state.context.mcp_resources or {})
        await _seed_default_resources(mcp_resources, state.context.mcp_services)
        # Create updated context with mcp_resources but no sandbox
        updated_context = AgentContext(
            user_context=state.context.user_context,
            github_context=state.context.github_context,
            sandbox_context=None,  # No sandbox in plan-only mode
            target_agent_version=state.context.target_agent_version,
            mcp_services=state.context.mcp_services,
            mcp_resources=mcp_resources,
            agent_name=state.context.agent_name,
            tool_entries=state.context.tool_entries,
        )
        return {
            "context": updated_context,
        }
    
    if not state.context.github_context:
        raise ValueError("GitHub context is required to provision the target agent.")

    repo_url = state.context.github_context.repo_url
    branch_name = state.context.github_context.branch_name
    
    # Validate repo_url is not empty
    if not repo_url or not repo_url.strip():
        raise ValueError(
            "GitHub repository URL is required but was not found in the user's message. "
            "Please include a GitHub URL in your request (e.g., 'Evaluate my agent at https://github.com/owner/repo')."
        )
    
    mcp_resources = dict(state.context.mcp_resources or {})
    await _seed_default_resources(mcp_resources, state.context.mcp_services)

    updates: Dict[str, Any] = {}

    if not state.context.sandbox_context:
        github_token = config.github_token
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
            cmd=config.target_agent_command,
            sb=sbx,
            cwd=repo_dir,
        )

        deployment_url = sandbox.get_host(config.target_agent_port)
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

    # Update the AgentContext with new sandbox and resources
    updated_context = AgentContext(
        user_context=state.context.user_context,
        github_context=state.context.github_context,
        sandbox_context=updates.get("sandbox_context", state.context.sandbox_context),
        target_agent_version=state.context.target_agent_version,
        mcp_services=state.context.mcp_services,
        mcp_resources=mcp_resources,
        agent_name=state.context.agent_name,
    )
    
    return {"context": updated_context}
